import glob, os, time, random, gc, sys, math, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from time import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from cdc.common.utils import *
from cdc.common.constants import *
from cdc.models.ensemble import EnsembleAverage
from cdc.models.pl.classifier import *
from cdc.models.pl.dataset import *
from cdc.script.utils import resize, normalize


# Averaged PL model prediction from a list of models.
def predict_pl(df, paths, tta=None, images_home=None):
    logits = []
    for path in paths:
        # Load config
        config = load_config(os.path.join(path, "pl-config.json"))
        config.images_home = images_home if images_home is not None else config.images_home
        # Image preprocessing
        preprocess = resize(config.img_size, config, p=1.0) if config.img_size is not None else None
        prepare = normalize(config.img_mean, config.img_std, config.max_pixel, p=1.0)
        # Load best weights
        pattern = "/best_epoch*.ckpt" if not config.fullfit else "last.ckpt"
        best_weights_files = glob.glob(path + pattern)
        best_weights = best_weights_files[0]
        if len(best_weights_files) > 1:
            for b in best_weights_files:
                if config.ema is not None:
                    if "EMA" in b:
                        best_weights = b
                        print("Selecting:", best_weights)
        print("Loading:", best_weights)
        if not os.path.exists(best_weights):
            raise (BaseException("Weights not found", best_weights))
        model = CDCModel(config, pretrained=None, infer=True)
        # Override the normalization
        if model.preprocessor is not None:
            prepare = model.preprocessor
            print("Override prepare stage")
        model_dump = torch.load(best_weights, map_location='cpu')
        model.load_state_dict(model_dump["state_dict"])
        model.eval()

        if tta is not None:
            device = model.device
            model = EnsembleAverage([model], tta=tta)
            model.eval()
            model.to(device)

        test_dataset = CDCDataset(df, config, mode="test", preprocess=preprocess, augment=None, prepare=prepare)
        # print("test_dataset:", len(test_dataset))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.eval_batch_size, drop_last=False,
                                                      num_workers=config.num_workers, shuffle=False, pin_memory=True)

        trainer = L.Trainer()
        logits_ = trainer.predict(model, dataloaders=test_dataloader)
        logits_ = torch.vstack(logits_).cpu().numpy()
        logits.append(logits_)
        del model, trainer, test_dataset, test_dataloader, logits_
        gc.collect()
        torch.cuda.empty_cache()

    logits = np.nanmean(np.stack(logits), axis=0)
    # print("Logits:", logits.shape)
    return logits


def infer_model(models_dict, df, images_home=None, tta=None, threshold=0.5, binary_thr_per_target=None):
    start_time = time()
    test_pd = df.copy()

    # Run all models and ensemble them
    for name, info in models_dict.items():

        ensemble_logits = []  # , ensemble_probs_binary = [], []
        root_dirs = [v for v in list(info.keys()) if "root_dir" in v]
        if len(root_dirs) < 1:
            raise(BaseException("No root_dir for %s" % info))

        for root_dir_current in root_dirs:
            # Search for folds to infer.
            root_dir = info.get(root_dir_current)
            if isinstance(root_dir, list):
                paths = []
                for root_dir_ in root_dir:
                    paths.extend(glob.glob(root_dir_ + "/fold*/"))
            else:
                paths = glob.glob(root_dir + "/fold*/")

            if len(paths) < 1:
                raise(BaseException("Nothing to infer", root_dir))

            # Ensemble of paths (if more than one), usually folds. (N, 2+)
            model_logits = predict_pl(test_pd, paths, tta=tta, images_home=images_home)
            ensemble_logits.append(model_logits)

        ensemble_logits = np.nanmean(np.stack(ensemble_logits), axis=0)

        if LABEL in name:
            # Multilabels
            probabilities = torch.sigmoid(torch.from_numpy(ensemble_logits)).numpy()
            predictions = (probabilities > threshold).astype(int)
        else:
            predictions = ensemble_logits.argmax(axis=1)

        test_idx = test_pd.index
        if LABEL in name:
            # Multilabels
            test_pd.loc[test_idx, ["preds_%d" % c for c in range(predictions.shape[1])]] = predictions
        else:
            test_pd.loc[test_idx, 'preds'] = predictions
        test_pd.loc[test_idx, ["logits_%d" % c for c in range(ensemble_logits.shape[1])]] = ensemble_logits

    end_time = time()
    print("Duration:", end_time - start_time)

    return test_pd


def execute_ensemble(models_dict, test_file, crop_test_file=None, full_images_test_home=None, boxes_images_test_home=None, inference_name=""):
    for name, info in models_dict.items():
        root_dirs = [c for c in info.keys() if c.startswith('root_dir')]
        for root_dir_name in root_dirs:
            root_dir = info.get(root_dir_name)
            if isinstance(root_dir, list):
                for root_dir_, tta in root_dir:
                    models_pl = {name: {"root_dir": root_dir_}}
                    if '-bb-multiclass' in name:
                        test_pd = pd.read_parquet(crop_test_file).reset_index(drop=True)
                        print("Executing multi-classes model:", models_pl, test_pd.shape, "TTA:", tta)
                        final_test_pd = infer_model(models_pl, test_pd, tta=tta, images_home=boxes_images_test_home)
                        final_test_pd.to_parquet(
                            os.path.join(root_dir_, "test_predictions%s.parquet" % inference_name))
                        # Debug
                        logits_col = [c for c in final_test_pd.columns if "logits_" in c]
                        if len(logits_col) > 23:
                            final_test_pd["preds"] = final_test_pd[logits_col[0:23]].values.argmax(axis=1).astype(
                                np.int32)
                        submission_csv_pd = final_test_pd[
                            ["trustii_id", "NAME", "pred_x1", "pred_y1", "pred_x2", "pred_y2", "preds"]].copy().rename(
                            columns={'pred_x1': 'x1', 'pred_y1': 'y1', 'pred_x2': 'x2', 'pred_y2': 'y2',
                                     'preds': 'class'})
                        submission_csv_pd["x1"] = submission_csv_pd["x1"].astype(np.int32)
                        submission_csv_pd["y1"] = submission_csv_pd["y1"].astype(np.int32)
                        submission_csv_pd["x2"] = submission_csv_pd["x2"].astype(np.int32)
                        submission_csv_pd["y2"] = submission_csv_pd["y2"].astype(np.int32)
                        submission_csv_pd["class"] = submission_csv_pd["class"].astype(np.int32)
                        submission_csv_pd["class"] = submission_csv_pd["class"].map(class_mapping)
                        submission_csv_pd = pd.merge(pd.read_csv(test_file), submission_csv_pd,
                                                     on=["trustii_id", "NAME"], how="left")
                        submission_csv_pd.to_csv(os.path.join(root_dir_, "submission%s.csv" % inference_name),
                                                 index=False)
                    elif '-multilabel' in name:
                        test_pd = pd.read_csv(test_file)  # .head(512)
                        test_pd["filename"] = test_pd["NAME"]
                        print("Executing multi-labels model:", models_pl, test_pd.shape, "TTA:", tta)
                        final_test_pd = infer_model(models_pl, test_pd, tta=tta, images_home=full_images_test_home)
                        final_test_pd.to_parquet(
                            os.path.join(root_dir_, "test_predictions%s.parquet" % inference_name))
            else:
                raise (Exception("List expected:%s" % root_dir))
