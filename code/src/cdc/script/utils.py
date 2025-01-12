import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import seaborn as sns
from matplotlib import pyplot as plt

from cdc.common.constants import LABEL, class_mapping
from cdc.yolo.inference import run_wbf


def resize(new_size, conf, p=1.0):
    interpolation = getattr(conf, "interpolation", cv2.INTER_LANCZOS4)
    if conf.ar is None:
        return A.Compose([
            A.Resize(new_size, new_size, interpolation=interpolation, p=1.0, always_apply=True),
        ], p=p)
    elif conf.ar == -1:
        return A.Compose([
            A.NoOp(),
        ], p=p)


def normalize(mean, std, max_pixel, p=1.0):
    return A.Compose([

        A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel, p=1.0, always_apply=True),
        ToTensorV2(p=1.0, always_apply=True)

    ], p=p)


def ensemble_multiclass_multilabels(multiclass_single_wbc_pd, multilabels_pd, weights=[0.5, 0.5]):
    # Move from logits to probabilities (softmax because of multiclasses) - Need to be followed by argmax
    logits_col = [c for c in multiclass_single_wbc_pd.columns if "logits_" in c][0:23]
    sprobs_col = [c.replace("logits_", "sprobs_") for c in multiclass_single_wbc_pd.columns if "logits_" in c][0:23]
    multiclass_single_wbc_pd[sprobs_col] = torch.softmax(torch.from_numpy(multiclass_single_wbc_pd[logits_col].values),
                                                         dim=1).numpy()
    sprobs_pd = multiclass_single_wbc_pd[["NAME"] + sprobs_col]
    # display(sprobs_pd)

    # Move from logits to probabilities (sigmoid because of multilabels) - Need to be followed by threshold
    logits_col = [c for c in multilabels_pd.columns if "logits_" in c][0:23]
    mprobs_col = [c.replace("logits_", "mprobs_") for c in multilabels_pd.columns if "logits_" in c][0:23]
    multilabels_pd[mprobs_col] = torch.sigmoid(torch.from_numpy(multilabels_pd[logits_col].values)).numpy()
    mprobs_pd = multilabels_pd[["NAME"] + mprobs_col]
    # display(mprobs_pd)

    # Merge both on NAME and ensemble with average
    probs_pd = pd.merge(sprobs_pd, mprobs_pd, on="NAME", how="inner")
    # print(probs_pd.shape)
    # print(probs_pd[sprobs_col].shape, probs_pd[mprobs_col].shape)
    ensemble_probs = [probs_pd[sprobs_col].values, probs_pd[mprobs_col].values]
    # ensemble_probs = [probs_pd[mprobs_col].values]
    # print(probs_pd[sprobs_col].values.shape, probs_pd[mprobs_col].values.shape)
    # print("Ensemble of:", np.stack(ensemble_probs).shape, weights)
    # ensemble_probs = np.nanmean(np.stack(ensemble_probs), axis=0)
    ensemble_probs = np.average(np.stack(ensemble_probs), axis=0, weights=weights)

    # print(ensemble_probs.shape)
    probs_col = [c.replace("sprobs_", "eprobs_") for c in probs_pd.columns if "sprobs_" in c]
    probs_pd[probs_col] = ensemble_probs
    # display(probs_pd[["NAME"] + probs_col])

    return probs_pd


def ensemble_oof(models_dict):
    verbose = True
    for name, info in models_dict.items():
        root_dirs = [c for c in info.keys() if c.startswith('root_dir')]
        root_weights = [float(x.split("/")[-1]) for x in root_dirs]
        root_dirs_ensemble_logits = []
        for root_dir_name in root_dirs:
            if '-ft' in root_dir_name:
                continue
            root_dir = info.get(root_dir_name)
            if isinstance(root_dir, list):
                ensemble_logits = []
                for root_dir_, aug in root_dir:
                    if '-ft' in root_dir_:
                        continue
                    oof_pd_ = pd.read_parquet(os.path.join(root_dir_, "oof_df.parquet"))
                    if "src" in oof_pd_.columns:
                        oof_pd_ = oof_pd_[~(oof_pd_["src"].astype(str).str.contains('background'))]
                    logits_col = [c for c in oof_pd_.columns if "logits_" in c][0:23]
                    # if 'bb-multiclass' in name:
                    #     print("Loading bb-multiclass:", root_dir_, oof_pd_[logits_col].shape)
                    ensemble_logits.append(oof_pd_[logits_col].values)
                ensemble_logits = np.nanmean(np.stack(ensemble_logits), axis=0)
                # Ensemble OOF with new logits and predictions
                oof_pd = oof_pd_.copy()
                oof_pd[logits_col] = ensemble_logits
                if LABEL in name:
                    oof_pd[[c for c in oof_pd.columns if "preds_" in c][0:23]] = (
                            torch.sigmoid(torch.from_numpy(ensemble_logits)).numpy() > 0.5).astype(int)
                else:
                    oof_pd["preds"] = ensemble_logits.argmax(1)
            else:
                oof_pd = pd.read_parquet(os.path.join(root_dir, "oof_df.parquet"))
                if "src" in oof_pd.columns:
                    oof_pd = oof_pd[~(oof_pd["src"].astype(str).str.contains('background'))]
                logits_col = [c for c in oof_pd.columns if "logits_" in c][0:23]
                oof_pd["preds"] = oof_pd[logits_col].values.argmax(1)

            root_dirs_ensemble_logits.append(oof_pd[logits_col].values)

            # display(oof_pd.head())
            # Check F1 score
            preds_col = [c for c in oof_pd.columns if "preds_" in c][0:23] if LABEL in name else "preds"
            gt = np.vstack(oof_pd[LABEL].values).astype(int)[:, 0:23] if LABEL in name else oof_pd["class"].values
            preds = oof_pd[preds_col].astype(int).values
            f1 = f1_score(gt, preds, average="macro")
            print("Loading:", root_dir_name, root_dir, oof_pd.shape, "F1=%.4f" % f1) if verbose else None
            info["oof_%s" % root_dir_name] = f1

        root_dirs_ensemble_logits = np.average(np.stack(root_dirs_ensemble_logits), axis=0, weights=root_weights)

        oof_pd[logits_col] = root_dirs_ensemble_logits
        if LABEL in name:
            oof_pd[[c for c in oof_pd.columns if "preds_" in c][0:23]] = (
                    torch.sigmoid(torch.from_numpy(root_dirs_ensemble_logits)).numpy() > 0.5).astype(int)
        else:
            oof_pd["preds"] = oof_pd[logits_col].values.argmax(1)
        gt = np.vstack(oof_pd[LABEL].values).astype(int)[:, 0:23] if LABEL in name else oof_pd["class"].values
        preds_col = [c for c in oof_pd.columns if "preds_" in c][0:23] if LABEL in name else "preds"
        f1 = f1_score(gt, oof_pd[preds_col].values, average="macro")
        print("Ensemble(x%d):" % len(root_dirs), name, oof_pd.shape, "F1=%.4f" % f1, "weights:",
              root_weights) if verbose else None
        info["oof"] = f1
        info["oof_pd"] = oof_pd


def mc_ml_ensemble_oof(models_dict, dump_folder=None, inference_name="", alpha=0.5, thr=0.5):
    multilabels_pd_ = models_dict['cnn_and_transformers-multilabel']["oof_pd"]  # Image level
    multiclass_pd_ = models_dict['cnn_and_transformers-bb-multiclass']["oof_pd"]  # Box level
    multiclass_pd_["wbc"] = multiclass_pd_.groupby(["NAME"])["trustii_id"].transform('count')
    multiclass_pd_["preds_unique"] = multiclass_pd_.groupby(["NAME"])["preds"].transform("nunique")
    multiclass_single_wbc_pd_ = multiclass_pd_[multiclass_pd_["wbc"] == 1].reset_index(drop=True)
    print("Images with single WBC", multiclass_single_wbc_pd_.shape, "%.2f%%"%(multiclass_single_wbc_pd_.shape[0]*100/multiclass_pd_.shape[0]), "F1: %.4f" % (f1_score(multiclass_single_wbc_pd_["class"].values, multiclass_single_wbc_pd_["preds"].astype(int).values, average="macro")))
    ensemble_probs_pd = ensemble_multiclass_multilabels(multiclass_single_wbc_pd_, multilabels_pd_, weights=[alpha, 1-alpha])
    eprobs_col = [c for c in ensemble_probs_pd.columns if "eprobs_" in c]
    # Ensemble ML + MC for single WBC per image
    multiclass_single_wbc_ensemble_pd = pd.merge(multiclass_single_wbc_pd_, ensemble_probs_pd[["NAME"] + eprobs_col], on="NAME", how="inner")
    # Merge back to get final OOF
    all_classes = sorted(multiclass_single_wbc_ensemble_pd["class"].unique())
    multiclass_single_wbc_ensemble_pd["preds_ensemble_argmax"] = multiclass_single_wbc_ensemble_pd[eprobs_col].values.argmax(axis=1)
    ensemble_multiclass_pd = pd.merge(multiclass_pd_, multiclass_single_wbc_ensemble_pd[["NAME", "preds_ensemble_argmax"]], on='NAME', how='left')
    ensemble_multiclass_pd.loc[ensemble_multiclass_pd["wbc"] > 1, "preds_ensemble_argmax"] = ensemble_multiclass_pd["preds"]
    ensemble_multiclass_pd["preds_ensemble_argmax"] = ensemble_multiclass_pd["preds_ensemble_argmax"].astype(np.int32)
    s = f1_score(ensemble_multiclass_pd["class"].values, ensemble_multiclass_pd["preds_ensemble_argmax"].values, average="macro")
    return ensemble_multiclass_pd, s


def ensemble_test(models_dict, inference_name="", thr=0.5):
    print(inference_name)
    for name, info in models_dict.items():
        root_dirs = [c for c in info.keys() if c.startswith('root_dir')]
        root_weights = [float(x.split("/")[-1]) for x in root_dirs]
        root_dirs_ensemble_logits = []
        for root_dir_name in root_dirs:
            root_dir = info.get(root_dir_name)
            if isinstance(root_dir, list):
                ensemble_logits = []
                for root_dir_, aug in root_dir:
                    test_pd_ = pd.read_parquet(os.path.join(root_dir_, "test_predictions%s.parquet" % inference_name))
                    logits_col = [c for c in test_pd_.columns if "logits_" in c][0:23]
                    ensemble_logits.append(test_pd_[logits_col].values)
                ensemble_logits = np.nanmean(np.stack(ensemble_logits), axis=0)
                # Ensemble OOF with new logits and predictions
                test_pd = test_pd_.copy()
                test_pd[logits_col] = ensemble_logits
                if LABEL in name:
                    test_pd[[c for c in test_pd.columns if "preds_" in c][0:23]] = (
                                torch.sigmoid(torch.from_numpy(ensemble_logits)).numpy() > thr).astype(int)
                else:
                    test_pd["preds"] = ensemble_logits.argmax(1)

            root_dirs_ensemble_logits.append(test_pd[logits_col].values)
            print("Loading:", root_dir_name, root_dir, test_pd.shape)

        root_dirs_ensemble_logits = np.average(np.stack(root_dirs_ensemble_logits), axis=0, weights=root_weights)

        test_pd[logits_col] = root_dirs_ensemble_logits
        if LABEL in name:
            test_pd[[c for c in test_pd.columns if "preds_" in c][0:23]] = (
                        torch.sigmoid(torch.from_numpy(root_dirs_ensemble_logits)).numpy() > 0.5).astype(int)
        else:
            test_pd["preds"] = test_pd[logits_col].values.argmax(1)

        if LABEL in name:
            test_pd = test_pd.drop(columns=["trustii_id"]).groupby(["NAME", "filename"]).first().reset_index()

        print("Ensemble(x%d):" % len(root_dirs), name, test_pd.shape, root_weights)
        # display(test_pd.head())
        info["test_pd"] = test_pd


def mc_ml_ensemble_test(models_dict, test_file, dump_folder, inference_name="", alpha=0.5, thr=0.5):
    multilabels_pd_ = models_dict['cnn_and_transformers-multilabel']["test_pd"]  # Image level
    multiclass_pd_ = models_dict['cnn_and_transformers-bb-multiclass']["test_pd"]  # Box level
    multiclass_pd_["wbc"] = multiclass_pd_.groupby(["NAME"])["trustii_id"].transform('count')
    multiclass_single_wbc_pd_ = multiclass_pd_[multiclass_pd_["wbc"] == 1].reset_index(drop=True)
    print("Images with single WBC", multiclass_single_wbc_pd_.shape,
          "%.2f%%" % (multiclass_single_wbc_pd_.shape[0] * 100 / multiclass_pd_.shape[0]))

    ensemble_probs_pd = ensemble_multiclass_multilabels(multiclass_single_wbc_pd_, multilabels_pd_,
                                                        weights=[alpha, 1 - alpha])
    eprobs_col = [c for c in ensemble_probs_pd.columns if "eprobs_" in c]
    multiclass_single_wbc_ensemble_pd = pd.merge(multiclass_single_wbc_pd_, ensemble_probs_pd[["NAME"] + eprobs_col],
                                                 on="NAME", how="inner")
    mpreds = (multiclass_single_wbc_ensemble_pd[eprobs_col].values > thr).astype(int)
    print("Threshold %.2f predictions has %d WBC with multilabels" % (thr, np.sum(np.sum(mpreds, axis=1) > 1)))
    multiclass_single_wbc_ensemble_pd["preds_ensemble_argmax"] = multiclass_single_wbc_ensemble_pd[
        eprobs_col].values.argmax(axis=1)
    multiclass_single_wbc_ensemble_pd["preds"] = multiclass_single_wbc_ensemble_pd["preds"].astype(np.int32)
    changes_argmax_pd = multiclass_single_wbc_ensemble_pd[
        multiclass_single_wbc_ensemble_pd["preds"] != multiclass_single_wbc_ensemble_pd["preds_ensemble_argmax"]][
        ["NAME", "preds", "preds_ensemble_argmax"]]
    print("Changes argmax (%d): %.2f%%" % (
    changes_argmax_pd.shape[0], changes_argmax_pd.shape[0] * 100 / multiclass_single_wbc_ensemble_pd.shape[0]))
    # Merge back to get final OOF
    ensemble_multiclass_pd = pd.merge(multiclass_pd_,
                                      multiclass_single_wbc_ensemble_pd[["NAME", "preds_ensemble_argmax"] + eprobs_col],
                                      on='NAME', how='left')
    ensemble_multiclass_pd.loc[ensemble_multiclass_pd["wbc"] > 1, "preds_ensemble_argmax"] = ensemble_multiclass_pd[
        "preds"]
    ensemble_multiclass_pd["preds_ensemble_argmax"] = ensemble_multiclass_pd["preds_ensemble_argmax"].astype(np.int32)
    # ensemble_multiclass_pd.loc[ensemble_multiclass_pd["wbc"] > 1, "preds_ensemble_thr"] = ensemble_multiclass_pd["preds"]
    # ensemble_multiclass_pd["preds_ensemble_thr"] = ensemble_multiclass_pd["preds_ensemble_thr"].astype(np.int32)
    # display(ensemble_multiclass_pd[["trustii_id", "NAME", "pred_x1", "pred_y1", "pred_x2", "pred_y2", "preds_ensemble_argmax"]])

    # Submission file (argmax)
    submission_csv_pd = ensemble_multiclass_pd[
        ["trustii_id", "NAME", "pred_x1", "pred_y1", "pred_x2", "pred_y2", "preds_ensemble_argmax"]].copy().rename(
        columns={'pred_x1': 'x1', 'pred_y1': 'y1', 'pred_x2': 'x2', 'pred_y2': 'y2', 'preds_ensemble_argmax': 'class'})
    submission_csv_pd["class"] = submission_csv_pd["class"].astype(np.int32)
    submission_csv_pd["class"] = submission_csv_pd["class"].map(class_mapping)
    # Keep same order
    submission_csv_pd = pd.merge(pd.read_csv(test_file), submission_csv_pd, on=["trustii_id", "NAME"], how="left")
    os.makedirs(dump_folder, exist_ok=True)
    submission_csv_pd.to_csv(os.path.join(dump_folder, "submission_argmax%s.csv" % inference_name), index=False)
    ensemble_multiclass_pd.to_parquet(os.path.join(dump_folder, "ensemble_argmax%s.parquet" % inference_name))
    return ensemble_multiclass_pd, submission_csv_pd


def extract_predictions(test_pd, test_file, confidence=0.920, verbose=False, ignore=None, files=None):
    submission_pd = pd.read_csv(test_file)
    submission_pd["expected_bbs"] = submission_pd.groupby(["NAME"])["trustii_id"].transform('count')
    submission_pd = submission_pd.drop(columns=['trustii_id'])
    test_pd = pd.merge(test_pd, submission_pd, on="NAME", how="inner")
    # Keep single cell
    test_pd = test_pd[(test_pd["wbc"] == 1) & (test_pd["expected_bbs"] == 1)]
    if ignore is not None:
        test_pd = test_pd[~test_pd["trustii_id"].isin(ignore)]
    # Extract predictions probabilities
    test_pd["preds_probs"] = test_pd[[c for c in test_pd.columns if "eprobs_" in c]].values.max(axis=1)
    test_pd = test_pd[["NAME", "filename", "img_width", "img_height", "predict_bb", "trustii_id", "pred_x1", "pred_y1", "pred_x2", "pred_y2", "pred_width", "pred_height", "predict_score_avg", "pred_score", "preds", "preds_probs", "wbc"]]
    test_pd["score"] = test_pd["predict_score_avg"]*test_pd["preds_probs"]
    test_pd["class_x"] = test_pd["preds"].map(class_mapping)
    if verbose:
        print(test_pd.shape)
        fig, ax = plt.subplots(1, 4, figsize=(32,5))
        d = sns.scatterplot(data=test_pd, x="pred_score", y="preds_probs", ax=ax[0])
        d = sns.histplot(data=test_pd, x="pred_score", ax=ax[1])
        d = sns.histplot(data=test_pd, x="preds_probs", ax=ax[1])
        d = sns.boxplot(data=test_pd, x="preds_probs", ax=ax[2])
        d = sns.countplot(data=test_pd, x="class_x", ax=ax[3])
        d = ax[3].tick_params(axis='x', labelrotation=90)
    if files is not None:
        test_pseudo_pd = test_pd[test_pd["NAME"].isin(files)].reset_index(drop=True)
    else:
        test_pseudo_pd = test_pd[test_pd["score"] > confidence].reset_index(drop=True)
    if verbose:
        print(test_pseudo_pd.shape)
        fig, ax = plt.subplots(1, 4, figsize=(32, 5))
        d = sns.scatterplot(data=test_pseudo_pd, x="pred_score", y="preds_probs", ax=ax[0])
        d = sns.histplot(data=test_pseudo_pd, x="pred_score", ax=ax[1])
        d = sns.histplot(data=test_pseudo_pd, x="preds_probs", ax=ax[1])
        d = sns.boxplot(data=test_pseudo_pd, x="preds_probs", ax=ax[2])
        d = sns.countplot(data=test_pseudo_pd, x="class_x", ax=ax[3])
        d = ax[3].tick_params(axis='x', labelrotation=90)
    return test_pseudo_pd


def extract_predictions_extended(ensemble_pd, bbx_raw_file, wbf_iou=0.25, skip_box_thr=0.010, surface_ratio=8.946, wbf_score=0.94, score=0.920, files=None):
    test_pseudo_pd_ = ensemble_pd.copy()
    test_pseudo_pd_["filename"] = test_pseudo_pd_["NAME"].apply(lambda x: x.split("/")[-1])
    if files is not None:
        test_pseudo_full_pd = test_pseudo_pd_[test_pseudo_pd_["NAME"].isin(files)]
        test_pseudo_full_pd = test_pseudo_full_pd.reset_index(drop=True)
    else:
        raw_pd = pd.read_parquet(bbx_raw_file)
        print("Raw boxes:", raw_pd.shape)
        # Keep filtered images
        raw_pd = raw_pd[raw_pd["filename"].isin(test_pseudo_pd_["filename"].unique())]
        print("Raw boxes filtered:", raw_pd.shape)
        # Run WBF with lower threshold
        roi_wbf = run_wbf(raw_pd, iou_thr=wbf_iou, skip_box_thr=skip_box_thr)
        roi_wbf["bbx_xtl"] = roi_wbf["bbx_xtl"].apply(lambda x: np.round(x)).astype(np.int32)
        roi_wbf["bbx_xbr"] = roi_wbf["bbx_xbr"].apply(lambda x: np.round(x)).astype(np.int32)
        roi_wbf["bbx_ybr"] = roi_wbf["bbx_ybr"].apply(lambda x: np.round(x)).astype(np.int32)
        roi_wbf["bbx_ytl"] = roi_wbf["bbx_ytl"].apply(lambda x: np.round(x)).astype(np.int32)
        roi_wbf["roi_width"] = roi_wbf["bbx_xbr"] - roi_wbf["bbx_xtl"]
        roi_wbf["roi_height"] = roi_wbf["bbx_ybr"] - roi_wbf["bbx_ytl"]
        roi_wbf["roi_surface"] = roi_wbf["roi_width"]*roi_wbf["roi_height"]
        roi_wbf["roi_surface_ratio"] = roi_wbf["roi_surface"]*100./(roi_wbf["slide_width"]*roi_wbf["slide_height"])
        roi_wbf["predict_bb"] = roi_wbf[["bbx_xtl", "bbx_xbr", "bbx_ytl", "bbx_ybr", "class", "roi_surface_ratio", "score"]].apply(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6]), axis=1)
        roi_wbf["cells"] = roi_wbf.groupby(["filename"])["predict_bb"].transform('count')
        print("After WBF:", roi_wbf.shape)
        # Drop images with cells > 1 or small cells
        drop_pd = roi_wbf[(roi_wbf["cells"] > 1) | (roi_wbf["roi_surface_ratio"] < surface_ratio) | (roi_wbf["score"] < wbf_score)]
        print("Drop:", drop_pd.shape)
        test_pseudo_pd_ = test_pseudo_pd_[~(test_pseudo_pd_["filename"].isin(drop_pd["filename"].unique()))]
        print("Filtered WBF:", test_pseudo_pd_.shape)
        # Keep high confidence
        test_pseudo_pd_ = test_pseudo_pd_[(test_pseudo_pd_["score"] >= score)]
        test_pseudo_full_pd = test_pseudo_pd_.reset_index(drop=True)
        print("Filtered:", test_pseudo_full_pd.shape)
    return test_pseudo_full_pd
