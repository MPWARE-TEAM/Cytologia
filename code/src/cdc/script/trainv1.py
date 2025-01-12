import glob, os, gc
import wandb
from cdc.common.utils import *
from cdc.common.constants import class_mapping, class_names
from cdc.common.metrics import *
from cdc.models.pl.classifier import *
from cdc.models.pl.dataset import *
from cdc.script.inferv1 import infer_model
from cdc.script.utils import resize, normalize
from cdc.utils.imaging import *
from cdc.yolo.tools import *
import albumentations as A
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
import wandb
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything


def ohe_multilabel(conf, multilabel_str):
  multilabels = np.zeros(conf.num_labels, dtype=np.float32)
  if len(multilabel_str) > 0:
    labels = multilabel_str.split("|")
    for label in labels:
      multilabels[int(label)] = 1.0
  return multilabels


# Extract crops from background images
def background_preprocessing_train(conf, p=1.0):
    return A.Compose([
        RandomUnsizedCrop([88, 224], h_w_location=None, p=1.0, always_apply=True),
        # Based on q1,q3 of BB dimensions
    ], p=p)


def background_preprocessing_valid(conf, p=1.0):
    return A.Compose([
        RandomUnsizedCrop([146, 146], h_w_location=[0.05, 0.05], p=1.0, always_apply=True),
        # Based on q2 of BB dimensions
    ], p=p)


class Rotate90_270(A.RandomRotate90):
    def get_params(self):
        return {"factor": random.choice([1, 3])}


def soft_augmentation_train(conf, p=1.0):
    return A.Compose([

        # Flips/Rotate
        A.OneOf([
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            Rotate90_270(p=0.50)],
            p=0.75),

        # Colors/Channels
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.25, brightness_by_max=True, p=0.25),
            A.RandomGamma(gamma_limit=(80, 110), p=0.25),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=5, p=0.25),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.25),
        ], p=0.20),

        # Blur/Noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.33),
            A.MotionBlur(blur_limit=(3, 5), p=0.33),
            A.CoarseDropout(max_holes=3, max_height=conf.img_size//8, max_width=conf.img_size//8, fill_value=0, p=0.33),
        ], p=0.15),

    ], p=p)


def light_augmentation_train(conf, p=1.0):
    return A.Compose([

        # Flips/Rotate
        A.OneOf([
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            Rotate90_270(p=0.50)],
            p=0.75),

        # ShiftScaleRotate
        A.ShiftScaleRotate(shift_limit=(-0.10, 0.10), scale_limit=(-0.2, 0.2), rotate_limit=(-30, 30),
                           interpolation=cv2.INTER_LINEAR, border_mode=0, value=(0, 0, 0), p=0.50),

        # Colors/Channels
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, brightness_by_max=True, p=0.16),
            A.RandomGamma(gamma_limit=(80, 110), p=0.16),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=20, p=0.16),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.16),
            A.ToGray(p=0.05),
            A.CLAHE(p=0.16),
        ], p=0.30),

        # Blur/Noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.45),
            A.MotionBlur(blur_limit=(3, 5), p=0.45),
            A.GaussNoise(var_limit=(0, 2.0), mean=0, p=0.10),
        ], p=0.10),

        # Cutout
        A.CoarseDropout(max_holes=3, max_height=conf.img_size // 6, max_width=conf.img_size // 6, fill_value=0, p=0.25),

    ], p=p)


# Too much colors/gamma and noise can be counterproductive during training.
def medium_augmentation_train(conf, p=1.0):
    return A.Compose([

        # Flips/Rotate
        A.OneOf([
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            Rotate90_270(p=0.50)],
            p=0.75),

        # ShiftScaleRotate
        A.ShiftScaleRotate(shift_limit=(-0.10, 0.10), scale_limit=(-0.2, 0.2), rotate_limit=(-30, 30),
                           interpolation=cv2.INTER_LINEAR, border_mode=0, value=(0, 0, 0), p=0.50),

        # Colors/Channels
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, brightness_by_max=True, p=0.16),
            A.RandomGamma(gamma_limit=(80, 120), p=0.16),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=20, p=0.16),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.16),
            A.ToGray(p=0.10),
            A.CLAHE(p=0.16),
        ], p=0.5),

        # Blur/Noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.33),
            A.MotionBlur(blur_limit=(3, 5), p=0.33),
            A.GaussNoise(var_limit=(0, 50.0), mean=0, p=0.33),
        ], p=0.5),

        # Cutout
        A.CoarseDropout(max_holes=3, max_height=conf.img_size // 8, max_width=conf.img_size // 8, fill_value=0, p=0.25),

    ], p=p)


def train_and_infer(config_file, images_home, train_file, test_file, test_boxes_file, test_boxes_home, train_augmentation,
                    background_home=None, background_file=None, tta=None, tta_name="_notta",
                    train_epochs=None, report_to=None):

    # Load config
    config = load_config(config_file)
    config.train_epochs = train_epochs if train_epochs is not None else config.train_epochs
    config.images_home = images_home if images_home is not None else config.images_home
    config.report_to = report_to if report_to is not None else config.report_to
    os.makedirs(config.default_root_dir, exist_ok=True)

    train_and_infer_model(config, images_home, train_file, test_file, test_boxes_file, test_boxes_home, train_augmentation,
                          background_home=background_home, background_file=background_file, tta=tta, tta_name=tta_name)


def train_and_infer_model(config, images_home, train_file, test_file, test_boxes_file, test_boxes_home, train_augmentation,
                          background_home=None, background_file=None, tta=None, tta_name="_notta"):

    # Load data with cross-validation split
    print("Loading:", train_file[0] if isinstance(train_file, list) else train_file)
    train_pd = pd.read_parquet(train_file[0] if isinstance(train_file, list) else train_file)
    print("Train", train_pd.shape)
    config.num_labels = train_pd[config.label_col].nunique() if config.label_col not in [LABEL] else len(class_mapping)

    # Add optional background images with a new class (23)
    if background_home is not None:
        background_pd = pd.read_parquet(background_file)
        config.num_labels = config.num_labels + 1
        background_pd["filename"] = background_pd["filename"].apply(lambda x: "background/" + x)
        background_pd[config.label_col] = "23" if config.label_col == LABEL else 23
        class_mapping[23] = 'Background'
        class_names.extend(["Background"])
        print("Background:", background_pd.shape)
        train_pd = pd.concat([train_pd, background_pd], ignore_index=True)
        print(train_pd.shape)

    # Add optional pseudo labels
    pseudo_pd = None
    if isinstance(train_file, list):
        pseudo_pd = []
        for file in train_file[1:]:
            pseudo_pd_ = pd.read_parquet(file)
            pseudo_pd.append(pseudo_pd_)
        pseudo_pd = pd.concat(pseudo_pd).reset_index(drop=True)
        pseudo_pd["class"] = pseudo_pd["preds"].astype(np.int32)
        if config.label_col == LABEL:
            pseudo_pd[LABEL] = pseudo_pd["class"].astype(str)
            pseudo_pd["multilabel_str"] = pseudo_pd["multilabel"]
            pseudo_pd[LABEL] = pseudo_pd["multilabel_str"].apply(lambda x: ohe_multilabel(config, x))

    if config.label_col == LABEL:
        train_pd["multilabel_str"] = train_pd["multilabel"]
        train_pd[LABEL] = train_pd["multilabel_str"].apply(lambda x: ohe_multilabel(config, x))

    # Resize input image
    preprocess_image = resize(config.img_size, config, p=1.0) if config.img_size is not None else None
    # Training augmentation
    image_augmentation_train = train_augmentation(config, p=1.0)
    # Optional background augmentation
    background_preprocess_train = background_preprocessing_train(config, p=1.0) if (background_home is not None) and (config.label_col == "class") else None
    background_preprocess_valid = background_preprocessing_valid(config, p=1.0) if (background_home is not None) and (config.label_col == "class") else None
    # Normalize and move to tensor
    prepare_feed = normalize(config.img_mean, config.img_std, config.max_pixel, p=1.0) if "foundation_" not in config.backbone else None

    # Train loop
    resume_fold = 0
    seed_ = config.folds_seed
    for fold in config.folds:
        if fold < resume_fold:
            continue

        if config.fullfit:
            fold = 99
            train, valid = train_pd, None
        else:
            train_idx = train_pd[train_pd["fold_s%d" % seed_] != fold].index
            valid_idx = train_pd[train_pd["fold_s%d" % seed_] == fold].index
            train, valid = train_pd.loc[train_idx], train_pd.loc[valid_idx]

        if pseudo_pd is not None:
            print("Finetuning with pseudo labels:", pseudo_pd.shape)
            train = pd.concat([train, pseudo_pd], ignore_index=True).sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

        root_dir = os.path.join(config.default_root_dir, "seed%d" % config.seed)
        output_dir = os.path.join(root_dir, "fold%d" % fold)
        os.makedirs(output_dir, exist_ok=True)

        model = CDCModel(config, pretrained=None)
        if model.preprocessor is not None:
            prepare_feed = model.preprocessor
        if config.finetune:
            pattern = "/best_epoch*.ckpt" if not config.fullfit else "last.ckpt"
            best_weights_files_ = glob.glob(output_dir.replace(config.stage, "stage%d" % (int(config.stage[-1]) - 1)) + pattern)
            best_weights_ = os.path.basename(best_weights_files_[0])
            # Load model's weights from previous stage
            model_weights = os.path.join(output_dir.replace(config.stage, "stage%d" % (int(config.stage[-1]) - 1)), best_weights_)  # "best_model.ckpt"
            print("Update pretrained model:", model_weights)
            model_dump = torch.load(model_weights, map_location='cpu')
            model.load_state_dict(model_dump["state_dict"])

        train_dataset = CDCDataset(train, config, mode="train", background_preprocess=background_preprocess_train,
                                   preprocess=preprocess_image, augment=image_augmentation_train, prepare=prepare_feed)
        valid_dataset = CDCDataset(valid, config, mode="valid", background_preprocess=background_preprocess_valid,
                                   preprocess=preprocess_image, augment=None,
                                   prepare=prepare_feed) if valid is not None else None

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                                       drop_last=False, num_workers=config.num_workers, shuffle=True,
                                                       pin_memory=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.eval_batch_size,
                                                       drop_last=False, num_workers=config.num_workers, shuffle=False,
                                                       pin_memory=True) if valid_dataset is not None else None

        monitor = "val_f1" if valid_dataloader is not None else "train_step_f1"
        logger_wandb = WandbLogger(project=config.report_name,
                                   name=output_dir.replace("/", "_")) if config.report_to != "none" else None
        logger_csv = CSVLogger("./logs", name=output_dir.replace("/", "_"))
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(dirpath=output_dir, monitor=monitor, mode='max',
                                              filename='best_{epoch}-{%s:.4f}' % monitor, save_top_k=1,
                                              save_last=config.fullfit, save_weights_only=True)
        callbacks = [lr_monitor]
        callbacks.extend([checkpoint_callback])

        d = save_config(config, os.path.join(output_dir, "pl-config.json"))
        print("Fold:", fold, "Train:", train.shape, "Valid:", valid.shape if valid is not None else None,
              model.metric_avg, "LR", config.lr, "Loss", model.loss)

        trainer = L.Trainer(
            default_root_dir=output_dir,
            max_epochs=config.train_epochs,
            accelerator=config.device,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            deterministic=config.deterministic,
            precision=config.precision,
            logger=logger_wandb if logger_wandb is not None else logger_csv,
            callbacks=callbacks,
            val_check_interval=config.eval_steps if config.evaluation_strategy == "steps" else 1.0,
            enable_progress_bar=True,
        )

        trainer.fit(model, train_dataloader, valid_dataloader)

        if logger_wandb is not None:
            wandb.finish()

        # Reload best and evaluate
        pattern = "/best_epoch*.ckpt" if not config.fullfit else "last.ckpt"
        best_weights_files = glob.glob(output_dir + pattern)
        best_weights = best_weights_files[0]
        if len(best_weights_files) > 1:
            for b in best_weights_files:
                if config.ema is not None:
                    if "EMA" in b:
                        best_weights = b
                        print("Selecting:", best_weights)
                    else:
                        os.remove(b)

        print("Loading:", best_weights)
        if os.path.exists(best_weights) and valid is not None:
            model = CDCModel(config, pretrained=None, infer=True)
            model_dump = torch.load(best_weights, map_location='cpu')
            model.load_state_dict(model_dump["state_dict"])
            model.eval()

            test_dataset = CDCDataset(valid, config, mode="test", background_preprocess=background_preprocess_valid,
                                      preprocess=preprocess_image, augment=None, prepare=prepare_feed)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.eval_batch_size,
                                                          drop_last=False, num_workers=config.num_workers,
                                                          shuffle=False, pin_memory=True)

            trainer = L.Trainer()
            logits = trainer.predict(model, dataloaders=test_dataloader)
            logits = torch.vstack(logits).cpu().numpy()

            if config.label_col == LABEL:
                # Multilabels
                probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
                predictions = (probabilities > config.probs_threshold).astype(int)
            else:
                predictions = logits.argmax(axis=1)

            if config.label_col == LABEL:
                train_pd.loc[valid_idx, ["preds_%d" % c for c in range(config.num_labels)]] = predictions
            else:
                train_pd.loc[valid_idx, 'preds'] = predictions
            train_pd.loc[valid_idx, ["logits_%d" % c for c in range(config.num_labels)]] = logits
            train_pd.loc[valid_idx].to_parquet(f'{output_dir}/valid_df.parquet')

            preds_col = [c for c in train_pd.columns if "preds_" in c] if config.label_col == LABEL else "preds"
            preds = train_pd.loc[valid_idx][preds_col].astype(int).values
            gt = np.vstack(train_pd.loc[valid_idx][config.label_col].values).astype(int) if config.label_col == LABEL else train_pd.loc[valid_idx][config.label_col].values
            f1 = f1_score(gt, preds, average=model.metric_avg)

            print("OOF F1:", f1)

            with open(os.path.join(f'{output_dir}', "eval.json"), "w") as f:
                json.dump({"f1": f1}, f)

        if config.half:
            print("Half quantization")
            model.half()
            torch.save(model.state_dict(), best_weights)

        del trainer, model

        gc.collect()
        torch.cuda.empty_cache()

        if config.fullfit:
            break

    if not config.fullfit:
        train_pd.to_parquet(f'{root_dir}/oof_df.parquet')

    # Compute OOF with score
    if not config.fullfit:
        preds_col = [c for c in train_pd.columns if "preds_" in c] if config.label_col == LABEL else "preds"
        txt_macro, txt_micro, txt_binary = [], [], []
        for fold in config.folds:
            train = train_pd[train_pd["fold_s%d" % seed_] != fold]
            valid = train_pd[train_pd["fold_s%d" % seed_] == fold]
            preds = valid[preds_col].astype(int).values
            gt = np.vstack(valid[config.label_col].values).astype(int) if config.label_col == LABEL else valid[config.label_col].astype(int).values
            # print(gt.shape, gt.dtype, preds.shape, preds.dtype)
            txt_macro.append("%.4f" % f1_score(gt, preds, average="macro"))
            txt_micro.append("%.4f" % f1_score(gt, preds, average="micro"))
            if (config.label_col != LABEL) and (len(valid[config.label_col].unique()) == 2):
                txt_binary.append("%.4f" % f1_score(gt, preds, average="binary"))

        gt = np.vstack(train_pd[config.label_col].values).astype(int) if config.label_col == LABEL else train_pd[config.label_col].astype(int).values
        preds = train_pd[preds_col].astype(int).values
        score_macro = "F1(macro): %.4f (%s)" % (f1_score(gt, preds, average="macro"), "/".join(txt_macro))
        score_micro = "F1(micro): %.4f (%s)" % (f1_score(gt, preds, average="micro"), "/".join(txt_micro))
        score_binary = ""
        if (config.label_col != LABEL) and len(valid[config.label_col].unique()) == 2:
            score_binary = "F1(binary): %.4f (%s)" % (f1_score(gt, preds, average="binary"), "/".join(txt_binary))
        print("OOF", score_macro, score_micro, score_binary)

        oof_res = {"OOF(macro)": score_macro, "OOF(micro)": score_micro, "OOF(binary)": score_binary}
        with open(os.path.join(root_dir, "oof.json"), "w") as f:
            json.dump(oof_res, f)

        with open(os.path.join(root_dir, "classification_report.txt"), "w") as f:
            f.write(classification_report(gt, preds, target_names=class_names, digits=4))

    # Inference
    if config.label_col == "class":
        test_pd = pd.read_parquet(test_boxes_file).reset_index(drop=True)
        print(test_pd.shape)

        key = "%s_%d-%s" % (config.backbone.replace("/", "_"), config.img_size, config.label_col)
        root_dir = os.path.join(config.default_root_dir, "seed%d" % config.seed)
        models_pl = {key: {"root_dir": root_dir}}
        print(models_pl)

        final_test_pd = infer_model(models_pl, test_pd, tta=tta, images_home=test_boxes_home)
        final_test_pd.to_parquet(os.path.join(root_dir, "test_predictions%s.parquet" % tta_name))

        # Recompute predictions without background
        logits_col = [c for c in final_test_pd.columns if "logits_" in c]
        if len(logits_col) > 23:
            final_test_pd["preds"] = final_test_pd[logits_col[0:23]].values.argmax(axis=1).astype(np.int32)

        # Submission file
        submission_csv_pd = final_test_pd[
            ["trustii_id", "NAME", "pred_x1", "pred_y1", "pred_x2", "pred_y2", "preds"]].copy().rename(
            columns={'pred_x1': 'x1', 'pred_y1': 'y1', 'pred_x2': 'x2', 'pred_y2': 'y2', 'preds': 'class'})
        submission_csv_pd["class"] = submission_csv_pd["class"].astype(np.int32)
        submission_csv_pd["class"] = submission_csv_pd["class"].map(class_mapping)
        # Keep same order
        submission_csv_pd = pd.merge(pd.read_csv(test_file), submission_csv_pd, on=["trustii_id", "NAME"], how="left")
        submission_csv_pd.to_csv(os.path.join(root_dir, "submission%s.csv" % tta_name), index=False)
        submission_csv_pd
    else:
        test_pd = pd.read_csv(test_file)
        test_pd["filename"] = test_pd["NAME"].apply(lambda x: "images_cytologia/%s" % x)

        key = "%s_%d-%s" % (config.backbone.replace("/", "_"), config.img_size, config.label_col)
        root_dir = os.path.join(config.default_root_dir, "seed%d" % config.seed)
        models_pl = {key: {"root_dir": root_dir}}
        print(models_pl)

        final_test_pd = infer_model(models_pl, test_pd, tta=tta, images_home=images_home)
        # final_test_pd = final_test_pd.drop(columns=["trustii_id"]).groupby(["NAME","filename"]).first().reset_index()
        final_test_pd.to_parquet(os.path.join(root_dir, "test_predictions%s.parquet" % tta_name))