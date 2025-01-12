import torch
import os
import numpy as np

import pytorch_lightning as L
from torch.nn import functional as F
import torch.nn as nn

from cdc.common.constants import LABEL
from cdc.features.extractor import load_foundation_model, features_from_foundation_model
from cdc.models.hr import apply_hyper_resolution
from cdc.models.pl.losses import *
from cdc.training.mix import apply_mix, mixup_cross_entropy, mixup_loss
import timm

from cdc.models.pl.pooling import get_pooling_layer


class CDCModel(L.LightningModule):
    def __init__(self, config, pretrained=None, infer=False):
        super().__init__()

        self.pretrained = pretrained
        self.config = config
        self.preprocessor = None

        # Timm model
        model_weights = config.backbone
        if self.pretrained is not None:
            model_weights = self.pretrained
            print("Loading weights:", model_weights)
        else:
            if self.config.pretrained is not None:
                model_weights = os.path.join(self.config.pretrained, "best_model.ckpt")
                print("Loading config weights:", model_weights)

        if config.backbone.startswith("foundation_"):
            self.backbone, self.preprocessor, self.in_features = load_foundation_model(config.backbone,
                                                                                       freeze=config.freeze,
                                                                                       freeze_layers=config.freeze_encoder_layers,
                                                                                       unfreeze_encoder_layers=config.unfreeze_encoder_layers,
                                                                                       device=config.device)
        else:
            self.backbone = timm.create_model(config.backbone, pretrained=not infer, num_classes=0)  # Remove default classification head
            self.in_features = self.backbone.num_features

        # Apply Hyper Resolution
        if self.config.hyperresolution:
            apply_hyper_resolution(self.backbone, config.backbone)

        # Replace Global Pooling
        if self.config.pooling_type is not None:
            self.backbone.global_pool = get_pooling_layer(self.config)

        # Add Classifier
        self.classifier = nn.Linear(self.in_features, self.config.num_labels)

        self._init_weights(self.classifier)

        if config.metric != "macro":
            self.metric_avg = "binary" if config.num_labels == 2 else config.metric
        else:
            self.metric_avg = config.metric

        if not infer:
            from torchmetrics import F1Score, Precision, Recall
            task = "multilabel" if self.config.label_col == LABEL else "multiclass"
            task = "binary" if self.config.num_labels == 2 else task
            self.valid_f1 = F1Score(task=task, num_labels=self.config.num_labels, num_classes=self.config.num_labels, average=self.metric_avg)
            self.valid_precision = Precision(task=task, num_labels=self.config.num_labels, num_classes=self.config.num_labels, average=self.metric_avg)
            self.valid_recall = Recall(task=task, num_labels=self.config.num_labels, num_classes=self.config.num_labels, average=self.metric_avg)

        self.loss = None
        if self.config.loss is not None:
            if self.config.loss == 'focal_bce':
                # For multi-labels/binary
                self.loss = FocalLossBCE(alpha=self.config.alpha, gamma=self.config.gamma)
            elif self.config.loss == 'focal_ce':
                # For multi-classes
                self.loss = FocalLossCE(alpha=self.config.alpha, gamma=self.config.gamma)
            elif self.config.loss == 'bce':
                # For multi-labels/binary
                self.loss = nn.BCEWithLogitsLoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # print("x:", x.shape, x.dtype, type(x))
        if self.config.backbone.startswith("foundation_"):
            x = features_from_foundation_model(self.config.backbone, self.backbone, x,
                                               prepare_feed=self.preprocessor, device=x.device,
                                               in_features=self.in_features)
            # print(self.config.backbone, x.shape, x.device)
        else:
            x = self.backbone(x)
        # print("backbone:", x.shape)
        x = self.classifier(x)
        # print("head:", x.shape)
        return x

    def _get_preds_loss_metrics(self, batch, is_valid=False):
        '''convenience function since train/valid/test steps are similar'''
        X, y = batch['image'], batch['label']  # (BS, C, H, W), (BS, C)

        # Optional mix within the batch
        X, y_a, y_b, lam, mixup_batch = apply_mix(X, y, self.config, is_valid=is_valid)

        logits = self(X)  # (BS, C)

        if (self.config.label_col == LABEL) and (self.loss is not None):
            # Multi-labels or binary
            if mixup_batch:
                loss = mixup_loss(logits, y_a, y_b, lam, self.loss)
                y_true_mixed = (lam * y_a + (1 - lam) * y_b)  # (BS, C) Two class enabled with prob
                y_true = (y_true_mixed.squeeze(1) > self.config.probs_threshold).long()
            else:
                loss = self.loss(logits, y.squeeze(1))
                y_true = y.squeeze(1)
            # Move to probability with sigmoid with BCEWithLogitsLoss
            probabilities = torch.sigmoid(logits)  # 1 / (1 + np.exp(-logits))
            # Default threshold
            preds = (probabilities > self.config.probs_threshold).long()  # (BS, C)
            # print(self.loss, "logits", logits.shape, "y", y.shape, "preds", preds.shape, "y_true", y_true.shape)
        else:
            # Multi-classes
            if mixup_batch:
                loss = mixup_cross_entropy(logits, y_a, y_b, lam, label_smoothing=self.config.label_smoothing)
                y_true_mixed = (lam * y_a + (1 - lam) * y_b)  # (BS, C) Two class enabled with prob
                y_true = torch.argmax(y_true_mixed, dim=1)  # (BS) # the highest one for training score
            else:
                if self.loss is not None:
                    loss = self.loss(logits, y, label_smoothing=self.config.label_smoothing)
                else:
                    loss = F.cross_entropy(logits, y, label_smoothing=self.config.label_smoothing)
                y_true = torch.argmax(y, dim=1)  # (BS)
            preds = torch.argmax(logits, dim=-1)  # (BS)
            # print("cross_entropy", "logits", logits.shape, "y", y.shape, "preds", preds.shape, "y_true", y_true.shape)

        # Batch score
        from torchmetrics.functional.classification import f1_score as tm_f1_score
        from torchmetrics.functional.classification import precision as tm_precision
        from torchmetrics.functional.classification import recall as tm_recall
        task = "multilabel" if self.config.label_col == LABEL else "multiclass"
        task = "binary" if self.config.num_labels == 2 else task
        f1 = tm_f1_score(preds, y_true, task=task, num_labels=self.config.num_labels, num_classes=self.config.num_labels, average=self.metric_avg)
        precision = tm_precision(preds, y_true, task=task, num_labels=self.config.num_labels, num_classes=self.config.num_labels, average=self.metric_avg)
        recall = tm_recall(preds, y_true, task=task, num_labels=self.config.num_labels, num_classes=self.config.num_labels, average=self.metric_avg)

        # Accumulate predictions/ground truth
        if is_valid:
            self.valid_precision.update(preds, y_true)
            self.valid_recall.update(preds, y_true)
            self.valid_f1.update(preds, y_true)

        return preds, loss, f1, precision, recall

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        _, loss, f1, precision, recall = self._get_preds_loss_metrics(batch)
        # Log loss and metric
        self.log('train_step_loss', loss, batch_size=self.config.train_batch_size)  #, sync_dist=True
        self.log('train_step_precision', precision, batch_size=self.config.train_batch_size)  #, sync_dist=True
        self.log('train_step_recall', recall, batch_size=self.config.train_batch_size)  #, sync_dist=True
        self.log('train_step_f1', f1, batch_size=self.config.train_batch_size)  #, sync_dist=True
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, f1, precision, recall = self._get_preds_loss_metrics(batch, is_valid=True)
        # Log loss and metric
        self.log('val_step_loss', loss, batch_size=self.config.eval_batch_size)  #, sync_dist=True
        self.log('val_step_precision', precision, batch_size=self.config.eval_batch_size)  #, sync_dist=True
        self.log('val_step_recall', recall, batch_size=self.config.eval_batch_size)  #, sync_dist=True
        self.log('val_step_f1', f1, batch_size=self.config.eval_batch_size)  #, sync_dist=True
        return preds

    def on_validation_epoch_end(self):
        self.log('val_precision', self.valid_precision.compute(), prog_bar=True)
        self.log('val_recall', self.valid_recall.compute(), prog_bar=True)
        self.log('val_f1', self.valid_f1.compute(), prog_bar=True)
        self.valid_precision.reset()
        self.valid_recall.reset()
        self.valid_f1.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch['image']
        logits = self(X)
        return logits

    def configure_optimizers(self):
        if self.config.weight_decay is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        else:
            optimizer = self.fetch_optimizer()
        if self.config.lr_scheduler_type == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.config.train_epochs)
        elif self.config.lr_scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
        elif self.config.lr_scheduler_type == "reduceonplateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.20, patience=10),
                "monitor": "val_f1",  # Monitor the F1 metric
                "interval": "epoch",  # Adjust every epoch
                "frequency": 1
            }
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.train_epochs, eta_min=0.0)
        return [optimizer], [scheduler]

    def fetch_optimizer(self):
        head_params = list(self.head.named_parameters())
        param_optimizer = list(self.backbone.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            # optional lr for head
            {
                "params": [p for n, p in head_params],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.head_lr,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.config.lr)

        return optimizer
