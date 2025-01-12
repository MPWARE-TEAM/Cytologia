import torch
import numpy as np
from torch.nn import functional as F


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
        print("index", index)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_cross_entropy(pred, y_a, y_b, lam, label_smoothing=0.0):
    return lam * F.cross_entropy(pred, y_a, label_smoothing=label_smoothing) + (1 - lam) * F.cross_entropy(pred, y_b, label_smoothing=label_smoothing)


def mixup_binary_cross_entropy_with_logits(pred, y_a, y_b, lam):
    return lam * F.binary_cross_entropy_with_logits(pred, y_a) + (1 - lam) * F.binary_cross_entropy_with_logits(pred, y_b)


def mixup_loss(pred, y_a, y_b, lam, loss):
    return lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b)


def apply_mix(x, y, config, is_valid=False):
    y_a, y_b, lam, mixup_batch = y, y, 0, False
    # CutMix + MixUp
    if (config.cutmix_prob is not None) and (config.mixup_prob is not None) and (is_valid == False):
        if np.random.random() > (config.mixup_prob + config.cutmix_prob) * 0.5:
            if np.random.random() > 0.50:
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=config.cutmix_alpha)
            else:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=config.mixup_alpha)
            mixup_batch = True
    else:
        # MixUp
        if (config.mixup_prob is not None) and (is_valid == False):
            if np.random.random() > config.mixup_prob:
                # # y_a is original target, y_b is the permuted one, lam is the factor
                x, y_a, y_b, lam = mixup_data(x, y, alpha=config.mixup_alpha)
                mixup_batch = True
        # CutMix
        if (config.cutmix_prob is not None) and (is_valid == False):
            if np.random.random() > config.cutmix_prob:
                # # y_a is original target, y_b is the permuted one, lam is the factor
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=config.cutmix_alpha)
                mixup_batch = True

    return x, y_a, y_b, lam, mixup_batch


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return bbx1, bby1, bbx2, bby2, lam


def cutmix_data(x, y, alpha=1.0, device="cuda"):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).to(device)

    bbx1, bby1, bbx2, bby2, lam = rand_bbox(x.size(), lam)

    mixed_x = x.clone()

    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

