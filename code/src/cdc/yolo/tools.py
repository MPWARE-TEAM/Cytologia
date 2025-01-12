import numpy as np
import math
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import pandas as pd
from yolox.utils import xyxy2cxcywh
from tqdm import tqdm
import os
from PIL import Image


# Yolo Pytorch helpers
def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def letterbox_(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        # im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        # print("im before resize", im.shape, new_unpad)
        im = F.interpolate(im.view(1, 1, im.size(0), im.size(1)), (new_unpad[1], new_unpad[0]), mode="bilinear")[0, 0]
        # print("im resize", im.shape)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # print("pad", (left, right, top, bottom))
    im = F.pad(input=im, pad=(left, right, top, bottom), mode='constant', value=color[0]/255.)
    # print("im pad", im.shape)
    # print(ratio, (dw, dh), new_unpad, top, bottom, left, right, color)
    return im


def make_divisible_(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def preprocess(ims, size, model_stride=64):
    size = (size, size)
    shape0, shape1, files = [], [], []  # image and inference shapes, filenames
    for i, im in enumerate(ims):
        f = f'image{i}'  # filename
        files.append(f + ".jpg")
        s = im.shape[:2]  # HWC
        shape0.append(s)  # image shape
        g = max(size) / max(s)  # gain
        shape1.append([int(y * g) for y in s])
    shape1 = [make_divisible_(x, model_stride) for x in np.array(shape1).max(0)]  # inf shape
    xs = [letterbox_(im, shape1, auto=False) for im in ims]  # pad
    return xs, files, shape0, shape1


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_intersection(bb1, bb2):
    """
    Calculate the Intersection of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0, 0, 0, 0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    return x_left, x_right, y_top, y_bottom


def compute_intersection(x):
    bb1 = {'x1': x["roi_x1"], 'x2': x["roi_x2"], 'y1': x["roi_y1"], 'y2': x["roi_y2"]}
    bb2 = {'x1': x["x1"], 'x2':x ["x2"], 'y1': x["y1"], 'y2': x["y2"]}
    return get_intersection(bb1, bb2)


def compute_iou(x):
    bb1 = {'x1':x["oof_bbx_xtl"], 'x2':x["oof_bbx_xbr"], 'y1':x["oof_bbx_ytl"], 'y2':x["oof_bbx_ybr"]}
    bb2 = {'x1':x["bbx_xtl"], 'x2':x["bbx_xbr"], 'y1':x["bbx_ytl"], 'y2':x["bbx_ybr"]}
    return get_iou(bb1, bb2)


def tiling_start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def convert_tlrb_to_yolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def calculate_iou(box1, box2, use_giou=False, eps=1e-16):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / (union_area + eps)

    if use_giou:
        enclose_x1 = min(x1, x3)
        enclose_y1 = min(y1, y3)
        enclose_x2 = max(x2, x4)
        enclose_y2 = max(y2, y4)
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        giou = iou - (enclose_area - union_area) / (enclose_area + eps)
        return giou
    else:
        return iou


def pair_iou(pred, target, use_giou=False, eps=1e-16):
    pred = pred.view(-1, 4)
    target = target.view(-1, 4)
    # Convert x1, y1, x2, y2 to center_x, center_y, width, height
    pred = xyxy2cxcywh(pred)
    target = xyxy2cxcywh(target)
    tl = torch.max(
        (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
    )
    br = torch.min(
        (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
    )

    area_p = torch.prod(pred[:, 2:], 1)
    area_g = torch.prod(target[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en
    area_u = area_p + area_g - area_i
    iou = (area_i) / (area_u + eps)

    if use_giou:
        c_tl = torch.min(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        c_br = torch.max(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )
        area_c = torch.prod(c_br - c_tl, 1)
        giou = iou - (area_c - area_u) / area_c.clamp(eps)
        return giou
    else:
        return iou


def calculate_precision_recall_at_iou_thr(gt_boxes, dt_boxes, use_giou=False, thr=0.5):
    tp = 0
    fp = 0
    fn = len(gt_boxes)
    for dt_box in dt_boxes:
        dt_x1, dt_x2, dt_y1, dt_y2 = dt_box[0:-1]
        dt_box_ = [dt_x1, dt_y1, dt_x2, dt_y2]
        matched = False
        for gt_box in gt_boxes:
            gt_x1, gt_x2, gt_y1, gt_y2 = gt_box[0:-1]
            gt_box_ = [gt_x1, gt_y1, gt_x2, gt_y2]
            giou = pair_iou(torch.tensor(dt_box_).unsqueeze(0), torch.tensor(gt_box_).unsqueeze(0), use_giou=use_giou).numpy()[0]
            # iou = calculate_iou(dt_box_, gt_box_, use_giou=use_giou)
            # assert(iou == giou), "%f %f" % (iou, giou)
            # print(iou, giou)
            if giou >= thr:
                matched = True
                fn -= 1
                break
        if matched:
            tp += 1
        else:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


def compute_fbeta(precision, recall, beta=1):
    fbeta = (1 + beta*beta)*(precision*recall/(beta*beta*precision + recall)) if (precision + recall) > 0 else 0.0
    return fbeta


def fit_format(bbx, bbs, img_width, img_height, padding=False, padding_policy="empty"):
    if bbx is not None:
        # Sort by confidence and keep top N
        bbx = sorted(bbx, key=lambda x: x[-1], reverse=True)[0:bbs]
        if (padding) and (len(bbx) < bbs):
            # Pad missing bb
            if padding_policy == "duplicate_first_jitter":
                bbx = bbx + [(bbx[0][0]+1, bbx[0][1]-1, bbx[0][2]+1, bbx[0][3]-1, bbx[0][4], bbx[0][5], bbx[0][6])] * (bbs - len(bbx))
            elif padding_policy == "duplicate_first":
                bbx = bbx + [(bbx[0])] * (bbs - len(bbx))
            elif padding_policy == "empty":
                bbx = bbx + [(0, 0, 0, 0, 0, 0, 0)]*(bbs - len(bbx))
            elif padding_policy == "full":
                bbx = bbx + [(0, img_width-1, 0, img_height-1, 0, 0, 0.0)]*(bbs - len(bbx))
    return bbx


def compute_bbx_avg_score(bbx):
    score = 0
    if bbx is not None:
        score = np.mean([bb[-1] for bb in bbx])
    return score


def fix_bbox(x1, y1, x2, y2, width, height):
    x1 = np.clip(x1, 0, width-1)
    x2 = np.clip(x2, 0, width-1)
    y1 = np.clip(y1, 0, height-1)
    y2 = np.clip(y2, 0, height-1)
    return x1, y1, x2, y2


def generalized_box_iou(boxes1, boxes2):
    gious_matrix = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    for idx1, bb_gt in enumerate(boxes1):
        for idx2, bb_pred in enumerate(boxes2):
            gious_matrix[idx1][idx2] = calculate_iou(bb_gt, bb_pred, use_giou=True, eps=1e-16)
    return gious_matrix


def find_max_index_values_by_row(gious_matrix):
    return np.argmax(gious_matrix, axis=1), np.max(gious_matrix, axis=1)


def compute_metric(true_df, pred_df, giou_only=False):
    images_df = pd.DataFrame()

    for image in tqdm(true_df["NAME"].unique()):
        # Load GT for each image
        true_mask = true_df["NAME"] == image
        true_image_df = true_df[true_mask].copy()

        pred_mask = (pred_df["NAME"] == image) & (pred_df["trustii_id"].isin(true_image_df["trustii_id"].unique()))

        pred_image_df = (
            pred_df[pred_mask]
            .drop_duplicates(subset="trustii_id")
            .drop_duplicates(subset=["x1", "y1", "x2", "y2", "class"])
        ).copy()

        # print("GT")
        # display(true_image_df)
        # print("PRED")
        # display(pred_image_df)

        if pred_image_df.shape[0] > 0:
            box1 = true_image_df[["x1", "y1", "x2", "y2"]].values
            box2 = pred_image_df[["x1", "y1", "x2", "y2"]].values
            # print("GT:", box1.shape)
            # print(box1)
            # print("PRED:", box2.shape)
            # print(box2)
            giou = generalized_box_iou(box1, box2)
            # print(giou)
            result_indices, result_values = find_max_index_values_by_row(giou)
            # print(result_indices, result_values)
            pred_image_df["boundingbox_id"] = range(len(pred_image_df))
            true_image_df["boundingbox_id"] = result_indices
            true_image_df["giou"] = result_values
            # print("GT")
            # display(true_image_df)
            # print("PRED")
            # display(pred_image_df)
            image_df = true_image_df.merge(pred_image_df[["boundingbox_id", "class", "x1", "y1", "x2", "y2", "score"]].rename(columns={'x1':'pred_x1', 'y1':'pred_y1', 'x2':'pred_x2', 'y2':'pred_y2', 'score':'pred_score'}), how="left", on="boundingbox_id")
            # print("image_df")
            # display(image_df)
        else:
            image_df = true_image_df.copy().rename(columns={'class': 'class_x'})
            image_df["boundingbox_id"] = range(len(image_df))
            image_df["class_x"] = None
            image_df["giou"] = -1
        images_df = pd.concat([images_df, image_df])

        # break
    images_df["rescaled_iou"] = ((images_df["giou"] + 1) / 2).fillna(0)
    if giou_only:
        return float(1.0 * images_df["rescaled_iou"].mean()), images_df
    else:
        f_score = f1_score(images_df["class_x"].values, images_df["class_y"].astype(str).values, average="macro")
        return float(.2 * images_df["rescaled_iou"].mean() + 0.8 * f_score), images_df


def dump_boxes(df, input_folder, out_folder, margins=0):
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename = row["NAME"]
        img_width = row["img_width"]
        img_height = row["img_height"]
        pred_width = row["pred_width"]
        pred_height = row["pred_height"]
        uid = row["trustii_id"]
        class_x = row["class_x"] if "class_x" in df.columns else ""
        file = os.path.join(input_folder, filename)
        image = np.array(Image.open(file))
        pred_x1, pred_y1, pred_x2, pred_y2 = row["pred_x1"], row["pred_y1"], row["pred_x2"], row["pred_y2"]

        if margins >= 0:
            pred_x1 = np.clip(pred_x1 - margins, 0, img_width - 1)
            pred_y1 = np.clip(pred_y1 - margins, 0, img_height - 1)
            pred_x2 = np.clip(pred_x2 + margins, 0, img_width - 1)
            pred_y2 = np.clip(pred_y2 + margins, 0, img_height - 1)

            crop = image[pred_y1:pred_y2, pred_x1:pred_x2]
            if margins == 0:
                assert(crop.shape[1] == pred_width)
                assert(crop.shape[0] == pred_height)
        else:
            crop = image

        new_filename = "%s-%d%s.png"%(filename.replace(".jpg", ""), uid, ("-%s"%class_x) if len(class_x) > 0 else "")
        new_filepath = os.path.join(out_folder, new_filename)
        if os.path.exists(new_filepath):
            print("WARNING: File already exists, ignore now", new_filepath)
        else:
            Image.fromarray(crop).save(new_filepath, "png")
            df.loc[idx, "filename"] = new_filename
