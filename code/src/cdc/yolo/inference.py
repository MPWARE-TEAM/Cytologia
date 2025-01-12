from loguru import logger
import numpy as np
import pandas as pd
import time, os
import torch
import torch.nn as nn
import cv2
from PIL import Image
from ensemble_boxes import *
from tqdm import tqdm
import torchvision.transforms as T

from yolox.data.data_augment import preproc as preprocess
from yolox.data import ValTransform
from yolox.utils import demo_postprocess, postprocess, multiclass_nms


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=None,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
            verbose=1
    ):
        self.model = model
        if device == "gpu":
            self.model = self.model.cuda()
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.verbose = verbose

    def inference_torch(self, img, min_score=0.0):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            # img = cv2.imread(img)
            # Convert to CV2 like
            img = np.array(Image.open(img))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_info["file_name"] = None

        # t0 = time.time()

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )

        filename = img_info["file_name"].split("/")[-1]

        output = outputs[0]
        ret = []
        if output is not None and len(output) > 0:
            output = output.cpu().numpy()
            bboxes = output[:, 0:4]
            bboxes /= ratio
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for i in range(0, len(output)):
                score_ = scores[i]
                bbox_ = bboxes[i]
                xmin, ymin, xmax, ymax = bbox_[0], bbox_[1], bbox_[2], bbox_[3]
                cls_ = cls[i]
                if score_ > min_score:
                    ret.append((filename, width, height, xmin, ymin, xmax, ymax, score_, cls_))
        else:
            print("Nothing detected", filename) if self.verbose > 0 else None

        # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return ret


def predict_yolox(home, ckpt_file, test_conf=0.00001, nmsthre=0.30, image_size=512, files=None, device="gpu", verbose=1, map_classes_list=None):
    if "yolox_s" in ckpt_file:
        from exps.example.custom.yolox_s_v320_640_seed_42_fold0 import Exp
    elif "yolox_m" in ckpt_file:
        from exps.example.custom.yolox_m_v323_640_seed_42_fold0 import Exp

    exp = Exp()
    exp.test_conf = test_conf
    exp.nmsthre = nmsthre
    exp.test_size = (image_size, image_size)

    model = exp.get_model().half().eval().cuda() if device == "gpu" else exp.get_model().eval()
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # Half only in GPU (slow conv2D not implemented for CPU)
    predictor = Predictor(
        model, exp, cls_names=map_classes_list, trt_file=None, decoder=None,
        device=device, fp16=True if device == "gpu" else False, legacy=False, verbose=verbose
    )

    df = image_predict(predictor, home, files=files)
    return df


def image_predict(predictor, path, files=None):
    if files is None:
        if os.path.isdir(path):
            files = get_image_list(path)
        else:
            files = [path]
    files.sort()

    results = []
    for image_name in tqdm(files):
        detected_items = predictor.inference_torch(image_name)
        results.extend(detected_items)

    return pd.DataFrame(results, columns=["filename", "slide_width", "slide_height", "bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr", "score", "class"])


def get_yolo_model(architecture, weights):
    from ultralytics import YOLO
    from ultralytics import RTDETR

    if architecture == "YOLO":
        model = YOLO(weights, task='detect')
    elif architecture == "RTDETR":
        model = RTDETR(weights)
    else:
        raise Exception("Architecture not found", architecture)

    return model


def predict_yolo(home, ckpt_file, test_conf=0.00001, nmsthre=0.30, image_size=512, files=None, device="gpu", verbose=1,
                 map_classes_list=None, max_det=20, architecture="YOLO"):
    if not os.path.exists(ckpt_file):
        raise (BaseException("Weights not found", ckpt_file))

    model = get_yolo_model(architecture, ckpt_file)
    model = model.eval().cuda() if device == "gpu" else model.eval()

    results = []
    for file in tqdm(files):
        filename = os.path.basename(file)
        outputs = model.predict(source=file, imgsz=image_size, max_det=max_det, conf=test_conf, iou=nmsthre,
                                augment=False, device="0" if device == "gpu" else device,
                                verbose=False)  # iou=0.7
        for r in outputs:
            boxes = r.boxes.cpu().numpy()
            for bbox in boxes:
                box = bbox.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                score = bbox.conf[0]
                label = bbox.cls[0]
                xmin_, ymin_, xmax_, ymax_ = box
                results.append((filename, bbox.orig_shape[1], bbox.orig_shape[0], xmin_, ymin_, xmax_, ymax_, score, label))
    df = pd.DataFrame(results, columns=["filename", "slide_width", "slide_height", "bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr", "score", "class"])
    return df


def predict_rtdetrv2(config, ckpt_file, test_conf=0.00001, nmsthre=0.30, image_size=512, files=None, device="gpu", verbose=1):
    device = "cuda" if device == "gpu" else device
    from src.core import YAMLConfig

    if not os.path.exists(ckpt_file):
        raise (BaseException("Weights not found", ckpt_file))

    cfg = YAMLConfig(config, resume=ckpt_file)

    if ckpt_file:
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()
    model = model.eval()
    model = model.to(device)

    transforms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    results = []
    for file in tqdm(files):
        filename = os.path.basename(file)

        im_pil = Image.open(file).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)
        im_data = transforms(im_pil)[None].to(device)

        with torch.no_grad():
            output = model(im_data, orig_size)
        labels, boxes, scores = output

        scr = scores[0]
        labs_ = labels[0][scr > test_conf].cpu().numpy()
        boxes_ = boxes[0][scr > test_conf].cpu().numpy()
        scrs_ = scores[0][scr > test_conf].cpu().numpy()

        for bbox, label, score in zip(boxes_, labs_, scrs_):
            xmin_, ymin_, xmax_, ymax_ = bbox
            results.append((filename, w, h, xmin_, ymin_, xmax_, ymax_, score, label))
    df = pd.DataFrame(results, columns=["filename", "slide_width", "slide_height", "bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr", "score", "class"])
    return df


def image_predict(predictor, path, files=None):
    if files is None:
        if os.path.isdir(path):
            files = get_image_list(path)
        else:
            files = [path]
    files.sort()

    results = []
    for image_name in tqdm(files):
        detected_items = predictor.inference_torch(image_name)
        results.extend(detected_items)

    return pd.DataFrame(results, columns=["filename", "slide_width", "slide_height", "bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr", "score", "class"])


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in [".jpg", ".jpeg", ".png"]:
                image_names.append(apath)
    return image_names


def run_wbf(df, iou_thr=0.5, skip_box_thr=0.0001):

    df_per_slide = df.groupby(["filename", "slide_width", "slide_height"])[["bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr", "score", "class"]].agg(list).reset_index()

    wbf_bbxs = []
    for idx, row in df_per_slide.iterrows():
        filename, img_w, img_h = row["filename"], row["slide_width"], row["slide_height"]

        bbx_xtls = row["bbx_xtl"]
        bbx_ytls = row["bbx_ytl"]
        bbx_xbrs = row["bbx_xbr"]
        bbx_ybrs = row["bbx_ybr"]
        scores = row["score"]
        labels = row["class"]

        # Normalize BB
        boxes_list, scores_list, labels_list = [], [scores], [labels]
        for xtl, ytl, xbr, ybr in zip(bbx_xtls, bbx_ytls, bbx_xbrs, bbx_ybrs):
            xtl, ytl, xbr, ybr = xtl / img_w, ytl / img_h, xbr / img_w, ybr / img_h
            boxes_list.append([xtl, ytl, xbr, ybr])

        boxes, scores, labels = weighted_boxes_fusion([boxes_list], scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr)  # weights=weights

        # Denormalize
        for box, score, label in zip(boxes, scores, labels):
            wbf_bbxs.append((filename, img_w, img_h, box[0] * img_w, box[1] * img_h, box[2] * img_w, box[3] * img_h, score, label))

    return pd.DataFrame(wbf_bbxs, columns=["filename", "slide_width", "slide_height", "bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr", "score", "class"])