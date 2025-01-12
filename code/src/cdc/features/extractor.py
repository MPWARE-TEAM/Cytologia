import os
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import timm
import cv2
import torch
import torch.nn as nn
from torchvision import transforms


# https://huggingface.co/histai/hibou-b
def load_hibou_model(weights="histai/hibou-b", device="cuda"):
    in_features = 1024 if "hibou-l" in weights else 768
    from transformers import AutoImageProcessor, AutoModel
    print('Loading weights:', weights)
    processor = AutoImageProcessor.from_pretrained(weights, trust_remote_code=True)
    model = AutoModel.from_pretrained(weights, trust_remote_code=True)
    model = model.to(device)
    return model, processor, in_features


def features_from_hibou(model, input, prepare_feed=None, device="cuda"):
    hf_output = model(input)
    return hf_output.pooler_output


# https://github.com/marrlab/DinoBloom
def get_dino_bloom(modelpath, modelname="dinov2_vitb14"):
    embed_sizes={"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}
    # load the original DINOv2 model with the correct architecture and parameters.
    model=torch.hub.load('facebookresearch/dinov2', modelname)
    # load finetuned weights
    print("Loading (%s) weights:" % modelname, modelpath)
    pretrained = torch.load(modelpath, map_location=torch.device('cpu'))
    # Make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key or "ibot_head" in key:
            pass
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    # Match to 224x224 image. patch size=14x14 => 16*16 patches
    in_features = embed_sizes[modelname]
    pos_embed = nn.Parameter(torch.zeros(1, 257, in_features))
    model.pos_embed = pos_embed
    model.load_state_dict(new_state_dict, strict=True)
    return model, in_features


def load_dinobloom_model(weights="1aurent/vit_base_patch14_224.dinobloom", device="cuda"):
    if '.pth' in weights:
        model, in_features = get_dino_bloom(weights.split("/")[1], modelname=weights.split("/")[0])
        processor = transforms.Compose([
            # transforms.Resize(size=224, interpolation="bicubic", max_size=None, antialias=True),
            transforms.ToTensor(),
            # transforms.CenterCrop(size=[224, 224]),
            transforms.Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]))
        ])
    else:
        # load model from the hub
        in_features = 1024 if "large" in weights else 768
        model = timm.create_model(model_name="hf-hub:"+weights, pretrained=True)
        data_config = timm.data.resolve_model_data_config(model)  # input_size=518
        processor = timm.data.create_transform(**data_config, is_training=False)
        model = model.to(device)
    return model, processor, in_features


def features_from_dinobloom(model, input, prepare_feed=None, device="cuda", patch_num_h=16, patch_num_w=16, in_features=None):
    # Tensor (BS, C, H, W) between 0.0 and 1.0 expected
    data = input.to(device)
    features = model(data)
    return features


def features_from_foundation_model(name, model, input, prepare_feed=None, device="gpu", in_features=None):
    device = "cuda" if device == "gpu" else device
    name = name.replace("foundation_", "") if name.startswith("foundation_") else name
    if "hibou" in name.lower():
        features = features_from_hibou(model, input, prepare_feed=prepare_feed, device=device)
    elif "dinobloom" in name.lower():
        features = features_from_dinobloom(model, input, prepare_feed=prepare_feed, device=device, in_features=in_features)
    else:
        raise(Exception("Foundation model not supported %s" % name))
    return features


def load_foundation_model(name, freeze=True, freeze_layers=None, unfreeze_encoder_layers=None, device="gpu"):
    device = "cuda" if device == "gpu" else device
    name = name.replace("foundation_", "") if name.startswith("foundation_") else name
    if "hibou" in name.lower():
        model, processor, in_features = load_hibou_model(weights=name, device=device)
        if freeze:  # Freeze all
            print("Freezing model, %s" % name, "device:", device)
            freeze_model(model)
        if (freeze_layers is not None) and (model is not None):  # Freeze some layers
            freeze_encoder_layers(model, layers=freeze_layers)
        if (unfreeze_encoder_layers is not None) and (model is not None):  # Unfreeze some layers
            unfreeze_encoder_layers(model, layers=unfreeze_encoder_layers)
    elif "dinobloom" in name.lower():
        model, processor, in_features = load_dinobloom_model(weights=name, device=device)
        if freeze:  # Freeze all
            print("Freezing model, %s" % name, "device:", device)
            freeze_model(model)
        if (freeze_layers is not None) and (model is not None):  # Freeze some layers
            freeze_encoder_blocks(model, layers=freeze_layers)
        if (unfreeze_encoder_layers is not None) and (model is not None):  # Unfreeze some layers
            unfreeze_encoder_blocks(model, layers=unfreeze_encoder_layers)
    else:
        raise(Exception("Foundation model not supported %s" % name))

    return model, processor, in_features


def freeze_model(model):
    # Freeze full model
    print("Freezing full model")
    if model is not None:
        for param in model.parameters():
            param.requires_grad = False


def unfreeze_encoder_blocks(model, layers=None):
    # Freeze some layers
    if (layers is not None) and (model is not None):
        print("Unfreezing %d blocks, last %d over %d" % (layers, len(model.blocks[layers:]), len(model.blocks)))
        for layer in model.blocks[layers:]:
            for param in layer.parameters():
                param.requires_grad = True


def freeze_encoder_blocks(model, layers=None):
    # Freeze some layers
    if (layers is not None) and (model is not None):
        print("Freezing %d blocks, %d over %d" % (layers, len(model.blocks[:layers]), len(model.blocks)))
        for layer in model.blocks[:layers]:
            for param in layer.parameters():
                param.requires_grad = False


def freeze_encoder_layers(model, layers=None):
    # Freeze some layers
    if (layers is not None) and (model is not None):
        print("Freezing %d layers, %d over %d" % (layers, len(model.encoder.layer[:layers]), len(model.encoder.layer)))
        for layer in model.encoder.layer[:layers]:
            for param in layer.parameters():
                param.requires_grad = False


def unfreeze_encoder_layers(model, layers=None):
    # Freeze some layers
    if (layers is not None) and (model is not None):
        print("Unfreezing %d layers, last %d over %d" % (layers, len(model.encoder.layer[layers:]), len(model.encoder.layer)))
        for layer in model.encoder.layer[layers:]:
            for param in layer.parameters():
                param.requires_grad = True
