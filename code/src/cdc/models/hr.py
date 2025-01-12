import torch.nn as nn
from timm.layers.conv2d_same import Conv2dSame


# Hyper Resolution
def set_hyper_resolution(stem_conv2d, new_stride=(1, 1)):
    stem_conv2d_pretrained_weight = stem_conv2d.weight.clone()
    print("Stride from ", stem_conv2d.stride, "to", new_stride)
    stem_conv2d_ = Conv2dSame(stem_conv2d.in_channels,
                              stem_conv2d.out_channels,
                              kernel_size=stem_conv2d.kernel_size,
                              stride=new_stride,
                              padding=stem_conv2d.padding,
                              dilation=stem_conv2d.dilation,
                              bias=True if stem_conv2d.bias is True else False)
    stem_conv2d_.weight = nn.Parameter(stem_conv2d_pretrained_weight)
    return stem_conv2d_


def apply_hyper_resolution(model, encoder_name, hyperresolution=(1, 1)):
    if "tf_efficientnet" in encoder_name:
        mfeatures = model.encoder.model
        mfeatures.conv_stem = set_hyper_resolution(mfeatures.conv_stem, new_stride=hyperresolution)
    elif "regnet" in encoder_name:
        mfeatures = model.encoder.model
        mfeatures.stem.conv = set_hyper_resolution(mfeatures.stem.conv, new_stride=hyperresolution)
    elif "csp" in encoder_name:
        mfeatures = model.encoder.model
        mfeatures.stem[0].conv = set_hyper_resolution(mfeatures.stem[0].conv,
                                                      new_stride=hyperresolution)
    elif "resnest" in encoder_name:
        mfeatures = model.encoder.model
        mfeatures.conv1[0] = set_hyper_resolution(mfeatures.conv1[0], new_stride=hyperresolution)
    elif "seresnext" in encoder_name:
        mfeatures = model.encoder.model
        mfeatures.conv1 = set_hyper_resolution(mfeatures.conv1, new_stride=hyperresolution)
    elif "densenet" in encoder_name:
        mfeatures = model.encoder.model
        mfeatures.features.conv0 = set_hyper_resolution(mfeatures.features.conv0,
                                                        new_stride=hyperresolution)
    elif "nfnet" in encoder_name:
        mfeatures = model.encoder.model
        mfeatures.stem.conv1 = set_hyper_resolution(mfeatures.stem.conv1, new_stride=hyperresolution)
    else:
        raise(Exception("Hyper resolution not supported for %s" % encoder_name))