import os, random
import numpy as np
import torch
import pickle
import json


def seed_everything_now(seed):
    """
    Seeds basic parameters for reproducibility of results.
    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_dict(tmp_dict, filename):
    pickle.dump(tmp_dict, open(filename, 'wb'))


def load_dict(filename):
    return pickle.load(open(filename, 'rb'))


class Config:
    """
    Placeholder to load a config from a saved json
    """

    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def save_config(config, path):
    """
    Saves a config as a json
    Args:
        config (Config): Config.
        path (str): Path to save at.
    """
    dic = config.__dict__.copy()
    if dic.get("__doc__") is not None:
        del dic["__doc__"]
    if dic.get("__module__") is not None:
        del dic["__module__"]
    if dic.get("__dict__") is not None:
        del dic["__dict__"]
    if dic.get("__weakref__") is not None:
        del dic["__weakref__"]

    with open(path, "w") as f:
        json.dump(dic, f)

    return dic


def load_config(config_path):
    config = Config(json.load(open(config_path, "r")))
    return config


# TTA
def hflip(data):
    w = data.shape[-1]
    return data[..., torch.arange(w - 1, -1, -1, device=data.device)]


def vflip(data):
    h = data.shape[-2]
    return data[..., torch.arange(h - 1, -1, -1, device=data.device), :]


def rot90(data):
    rotated = torch.rot90(data, k=3, dims=(2, 3))  # k=3 means 90 clockwise
    return rotated


def rot180(data):
    rotated = torch.rot90(data, k=2, dims=(2, 3))  # k=2 means 180
    return rotated


def rot270(data):
    rotated = torch.rot90(data, k=1, dims=(2, 3))  # Rotate 270 degrees clockwise
    return rotated
