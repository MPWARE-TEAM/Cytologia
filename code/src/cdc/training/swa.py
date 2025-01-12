from collections import OrderedDict
from os.path import join as pjoin
import glob

import torch


def swa_and_save(path: str, pattern="best_*.ckpt"):
    swa_chkp = OrderedDict({"state_dict": None})
    checkpoints = glob.glob(pjoin(path, pattern))
    for checkpoint in checkpoints:
        temp_chkp = torch.load(checkpoint, map_location="cpu")
        if swa_chkp['state_dict'] is None:
            swa_chkp['state_dict'] = temp_chkp['state_dict']
        else:
            for k in swa_chkp['state_dict'].keys():
                if isinstance(swa_chkp['state_dict'][k], torch.FloatTensor):
                    swa_chkp['state_dict'][k] += temp_chkp['state_dict'][k]

    for k in swa_chkp['state_dict'].keys():
        if isinstance(swa_chkp['state_dict'][k], torch.FloatTensor):
            swa_chkp['state_dict'][k] /= len(checkpoints)
    swa_chkp["epoch"] = 0
    num_best = len(checkpoints)
    swa_checkpoint_name = f"swa_{num_best}_best.pt"
    torch.save(swa_chkp, pjoin(path, swa_checkpoint_name))
