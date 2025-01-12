import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class GeM(nn.Module):
    def __init__(self, pooling_config):
        super().__init__()

        self.eps = pooling_config["eps"]
        self.p = nn.Parameter(torch.ones(1) * pooling_config["p"])  # Learnable parameter

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)
        return x.view(batch_size, -1)  # torch.Size([BS, FEATS])

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()}, eps={self.eps})"


def get_pooling_layer(config):
    if config.pooling_type == 'GeM':
        return GeM(config.gem_pooling)

    else:
        raise ValueError(f'Invalid pooling type: {config.pooling_type}')
