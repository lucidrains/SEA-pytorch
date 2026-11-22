from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# ein equations

import einx
from einops import rearrange, repeat, pack, unpack
from torch_einops_utils.shape import size, shape

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class SEA(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(
        self,
        x
    ):
        return x
