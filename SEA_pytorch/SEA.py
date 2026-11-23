from __future__ import annotations

import torch
from torch import nn, is_tensor, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# ein equations

import einx
from einops import rearrange, repeat, pack, unpack
from torch_einops_utils import cast_tensor
from torch_einops_utils.shape import size, shape

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# truncated geometric sampling

def sample_truncated_geometric(
    max_steps: Tensor | int,
    discount: float | Tensor,
    *,
    allow_zero: bool = False,
    sample_shape: tuple[int, ...] | int = (),
    rand_uniform: Tensor | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None
) -> Tensor:
    # device

    device = default(device, next((t.device for t in (max_steps, discount, rand_uniform) if is_tensor(t)), None))

    max_steps = cast_tensor(max_steps, device = device, dtype = torch.long)
    discount = cast_tensor(discount, device = device, dtype = torch.float32)

    assert ((discount >= 0.) & (discount <= 1.)).all(), 'discount must be in [0, 1]'

    # uniform draws

    out_shape = (*cast_tuple(sample_shape), *torch.broadcast_shapes(max_steps.shape, discount.shape))

    if exists(rand_uniform):
        out_shape = torch.broadcast_shapes(out_shape, rand_uniform.shape)
        rand_uniform = rand_uniform.to(device = device, dtype = torch.float32).expand(out_shape)
    else:
        rand_uniform = torch.rand(out_shape, device = device, dtype = torch.float32, generator = generator)

    # inverse cdf

    num_steps = (max_steps + int(allow_zero)).clamp(min = 1)
    max_zero_based = num_steps - 1

    is_zero = discount == 0.
    is_undiscounted = discount == 1.

    safe_discount = torch.where(is_zero | is_undiscounted, 0.5, discount)

    log_discount = torch.where(safe_discount < 0.5, safe_discount.log(), torch.log1p(safe_discount - 1.))
    prob_reach = -torch.expm1(num_steps * log_discount)

    delta = torch.where(
        is_undiscounted,
        rand_uniform * num_steps,
        torch.log1p(-rand_uniform * prob_reach) / log_discount
    )

    delta = torch.where(is_zero, 0., delta)

    # clamp before long cast

    delta = delta.floor().clamp(min = 0.).minimum(max_zero_based).long().minimum(max_zero_based)

    return delta if allow_zero else (delta + 1)

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
