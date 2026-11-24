from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn, arange, cat, is_tensor, rand, randn, randn_like, tensor, where, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# ein equations

import einx
from einops import einsum, rearrange, repeat, pack, unpack
from torch_einops_utils import cast_tensor, clamp, device_dtype as dd
from torch_einops_utils.shape import size, shape

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def l2norm(t):
    return F.normalize(t, dim = -1)

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
        rand_uniform = rand_uniform.to(**dd(discount)).expand(out_shape)
    else:
        rand_uniform = rand(out_shape, **dd(discount), generator = generator)

    # inverse cdf

    num_steps = (max_steps + int(allow_zero)).clamp(min = 1)
    max_zero_based = num_steps - 1

    is_zero = discount == 0.
    is_undiscounted = discount == 1.

    safe_discount = where(is_zero | is_undiscounted, 0.5, discount)

    log_discount = where(safe_discount < 0.5, safe_discount.log(), torch.log1p(safe_discount - 1.))
    prob_reach = -torch.expm1(num_steps * log_discount)

    delta = where(
        is_undiscounted,
        rand_uniform * num_steps,
        torch.log1p(-rand_uniform * prob_reach) / log_discount
    )

    delta = where(is_zero, 0., delta)

    # clamp before long cast

    delta = delta.floor().clamp(min = 0.).minimum(max_zero_based).long().minimum(max_zero_based)

    return delta if allow_zero else (delta + 1)

# bilinear contrastive learning

def bilinear_contrastive(
    embed: Tensor,
    future_embed: Tensor,
    skill: Tensor,
    *,
    temperature: float = 0.1,
    return_loss: bool = True
) -> Tensor:
    assert temperature > 0.

    delta = future_embed - embed

    if not return_loss:
        return einsum(delta, skill, '... d, ... d -> ...') / temperature

    # asymmetric infonce over skills

    batch, dim = shape(delta, 'b d')
    shape(skill, 'b d', b = batch, d = dim)

    logits = einsum(delta, skill, 'i d, j d -> i j') / temperature
    labels = arange(batch, device = delta.device)

    return F.cross_entropy(logits, labels)

# skill prior and effective reward

def sample_skills(
    sample_shape: tuple[int, ...] | int,
    dim_skill: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32
) -> Tensor:
    return l2norm(randn((*cast_tuple(sample_shape), dim_skill), device = device, dtype = dtype))

def effective_reward(
    anchor_embed: Tensor,
    next_embed: Tensor,
    skill: Tensor,
    *,
    temperature: float = 0.1,
    partition_weight: float = 1.,
    num_samples: int = 1024
) -> Tensor:
    assert 0. <= partition_weight <= 1.

    pos = bilinear_contrastive(anchor_embed, next_embed, skill, temperature = temperature, return_loss = False)

    if partition_weight == 0.:
        return pos

    assert num_samples >= 2

    # negative skills on sphere

    dim = size(anchor_embed, '... [d]')
    neg_skills = sample_skills(num_samples - 1, dim, **dd(anchor_embed))

    neg = einsum(next_embed - anchor_embed, neg_skills, '... d, k d -> ... k') / temperature

    # log mean exp over pos and negs

    logits = cat((rearrange(pos, '... -> ... 1'), neg), dim = -1)
    log_partition = logits.logsumexp(dim = -1) - math.log(num_samples)

    return pos - partition_weight * log_partition

# classes

class EffectiveReward(Module):
    def __init__(
        self,
        total_steps: int,
        *,
        anneal_fracs: tuple[float, float] = (0.3, 0.8),
        temperature: float = 0.1,
        num_samples: int = 1024
    ):
        super().__init__()
        start, end = anneal_fracs
        assert total_steps > 0 and 0. <= start <= end <= 1.

        self.total_steps = total_steps
        self.anneal_fracs = anneal_fracs
        self.temperature = temperature
        self.num_samples = num_samples

        self.register_buffer('step', tensor(0))

    def set_step_(self, step: int):
        assert step >= 0
        self.step.fill_(step)
        return self

    @property
    def partition_weight(self) -> float:
        start, end = self.anneal_fracs
        progress = self.step.item() / self.total_steps

        if start == end:
            return float(progress >= end)

        return clamp((progress - start) / (end - start), 0., 1.)

    def forward(
        self,
        anchor_embed: Tensor,
        next_embed: Tensor,
        skill: Tensor,
        *,
        partition_weight: float | None = None,
        increment_step: bool | int = True
    ) -> Tensor:
        reward = effective_reward(
            anchor_embed,
            next_embed,
            skill,
            temperature = self.temperature,
            partition_weight = default(partition_weight, self.partition_weight),
            num_samples = self.num_samples
        )

        if self.training and increment_step:
            self.step += int(increment_step)

        return reward

class SkillClassifier(Module):
    def __init__(
        self,
        state_encoder: Module,
        total_steps: int,
        *,
        anneal_fracs: tuple[float, float] = (0.3, 0.8),
        temperature: float = 0.1,
        noise_std: float = 0.,
        num_samples: int = 1024,
        extract_obs: Callable[[Tensor], Tensor] | None = None
    ):
        super().__init__()
        assert noise_std >= 0.

        self.state_encoder = state_encoder
        self.temperature = temperature
        self.noise_std = noise_std
        self.extract_obs = default(extract_obs, identity)

        self.effective_reward = EffectiveReward(
            total_steps = total_steps,
            anneal_fracs = anneal_fracs,
            temperature = temperature,
            num_samples = num_samples
        )

    @property
    def partition_weight(self) -> float:
        return self.effective_reward.partition_weight

    def set_step_(self, step: int):
        self.effective_reward.set_step_(step)
        return self

    def encode_state(
        self,
        state: Tensor,
        *,
        add_noise: bool = False
    ) -> Tensor:
        state = self.extract_obs(state)

        if add_noise and self.noise_std > 0.:
            state = state + randn_like(state) * self.noise_std

        return self.state_encoder(state)

    @torch.no_grad()
    def reward(
        self,
        anchor: Tensor,
        next_state: Tensor,
        skill: Tensor,
        *,
        partition_weight: float | None = None,
        increment_step: bool | int = True
    ) -> Tensor:
        anchor_embed = self.encode_state(anchor)
        next_embed = self.encode_state(next_state)

        return self.effective_reward(
            anchor_embed,
            next_embed,
            skill,
            partition_weight = partition_weight,
            increment_step = increment_step
        )

    def forward(
        self,
        state: Tensor,
        future_state: Tensor,
        skill: Tensor,
        *,
        add_noise: bool | None = None,
        return_loss: bool = True
    ) -> Tensor:
        add_noise = default(add_noise, self.training)

        embed = self.encode_state(state, add_noise = add_noise)
        future_embed = self.encode_state(future_state, add_noise = add_noise)

        return bilinear_contrastive(
            embed,
            future_embed,
            skill,
            temperature = self.temperature,
            return_loss = return_loss
        )

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
