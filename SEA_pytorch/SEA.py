from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch import nn, arange, cat, is_tensor, rand, randn, randn_like, stack, tensor, where, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# ein equations

import einx
from einops import einsum, rearrange, repeat, pack, unpack
from torch_einops_utils import (
    batched_index_select,
    cast_tensor,
    clamp,
    device_dtype as dd,
    pad_right_ndim_to,
    repeat_interleave_to_match
)
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

# causal state / action chunking

def causal_chunks(
    seq: Tensor,
    chunk_len: int,
    *,
    indices: Tensor | int | None = None,
    stride: int = 1,
    offset: int = 0,
    pad_value: float | int | None = None,
    return_mask: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    assert chunk_len >= 1 and stride >= 1

    batch, seq_len = shape(seq, '[b] [t] ...')
    device = seq.device

    if not exists(indices):
        t = repeat(arange(seq_len, device = device), 't -> b t', b = batch)
    else:
        t = cast_tensor(indices, device = device, dtype = torch.long)
        t = repeat(t, '-> b', b = batch) if t.ndim == 0 else repeat_interleave_to_match(t, batch, dim = 0)

    rel = arange(-(chunk_len - 1), 1, device = device) * stride + offset
    raw_idx = einx.add('..., c -> ... c', t, rel)
    mask = (raw_idx >= 0) & (raw_idx < seq_len)

    chunks = batched_index_select(seq, raw_idx.clamp(0, seq_len - 1), dim = 1)

    if exists(pad_value):
        chunks = where(pad_right_ndim_to(mask, chunks.ndim), chunks, pad_value)

    return (chunks, mask) if return_mask else chunks

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
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None
) -> Tensor:
    return l2norm(randn((*cast_tuple(sample_shape), dim_skill), device = device, dtype = dtype, generator = generator))

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

# empowerment estimation

def estimate_empowerment(
    state: Tensor,
    value_fn: Callable[..., Tensor | tuple[Tensor, ...]],
    dim_skill: int,
    *,
    num_skills: int = 8,
    policy: Callable[..., Tensor | tuple[Tensor, ...]] | None = None,
    anchor: Tensor | None = None,
    skills: Tensor | None = None,
    discount: float = 0.,
    clamp_min: float | None = None
) -> Tensor:
    anchor = default(anchor, state)
    batch = size(state, '[b] ...')

    if not exists(skills):
        skills = sample_skills((batch, num_skills), dim_skill, **dd(state))
    elif skills.ndim == 2:
        skills = repeat(skills, 'k d -> b k d', b = batch)

    k = size(skills, 'b [k] d', b = batch, d = dim_skill)

    state, anchor = (repeat(t, 'b ... -> (b k) ...', k = k) for t in (state, anchor))
    skills = rearrange(skills, 'b k d -> (b k) d')

    args = (state, anchor, skills)

    if exists(policy):
        action, *_ = cast_tuple(policy(*args))
        args = (*args, action)

    values = value_fn(*args)

    if isinstance(values, tuple):
        values = einx.min('[c] ...', stack(values))

    if values.ndim == 2:
        values = einx.max('n [a]', values)

    # main contribution - eq. (7)-(9), appendix A.3.2
    # average skill values anchored at the current state over randomly sampled skills to lower-bound potential empowerment

    empowerment = einx.mean('(b [k]) -> b', values, b = batch) * (1. - discount)

    return clamp(empowerment, lo = clamp_min)

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
        action_encoder: Module | None = None,
        anneal_fracs: tuple[float, float] = (0.3, 0.8),
        temperature: float = 0.1,
        noise_std: float = 0.,
        action_noise_std: float | None = None,
        num_samples: int = 1024,
        extract_obs: Callable[[Tensor], Tensor] | None = None,
        extract_action: Callable[[Tensor], Tensor] | None = None,
        state_compressor: Callable[[Tensor], Tensor] | None = None,
        detach_state_compressor: bool = True
    ):
        super().__init__()
        action_noise_std = default(action_noise_std, noise_std)
        assert noise_std >= 0. and action_noise_std >= 0.

        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.temperature = temperature
        self.noise_std = noise_std
        self.action_noise_std = action_noise_std
        self.extract_obs = default(extract_obs, identity)
        self.extract_action = default(extract_action, identity)
        self.detach_state_compressor = detach_state_compressor

        # keep detached compressors out of _modules so classifier optimizers and .train() don't touch them

        self._detached_state_compressor = None
        self._attached_state_compressor = None

        if exists(state_compressor):
            if detach_state_compressor:
                self._detached_state_compressor = (state_compressor,)
            else:
                self._attached_state_compressor = state_compressor

        self.effective_reward = EffectiveReward(
            total_steps = total_steps,
            anneal_fracs = anneal_fracs,
            temperature = temperature,
            num_samples = num_samples
        )

    @property
    def state_compressor(self) -> Callable[[Tensor], Tensor] | None:
        return self._detached_state_compressor[0] if exists(self._detached_state_compressor) else self._attached_state_compressor

    @property
    def partition_weight(self) -> float:
        return self.effective_reward.partition_weight

    def set_step_(self, step: int):
        self.effective_reward.set_step_(step)
        return self

    def encode_state(
        self,
        state: Tensor,
        actions: Tensor | None = None,
        *,
        action_mask: Tensor | None = None,
        add_noise: bool = False
    ) -> Tensor:
        state = self.extract_obs(state)

        compressor = self.state_compressor
        if exists(compressor):
            context = torch.no_grad if self.detach_state_compressor else nullcontext
            with context():
                state = compressor(state)

        if add_noise and self.noise_std > 0.:
            state = state + randn_like(state) * self.noise_std

        if not exists(actions):
            return self.state_encoder(state)

        actions = self.extract_action(actions)

        if add_noise and self.action_noise_std > 0.:
            actions = actions + randn_like(actions) * self.action_noise_std

        if exists(action_mask):
            actions = where(pad_right_ndim_to(action_mask, actions.ndim), actions, 0.)

        if not exists(self.action_encoder):
            return self.state_encoder(state, actions)

        state_embed = self.state_encoder(state)
        action_embed = self.action_encoder(actions)

        if exists(action_mask):
            has_action = einx.any('... [c]', action_mask) if action_mask.ndim >= action_embed.ndim else action_mask
            action_embed = where(pad_right_ndim_to(has_action, action_embed.ndim), action_embed, 0.)

        return state_embed + action_embed

    @torch.no_grad()
    def reward(
        self,
        anchor: Tensor,
        next_state: Tensor,
        skill: Tensor,
        *,
        anchor_actions: Tensor | None = None,
        next_actions: Tensor | None = None,
        anchor_action_mask: Tensor | None = None,
        next_action_mask: Tensor | None = None,
        partition_weight: float | None = None,
        increment_step: bool | int = True
    ) -> Tensor:
        anchor_embed = self.encode_state(anchor, anchor_actions, action_mask = anchor_action_mask)
        next_embed = self.encode_state(next_state, next_actions, action_mask = next_action_mask)

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
        actions: Tensor | None = None,
        future_actions: Tensor | None = None,
        action_mask: Tensor | None = None,
        future_action_mask: Tensor | None = None,
        add_noise: bool | None = None,
        return_loss: bool = True
    ) -> Tensor:
        add_noise = default(add_noise, self.training)

        embed = self.encode_state(state, actions, action_mask = action_mask, add_noise = add_noise)
        future_embed = self.encode_state(future_state, future_actions, action_mask = future_action_mask, add_noise = add_noise)

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
