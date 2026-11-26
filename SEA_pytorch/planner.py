from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import full_like, randn, zeros, zeros_like, Tensor
from torch.nn import Module

import einx
from einops import rearrange, repeat
from torch_einops_utils import batched_index_select, device_dtype as dd, pad_at_dim

# helpers

def exists(v):
    return v is not None

# latent-space goal scoring & CEM planner

def latent_goal_score(
    z_seq: Tensor,
    z_goal: Tensor,
    *,
    step_weight: float = 0.25
) -> Tensor:
    assert step_weight >= 0.
    diff = einx.subtract('b ... d, b d -> b ... d', z_seq, z_goal)
    dist_sq = einx.mean('... [d]', diff.square())
    return -(dist_sq[..., -1] + step_weight * einx.mean('... [h]', dist_sq))

class CEMPlan(NamedTuple):
    action: Tensor
    mean: Tensor
    std: Tensor
    elite_scores: Tensor

class CEMPlanner(Module):
    def __init__(
        self,
        world_model: Module,
        dim_action: int,
        *,
        horizon: int = 12,
        pop_size: int = 256,
        generations: int = 4,
        elite_frac: float = 0.1,
        action_range: tuple[float, float] = (-1., 1.),
        init_std: float = 0.7,
        min_std: float = 0.05,
        step_weight: float = 0.25
    ):
        super().__init__()
        lo, hi = action_range
        assert dim_action >= 1 and horizon >= 1 and pop_size >= 2 and generations >= 1
        assert 0. < elite_frac <= 1. and lo < hi and init_std > 0. and min_std > 0.

        self.world_model = world_model
        self.dim_action = dim_action
        self.horizon = horizon
        self.pop_size = pop_size
        self.generations = generations
        self.num_elites = max(1, int(pop_size * elite_frac))
        self.action_range = action_range
        self.init_std = float(init_std)
        self.min_std = float(min_std)
        self.step_weight = float(step_weight)

    @staticmethod
    def shift_warm_start(mean: Tensor, commit_k: int = 1) -> Tensor:
        assert commit_k >= 1
        if commit_k >= mean.shape[-2]:
            return zeros_like(mean)
        return pad_at_dim(mean[..., commit_k:, :], (0, commit_k), dim = -2, value = 0.)

    @torch.no_grad()
    def plan(
        self,
        z_cur: Tensor,
        score_fn: Callable[[Tensor], Tensor],
        *,
        warm_mean: Tensor | None = None,
        generator: torch.Generator | None = None
    ) -> CEMPlan:
        unbatched = z_cur.ndim == 1
        if unbatched:
            z_cur = rearrange(z_cur, 'd -> 1 d')

        batch = z_cur.shape[0]
        lo, hi = self.action_range

        if exists(warm_mean):
            mean = rearrange(warm_mean, 'h a -> 1 h a') if (unbatched and warm_mean.ndim == 2) else warm_mean
            mean = mean.to(**dd(z_cur)).clone()
        else:
            mean = zeros((batch, self.horizon, self.dim_action), **dd(z_cur))

        std = full_like(mean, self.init_std)
        z0_rep = repeat(z_cur, 'b d -> (b p) d', p = self.pop_size)
        elite_scores = zeros((batch, self.num_elites), **dd(z_cur))

        for _ in range(self.generations):
            eps = randn((batch, self.pop_size, self.horizon, self.dim_action), **dd(z_cur), generator = generator)
            actions = einx.add('b h a, b p h a -> b p h a', mean, einx.multiply('b h a, b p h a -> b p h a', std, eps)).clamp(lo, hi)

            z_seq = rearrange(
                self.world_model.rollout(z0_rep, rearrange(actions, 'b p h a -> (b p) h a')),
                '(b p) h d -> b p h d',
                b = batch
            )

            elite_scores, top_idx = score_fn(z_seq).topk(self.num_elites, dim = 1)
            elites = batched_index_select(actions, top_idx, dim = 1)

            mean = einx.mean('b [k] h a', elites)
            std = elites.std(dim = 1, unbiased = self.num_elites > 1).clamp(min = self.min_std) if self.num_elites > 1 else full_like(mean, self.min_std)

        action = mean[:, 0]
        if unbatched:
            return CEMPlan(action[0], mean[0], std[0], elite_scores[0])

        return CEMPlan(action, mean, std, elite_scores)

    @torch.no_grad()
    def plan_to_latent(
        self,
        z_cur: Tensor,
        z_goal: Tensor,
        *,
        warm_mean: Tensor | None = None,
        generator: torch.Generator | None = None
    ) -> CEMPlan:
        goal = rearrange(z_goal, 'd -> 1 d') if (z_cur.ndim == 1 and z_goal.ndim == 1) else z_goal
        return self.plan(
            z_cur,
            lambda z_seq: latent_goal_score(z_seq, goal, step_weight = self.step_weight),
            warm_mean = warm_mean,
            generator = generator
        )

    @torch.no_grad()
    def plan_obs(
        self,
        obs_cur: Tensor,
        obs_goal: Tensor | None = None,
        *,
        score_fn: Callable[[Tensor], Tensor] | None = None,
        warm_mean: Tensor | None = None,
        generator: torch.Generator | None = None
    ) -> CEMPlan:
        assert exists(obs_goal) ^ exists(score_fn), 'provide exactly one of obs_goal or score_fn'
        z_cur = self.world_model.encode_detached(obs_cur)
        if exists(obs_goal):
            return self.plan_to_latent(z_cur, self.world_model.encode_detached(obs_goal), warm_mean = warm_mean, generator = generator)
        return self.plan(z_cur, score_fn, warm_mean = warm_mean, generator = generator)

    def forward(
        self,
        obs_cur: Tensor,
        obs_goal: Tensor | None = None,
        *,
        score_fn: Callable[[Tensor], Tensor] | None = None,
        warm_mean: Tensor | None = None,
        generator: torch.Generator | None = None
    ) -> CEMPlan:
        return self.plan_obs(obs_cur, obs_goal = obs_goal, score_fn = score_fn, warm_mean = warm_mean, generator = generator)
