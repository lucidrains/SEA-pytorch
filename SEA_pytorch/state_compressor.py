from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from math import ceil, log, sqrt
import random
from typing import Literal, NamedTuple

import torch
from torch import nn, arange, cat, eye, linspace, randn, stack, where, Tensor
from torch.nn import Module, Linear, LayerNorm, Sequential
import torch.nn.functional as F
from torch.linalg import qr, solve_triangular, svdvals

import einx
from einops import einsum, repeat
from einops.layers.torch import Rearrange
from torch_einops_utils import (
    device_dtype as dd,
    fallback_from,
    masked_mean,
    pack_with_inverse,
    pad_at_dim,
    z_score
)
from torch_einops_utils.shape import shape, size

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t, dim = -1):
    return F.normalize(t, p = 2, dim = dim)

def safe_div(num: Tensor, den: Tensor, eps: float = 1e-4) -> Tensor:
    return einx.divide('... d, ... -> ... d', num, den.clamp(min = eps))

# SIGReg - Balestriero & LeCun, LeJEPA (https://arxiv.org/abs/2511.08544)

class SIGReg(Module):
    def __init__(
        self,
        num_slices: int = 256,
        num_knots: int = 17,
        t_max: float = 3.
    ):
        super().__init__()
        assert num_slices >= 1 and num_knots >= 2 and t_max > 0.
        self.num_slices = num_slices

        t = linspace(0., t_max, num_knots)
        dt = t_max / (num_knots - 1)
        phi = (-0.5 * t.square()).exp()

        weights = torch.full((num_knots,), 2. * dt)
        weights[0] = weights[-1] = dt

        self.register_buffer('t', t, persistent = False)
        self.register_buffer('phi', phi, persistent = False)
        self.register_buffer('weights', weights * phi, persistent = False)

    def forward(
        self,
        z: Tensor,
        *,
        generator: torch.Generator | None = None
    ) -> Tensor:
        batch, dim = shape(z, '[b] ... [d]')

        proj = l2norm(randn((dim, self.num_slices), **dd(z), generator = generator), dim = 0)
        tx = einsum(z, proj, self.t, 'b ... d, d m, k -> b ... m k')

        err_sq = (einx.mean('[b] ... m k', tx.cos()) - self.phi).square() + einx.mean('[b] ... m k', tx.sin()).square()
        return (batch * einsum(err_sq, self.weights, '... m k, k -> ... m')).mean()

# ReservoirEpiplexity - Yanbo Zhang & Michael Levin (https://arxiv.org/abs/2607.18433)

def logdet_epiplexity(t: Tensor, scale: float = 1.) -> Tensor:
    if size(t, '... [d]') == 1:
        sv_sq = einx.sum('... [n] d -> ... d', t.square())
    else:
        # rescale to avoid subnormal svd nans
        max_val = t.detach().abs().amax(dim = (-2, -1), keepdim = True).clamp(min = 1e-12)
        sv_sq = (svdvals(t / max_val) * max_val.squeeze(-1)).square()

    return einx.sum('... [d]', 0.5 * torch.log1p(scale * sv_sq) / log(2.))

@lru_cache(maxsize = 64)
@torch.no_grad()
def deterministic_orthogonal(dim: int) -> Tensor:
    rng = random.Random(dim)
    mat = torch.tensor([[rng.gauss(0., 1.) for _ in range(dim)] for _ in range(dim)], dtype = torch.float64)
    q, r = qr(mat)
    return q * where(r.diagonal() >= 0., 1., -1.)

def sketched_logdet_diff(both: Tensor, scale: float, max_dim: int) -> Tensor:
    # split high-dim targets into orthogonal blocks to avoid noise leak
    dim = size(both, 's n [d]')

    if dim <= max_dim:
        total, loo = logdet_epiplexity(both, scale)
        return total - loo

    rot = deterministic_orthogonal(dim).to(**dd(both))
    blocks = torch.tensor_split(einsum(both, rot, 's n d, d e -> s n e'), ceil(dim / max_dim), dim = -1)
    block_dim = size(blocks[0], 's n [d]')

    sketches = stack([
        pad_at_dim(b, (0, block_dim - b.shape[-1]), dim = -1) * sqrt(dim / b.shape[-1])
        for b in blocks
    ], dim = 1)

    total, loo = logdet_epiplexity(sketches, scale)
    weights = both.new_tensor([b.shape[-1] / dim for b in blocks])
    return einsum(total - loo, weights, 'k, k ->')

@fallback_from('mps')
def ridge_epiplexity(
    reservoir_out: Tensor,
    targets: Tensor,
    *,
    ridge_penalty: float,
    target_scale: float,
    eta: float,
    use_loo: bool,
    loo_min_sample_dim_ratio: int | None,
    z_score_eps: float
) -> Tensor:
    num_samples, dim = shape(reservoir_out, 'n d')

    H = z_score(reservoir_out, dim = 0, eps = z_score_eps) * (dim ** -0.5)
    Y = einx.subtract('n d, d -> n d', targets, einx.mean('[n] d', targets)) / target_scale

    # differentiable ridge solve via augmented QR, eq (21)-(22)

    I = eye(dim, **dd(H))
    Q, R = qr(cat((H, I * sqrt(ridge_penalty))))
    Q = Q[:num_samples]
    rhs = einsum(Q, Y, 'n d, n e -> d e')

    if not use_loo:
        W = solve_triangular(R, rhs, upper = True)
        return logdet_epiplexity(W, eta)

    residuals = Y - einsum(Q, rhs, 'n d, d e -> n e')
    error_scale = 1. - einx.sum('n [d]', Q.square()) - 1. / num_samples
    loo_residuals = safe_div(residuals, error_scale)

    max_dim = max(1, int(num_samples // loo_min_sample_dim_ratio)) if exists(loo_min_sample_dim_ratio) else size(Y, 'n [e]')
    return sketched_logdet_diff(stack((Y, loo_residuals)), eta / num_samples, max_dim)

class ReservoirEpiplexity(Module):
    def __init__(
        self,
        *,
        dim_data: int,
        dim_target: int | None = None,
        reservoir: Module | None = None,
        mlp_reservoir_dim: int = 32,
        mlp_reservoir_layers: int = 4,
        mlp_reservoir_activation: Module | None = None,
        ridge_penalty: float = 0.03,
        target_scale: float = 1.,
        eta: float = 1.,
        channel_first: bool = False,
        use_loo: bool = True,
        loo_min_sample_dim_ratio: int | None = 4
    ):
        super().__init__()
        assert not exists(loo_min_sample_dim_ratio) or loo_min_sample_dim_ratio >= 1

        self.has_custom_reservoir = exists(reservoir)

        if not self.has_custom_reservoir:
            assert mlp_reservoir_layers > 1
            dim = mlp_reservoir_dim
            act = default(mlp_reservoir_activation, nn.ELU())

            reservoir = Sequential(
                Linear(dim_data, dim),
                LayerNorm(dim, elementwise_affine = False),
                deepcopy(act),
                *[Sequential(Linear(dim, dim), LayerNorm(dim, elementwise_affine = False), deepcopy(act)) for _ in range(mlp_reservoir_layers - 2)],
                Linear(dim, dim)
            )

        self.reservoir = reservoir.requires_grad_(False)
        self.to_samples = Rearrange('n d ... -> (n ...) d' if channel_first else '... d -> (...) d')

        self.target_scale = target_scale
        self.ridge_penalty = ridge_penalty
        self.eta = eta
        self.use_loo = use_loo
        self.loo_min_sample_dim_ratio = loo_min_sample_dim_ratio

    def forward(
        self,
        data: Tensor,
        targets: Tensor,
        z_score_eps: float = 1e-6
    ) -> Tensor:
        self.reservoir.eval()
        reservoir_out = self.to_samples(self.reservoir(data)) if self.has_custom_reservoir else self.reservoir(self.to_samples(data))
        targets = self.to_samples(targets)

        compute_dtype = torch.float64 if torch.float64 in (reservoir_out.dtype, targets.dtype) else torch.float32

        return ridge_epiplexity(
            reservoir_out.to(compute_dtype),
            targets.to(compute_dtype),
            ridge_penalty = self.ridge_penalty,
            target_scale = self.target_scale,
            eta = self.eta,
            use_loo = self.use_loo,
            loo_min_sample_dim_ratio = self.loo_min_sample_dim_ratio,
            z_score_eps = z_score_eps
        )

# state compressor: next-state JEPA prediction + ReservoirEpiplexity (or SIGReg)

class CompressorLossBreakdown(NamedTuple):
    pred: Tensor
    reg: Tensor

class StateCompressor(Module):
    def __init__(
        self,
        encoder: Module,
        dim_latent: int,
        *,
        dim_action: int | None = None,
        num_discrete_actions: int | None = None,
        dim_action_embed: int | None = None,
        dim_encoder_out: int | None = None,
        predictor_hidden_dim: int = 64,
        obs_ndim: int | None = None,
        regularizer: Literal['epiplexity', 'sigreg'] = 'epiplexity',
        dynamics_loss_weight: float = 1.,
        rollout_discount: float = 0.8,
        reg_weight: float | None = None,
        l2norm_latent: bool | None = None,
        **reg_kwargs
    ):
        super().__init__()
        assert regularizer in ('epiplexity', 'sigreg')
        assert dim_latent >= 1 and dynamics_loss_weight >= 0. and 0. < rollout_discount <= 1.
        assert not (exists(dim_action) and exists(num_discrete_actions))

        self.dim_latent = dim_latent
        self.obs_ndim = obs_ndim
        self._obs_pack_pattern = ('* ' + ' '.join(f'd{i}' for i in range(obs_ndim))) if exists(obs_ndim) else None
        self.regularizer_type = regularizer
        self.dynamics_loss_weight = dynamics_loss_weight
        self.rollout_discount = rollout_discount
        self.reg_weight = default(reg_weight, 0.1 if regularizer == 'epiplexity' else 0.01)
        self.l2norm_latent = default(l2norm_latent, regularizer == 'epiplexity')
        assert self.reg_weight >= 0. and not (self.l2norm_latent and dim_latent < 2)

        self.encoder = Sequential(
            encoder,
            Linear(dim_encoder_out, dim_latent) if exists(dim_encoder_out) else nn.Identity()
        )

        if regularizer == 'epiplexity':
            epx_kwargs = dict(mlp_reservoir_dim = 64, eta = 20., use_loo = True)
            epx_kwargs.update(reg_kwargs)
            self.regularizer = ReservoirEpiplexity(dim_data = dim_latent, dim_target = dim_latent, **epx_kwargs)
        else:
            self.regularizer = SIGReg(**reg_kwargs)

        if exists(num_discrete_actions):
            eff_dim_action = default(dim_action_embed, dim_latent)
            self.to_action_embed = nn.Embedding(num_discrete_actions, eff_dim_action)
        else:
            eff_dim_action = dim_action
            self.to_action_embed = None

        self.predictor = (
            Sequential(
                Linear(dim_latent + eff_dim_action, predictor_hidden_dim),
                nn.GELU(),
                Linear(predictor_hidden_dim, dim_latent)
            )
            if (dynamics_loss_weight > 0. and exists(eff_dim_action)) else None
        )

    def _normalize(self, z: Tensor) -> Tensor:
        return (l2norm(z) * sqrt(self.dim_latent)) if self.l2norm_latent else z

    def encode(self, obs: Tensor) -> Tensor:
        if exists(self._obs_pack_pattern):
            packed, unpack_obs = pack_with_inverse(obs, self._obs_pack_pattern)
            return unpack_obs(self._normalize(self.encoder(packed)), '* d')

        return self._normalize(self.encoder(obs))

    @torch.no_grad()
    def encode_detached(self, obs: Tensor) -> Tensor:
        return self.encode(obs)

    def predict_next(self, latent: Tensor, action: Tensor) -> Tensor:
        assert exists(self.predictor)
        act = self.to_action_embed(action.long()) if exists(self.to_action_embed) else action
        return latent + self.predictor(cat((latent, act), dim = -1))

    def rollout(self, latent: Tensor, actions: Tensor) -> Tensor:
        preds = []

        for action in actions.unbind(dim = 1):
            latent = self.predict_next(latent, action)
            preds.append(latent)

        return stack(preds, dim = 1)

    def loss(
        self,
        obs_seq: Tensor,
        actions: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        return_breakdown: bool = False
    ) -> Tensor | tuple[Tensor, CompressorLossBreakdown]:
        batch, num_obs = shape(obs_seq, '[b] [t] ...')
        horizon = num_obs - 1
        assert num_obs >= 2 and (not exists(actions) or size(actions, 'b [h] ...') == horizon)

        use_dynamics = self.dynamics_loss_weight > 0. and exists(self.predictor) and exists(actions)

        frames = obs_seq if (use_dynamics or self.regularizer_type == 'sigreg') else obs_seq[:, :2]
        h = self._normalize(self.encoder(frames.flatten(0, 1))).unflatten(0, frames.shape[:2])
        h_0, h_1 = h[:, 0], h[:, 1]

        # anti-collapse regularizer (maximize LOO epiplexity or minimize SIGReg)

        if self.regularizer_type == 'epiplexity':
            if exists(mask):
                valid = mask[:, 0]
                reg_val = self.regularizer(h_0[valid], h_1[valid]) if valid.sum() >= 2 else h_0.new_zeros(())
            else:
                reg_val = self.regularizer(h_0, h_1)

            reg_term = -self.reg_weight * reg_val
        else:
            reg_latents = h_0 if exists(mask) else stack((h[:, 0], h[:, -1]), dim = 1)
            reg_val = self.regularizer(reg_latents)
            reg_term = self.reg_weight * reg_val

        # optional discounted multi-step next-state rollout loss

        if use_dynamics:
            pred_mse = einx.mean('b h [d]', (self.rollout(h_0, actions) - h[:, 1:]).square())
            weights = self.rollout_discount ** arange(horizon, **dd(pred_mse))
            weights = einx.where('b h, h, -> b h', mask, weights, 0.) if exists(mask) else repeat(weights, 'h -> b h', b = batch)
            pred_loss = masked_mean(pred_mse, weights)
        else:
            pred_loss = reg_val.new_zeros(())

        total_loss = self.dynamics_loss_weight * pred_loss + reg_term

        if not return_breakdown:
            return total_loss

        return total_loss, CompressorLossBreakdown(pred = pred_loss.detach(), reg = reg_val.detach())

    def forward(
        self,
        obs: Tensor,
        actions: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        return_breakdown: bool = False
    ) -> Tensor | tuple[Tensor, CompressorLossBreakdown]:
        # drop-in state_compressor when called with obs alone; training loss when actions are passed

        if not exists(actions):
            return self.encode(obs)

        return self.loss(obs, actions, mask = mask, return_breakdown = return_breakdown)
