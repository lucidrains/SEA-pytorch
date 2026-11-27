# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "SEA-pytorch",
#   "sac-pytorch>=0.3.7",
#   "memmap-replay-buffer>=0.3.0",
#   "einx>=0.4.0",
#   "fire",
#   "pillow",
#   "imageio[ffmpeg]",
# ]
# [tool.uv.sources]
# SEA-pytorch = { path = ".", editable = true }
# ///

"""
Two-agent asymmetric PushT with Scaling Empowered Agents (SEA, arXiv:2610.13002).

  - Weak Agent A   (small blue)   : slow, light pusher running the SEA skill policy pi_eff(a_A | s_A, s_A0, z)
  - Strong Agent B (large orange) : fast, heavy pusher running the SEA potential policy pi_pot(a_B | s_B)

Zero task reward, zero goal pose, zero distance shaping.
Strong Agent B maximizes Weak Agent A's empowerment via either:
  - oracle (default)        : E_pot^A(s') = estimate_empowerment(s_A', Q_eff^A, pi_eff^A)
  - --predict-empowerment   : E_hat(s') predicted purely from watching Weak A's observed rollouts over time
                              (no access to Weak A's weights, skills z, actions, or SkillClassifier)

Run:
  uv run train_sea_two_agent_pusht.py
  uv run train_sea_two_agent_pusht.py --predict-empowerment --out-dir runs/sea_two_agent_predicted
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import einx
import fire
import numpy as np
import torch
from torch import nn, arange, cat, full, ones, rand, stack, tensor, where, zeros, zeros_like, Tensor
from torch.nn import Module, Linear, Sequential, GELU
import torch.nn.functional as F

from SAC_pytorch import SAC, Actor, Critic
from memmap_replay_buffer import ReplayBuffer
from SEA_pytorch import (
    SkillClassifier,
    estimate_empowerment,
    sample_skills,
    sample_truncated_geometric,
)

# constants

ACTION_DIM = 2
WEAK_STATE_DIM = 8      # [weak_xy, weak_vel, block_com, cos th, sin th]
STRONG_STATE_DIM = 14   # [strong_xy, strong_vel, block_com, cos th, sin th, weak_xy, block - strong, weak - block]
BLOCK_DIM = 4           # [block_com_x, block_com_y, cos th, sin th]

CANONICAL_START = (115.0, 256.0, 445.0, 220.0, 365.0, 256.0, 0.0)

COMPARISON_SCENARIOS = (
    ('Weak A Left',       (115.0, 256.0, 445.0, 200.0, 365.0, 256.0,  0.0)),
    ('Weak A Top-Right',  (385.0, 125.0,  95.0, 415.0, 165.0, 345.0,  0.8)),
    ('Weak A Low-Left',   (125.0, 385.0, 425.0, 100.0, 350.0, 165.0, -0.7)),
    ('Weak A Top-Center', (256.0, 115.0, 190.0, 445.0, 256.0, 370.0,  1.2)),
)

HEATMAP_LOCATIONS = (
    ((140.0, 256.0), 'Weak A at Left (140, 256)',      (110, 220, 255)),
    ((360.0, 350.0), 'Weak A at Low-Right (360, 350)', (255, 195, 95)),
)

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def l2norm(t: Tensor) -> Tensor:
    return F.normalize(t, dim = -1)

def pearson_r(a: Tensor, b: Tensor) -> float:
    ac, bc = a - a.mean(), b - b.mean()
    denom = ac.norm() * bc.norm()
    return float(((ac * bc).sum() / denom).item()) if denom > 1e-8 else 0.0

def step_opt(opt, loss: Tensor, params, max_norm: float = 10.0) -> float:
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(params, max_norm)
    opt.step()
    return float(loss.item())

@contextmanager
def eval_mode(*modules):
    active = [m for m in modules if exists(m)]
    prev = [m.training for m in active]
    for m in active:
        m.eval()
    try:
        yield
    finally:
        for m, was_training in zip(active, prev):
            m.train(was_training)

def load_faster_pusht(x_jepa_dir: str):
    path = str(Path(x_jepa_dir).expanduser().resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

    import env_faster_pusht as fp
    return fp

def make_weak_probe_states(weak_xy: Tensor, block_xy: Tensor) -> Tensor:
    n = block_xy.shape[0]
    w = weak_xy.expand(n, 2)
    return cat((w / 256.0 - 1.0, zeros(n, 2), block_xy / 256.0 - 1.0, ones(n, 1), zeros(n, 1)), dim = -1)

# environment

class TwoAgentPushTEnv:
    """Vectorized 2D PushT with an underpowered Agent A (weak) and a strong Agent B (empowerer)."""

    def __init__(
        self,
        fp_mod,
        num_envs: int,
        *,
        weak_radius: float = 11.0,
        weak_step_scale: float = 18.0,
        weak_kp: float = 65.0,
        weak_kv: float = 20.0,
        weak_mass: float = 5.0,
        strong_radius: float = 22.0,
        strong_step_scale: float = 65.0,
        strong_kp: float = 130.0,
        strong_kv: float = 20.0,
        solver_iterations: int = 5,
        seed: int | None = None,
    ):
        self.fp = fp_mod
        self.num_envs = num_envs

        self.weak_radius = weak_radius
        self.weak_step_scale = weak_step_scale
        self.weak_kp = weak_kp
        self.weak_kv = weak_kv
        self.weak_mass = weak_mass

        self.strong_radius = strong_radius
        self.strong_step_scale = strong_step_scale
        self.strong_kp = strong_kp
        self.strong_kv = strong_kv

        self.solver_iterations = solver_iterations
        self.relaxation = 0.5
        self._rng = np.random.default_rng(seed)

        self._block_centers = tensor(fp_mod.BLOCK_CENTERS, dtype = torch.float32)
        self._block_halfs = tensor(fp_mod.BLOCK_HALFS, dtype = torch.float32)
        self._block_cog = tensor(fp_mod.BLOCK_COG, dtype = torch.float32)
        self._wall_centers = tensor(fp_mod.WALL_CENTERS, dtype = torch.float32)
        self._wall_halfs = tensor(fp_mod.WALL_HALFS, dtype = torch.float32)

        self.reset(seed = seed)

    @property
    def block_pose(self) -> Tensor:
        return cat((self.block_p, self.block_theta[:, None]), dim = -1)

    def block_cog(self) -> Tensor:
        return self.block_p + self.fp.rotate(self._block_cog.expand_as(self.block_p), self.block_theta)

    def weak_state(self) -> Tensor:
        th = self.block_theta
        return cat((
            self.weak_pos / 256.0 - 1.0,
            self.weak_vel / 256.0,
            self.block_cog() / 256.0 - 1.0,
            th.cos()[:, None],
            th.sin()[:, None],
        ), dim = -1)

    def strong_state(self) -> Tensor:
        s_xy = self.strong_pos / 256.0 - 1.0
        s_vel = self.strong_vel / 256.0
        b_xy = self.block_cog() / 256.0 - 1.0
        w_xy = self.weak_pos / 256.0 - 1.0
        th = self.block_theta

        return cat((
            s_xy, s_vel, b_xy,
            th.cos()[:, None], th.sin()[:, None],
            w_xy, b_xy - s_xy, w_xy - b_xy,
        ), dim = -1)

    def _place_block_from_cog(self, cog: Tensor, angle: Tensor) -> Tensor:
        return cog - self.fp.rotate(self._block_cog.expand_as(cog), angle)

    def _sample_random_blocks(self, k: int) -> tuple[Tensor, Tensor, Tensor]:
        cog = torch.as_tensor(self._rng.uniform(115.0, 397.0, size = (k, 2)), dtype = torch.float32)
        th = torch.as_tensor(self._rng.uniform(-math.pi, math.pi, size = (k,)), dtype = torch.float32)
        return cog, th, self._place_block_from_cog(cog, th)

    def _overlaps_block(self, pos: Tensor, block_p: Tensor, block_theta: Tensor, radius: float, clearance: float = 4.0) -> Tensor:
        fp = self.fp
        u = fp.rotate(tensor((1.0, 0.0)).expand_as(block_p), block_theta)
        v = fp.rotate(tensor((0.0, 1.0)).expand_as(block_p), block_theta)
        c = block_p[:, None, :] + fp.rotate(self._block_centers, block_theta[:, None])
        zero_vel = zeros_like(pos)
        hit = zeros(pos.shape[0], dtype = torch.bool)

        for i in range(2):
            _, _, pen, *_ = fp.circle_obb_contact(pos, zero_vel, c[:, i], u, v, fp.BLOCK_HALFS[i], radius + clearance, rect_idx = i)
            hit |= (pen > 0.0)

        return hit

    def _sample_clear_positions(
        self,
        block_p: Tensor,
        block_theta: Tensor,
        radius: float,
        lo: float,
        hi: float,
        *,
        near_cog: Tensor | None = None,
        near_frac: float = 0.0,
    ) -> Tensor:
        n = block_p.shape[0]
        near_mask = torch.as_tensor(self._rng.random(n) < near_frac) if near_frac > 0.0 else None
        use_near = exists(near_cog) and exists(near_mask) and bool(near_mask.any())

        def draw():
            pos = torch.as_tensor(self._rng.uniform(lo, hi, size = (n, 2)), dtype = torch.float32)
            if use_near:
                ang = torch.as_tensor(self._rng.uniform(0.0, 2.0 * math.pi, size = (n,)), dtype = torch.float32)
                rad = torch.as_tensor(self._rng.uniform(36.0, 96.0, size = (n,)), dtype = torch.float32)
                cand = (near_cog + stack((ang.cos(), ang.sin()), dim = -1) * rad[:, None]).clamp(lo, hi)
                pos = where(near_mask[:, None], cand, pos)
            return pos

        pos = draw()
        for _ in range(32):
            bad = self._overlaps_block(pos, block_p, block_theta, radius)
            if not bad.any():
                break
            pos = where(bad[:, None], draw(), pos)

        return pos

    @torch.no_grad()
    def reset(
        self,
        seed: int | None = None,
        *,
        reset_to_state: np.ndarray | None = None,
        weak_near_frac: float = 0.0,
        strong_near_frac: float = 0.0,
    ):
        if exists(seed):
            self._rng = np.random.default_rng(seed)

        n = self.num_envs

        if exists(reset_to_state):
            st = np.asarray(reset_to_state, dtype = np.float32)
            if st.ndim == 1:
                st = np.broadcast_to(st, (n, 7))
            self.weak_pos = torch.as_tensor(st[:, 0:2].copy(), dtype = torch.float32)
            self.strong_pos = torch.as_tensor(st[:, 2:4].copy(), dtype = torch.float32)
            cog = torch.as_tensor(st[:, 4:6].copy(), dtype = torch.float32)
            self.block_theta = torch.as_tensor(st[:, 6].copy(), dtype = torch.float32)
            self.block_p = self._place_block_from_cog(cog, self.block_theta)
        else:
            cog, self.block_theta, self.block_p = self._sample_random_blocks(n)
            self.weak_pos = self._sample_clear_positions(self.block_p, self.block_theta, self.weak_radius, 85.0, 427.0, near_cog = cog, near_frac = weak_near_frac)
            self.strong_pos = self._sample_clear_positions(self.block_p, self.block_theta, self.strong_radius, 65.0, 447.0, near_cog = cog, near_frac = strong_near_frac)

        self.weak_vel = zeros(n, 2)
        self.strong_vel = zeros(n, 2)
        self.block_v = zeros(n, 2)
        self.block_w = zeros(n)
        self.v_bias = zeros(n, 2)
        self.w_bias = zeros(n)

    @torch.no_grad()
    def reset_first_k(self, k: int, *, weak_near_frac: float = 0.50):
        cog, th, bp = self._sample_random_blocks(k)
        self.block_p[:k] = bp
        self.block_theta[:k] = th
        self.weak_pos[:k] = self._sample_clear_positions(bp, th, self.weak_radius, 85.0, 427.0, near_cog = cog, near_frac = weak_near_frac)

        for buf in (self.weak_vel, self.block_v, self.block_w, self.v_bias, self.w_bias):
            buf[:k] = 0.0

    def _build_contacts(self, active: str):
        fp = self.fp
        p, th = self.block_p, self.block_theta
        c = p[:, None, :] + fp.rotate(self._block_centers, th[:, None])
        u = fp.rotate(tensor((1.0, 0.0)).expand_as(p), th)
        v = fp.rotate(tensor((0.0, 1.0)).expand_as(p), th)

        wn, wpt, wpen, wval, _ = fp.obb_wall_contacts_batched(c, u, v, self._block_halfs, self._wall_centers, self._wall_halfs)
        buckets = ([wn], [wpt], [wpen], [wval], [zeros_like(wpt)], [zeros_like(wpen)])

        def add_circle(pos: Tensor, vel: Tensor, radius: float, inv_m: float):
            for i in range(2):
                n_c, pt_c, pen_c, val_c, vo_c, _ = fp.circle_obb_contact(pos, vel, c[:, i], u, v, fp.BLOCK_HALFS[i], radius, rect_idx = i)
                for lst, item in zip(buckets, (n_c[:, None], pt_c[:, None], pen_c[:, None], val_c[:, None], vo_c[:, None], full((p.shape[0], 1), inv_m))):
                    lst.append(item)

        # Weak A collides in both phases (stationary boundary in Phase 1; finite mass in Phase 2)
        # Strong B pushes in Phase 1 and retracts out of contact in Phase 2

        is_weak = active == 'weak'
        add_circle(self.weak_pos, self.weak_vel if is_weak else zeros_like(self.weak_vel), self.weak_radius, (1.0 / self.weak_mass) if is_weak else 0.0)
        if not is_weak:
            add_circle(self.strong_pos, self.strong_vel, self.strong_radius, 0.0)

        return tuple(cat(xs, dim = 1) for xs in buckets)

    def _substep(self, action: Tensor, active: str):
        fp = self.fp
        dt = fp.DT

        if active == 'weak':
            target = (self.weak_pos + action * self.weak_step_scale).clamp(18.0, 494.0)
            self.weak_vel = self.weak_vel + (self.weak_kp * (target - self.weak_pos) - self.weak_kv * self.weak_vel) * dt
            self.weak_pos = (self.weak_pos + self.weak_vel * dt).clamp(18.0, 494.0)
        else:
            target = (self.strong_pos + action * self.strong_step_scale).clamp(28.0, 484.0)
            self.strong_vel = self.strong_vel + (self.strong_kp * (target - self.strong_pos) - self.strong_kv * self.strong_vel) * dt
            self.strong_pos = (self.strong_pos + self.strong_vel * dt).clamp(28.0, 484.0)
            self.weak_vel.zero_()

        cog = self.block_cog() + (self.block_v + self.v_bias) * dt
        theta = self.block_theta + (self.block_w + self.w_bias) * dt
        self.block_p = self._place_block_from_cog(cog, theta)
        self.block_theta = theta
        self.v_bias.zero_()
        self.w_bias.zero_()

        normal, point, pen, valid, v_other, inv_m = self._build_contacts(active)

        r = point - cog[:, None, :]
        rn = r[..., 0] * normal[..., 1] - r[..., 1] * normal[..., 0]
        n_mass = 1.0 / (1.0 / fp.BLOCK_MASS + inv_m + rn * rn / fp.BLOCK_INERTIA)
        bias = fp.BIAS_COEF / dt * torch.clamp_min(pen - fp.COLLISION_SLOP, 0.0)
        perp = stack((-r[..., 1], r[..., 0]), dim = -1)

        v, w = zeros_like(self.block_v), zeros_like(self.block_w)
        v_bias, w_bias = self.v_bias, self.w_bias
        jn, jb = zeros_like(pen), zeros_like(pen)

        for _ in range(self.solver_iterations):
            vbn = ((v_bias[:, None, :] + w_bias[:, None, None] * perp) * normal).sum(dim = -1)
            jb_new = torch.clamp_min(jb + self.relaxation * (bias - vbn) * n_mass, 0.0)
            jb_delta = where(valid, jb_new - jb, zeros_like(jb))
            v_bias = v_bias + (jb_delta[..., None] * normal).sum(dim = 1) / fp.BLOCK_MASS
            w_bias = w_bias + (jb_delta * rn).sum(dim = 1) / fp.BLOCK_INERTIA
            jb = where(valid, jb_new, jb)

            vrn = ((v[:, None, :] + w[:, None, None] * perp - v_other) * normal).sum(dim = -1)
            jn_new = torch.clamp_min(jn + self.relaxation * (-vrn) * n_mass, 0.0)
            jn_delta = where(valid, jn_new - jn, zeros_like(jn))
            v = v + (jn_delta[..., None] * normal).sum(dim = 1) / fp.BLOCK_MASS
            w = w + (jn_delta * rn).sum(dim = 1) / fp.BLOCK_INERTIA
            jn = where(valid, jn_new, jn)

            # Newton's 3rd-law reaction on Weak Agent A (finite mass resists pushing)

            if active == 'weak':
                self.weak_vel = self.weak_vel - (jn_delta[:, 16:18, None] * normal[:, 16:18, :]).sum(dim = 1) / self.weak_mass
                v_other[:, 16:18, :] = self.weak_vel[:, None, :]

        self.block_v, self.block_w = v, w
        self.v_bias, self.w_bias = v_bias, w_bias

    @torch.no_grad()
    def step(self, action: Tensor, *, active: str):
        assert active in ('weak', 'strong')
        action = action.clamp(-1.0, 1.0)
        for _ in range(self.fp.N_SUBSTEPS):
            self._substep(action, active)

# models

def extract_block_obs(s_weak: Tensor) -> Tensor:
    return cat((s_weak[..., 4:6] * 2.0, s_weak[..., 6:8]), dim = -1)

def skill_obs(s_weak: Tensor, s_weak_0: Tensor, z: Tensor) -> Tensor:
    blk_cur, blk_0 = s_weak[..., 4:8], s_weak_0[..., 4:8]
    rel_blk = blk_cur[..., :2] - s_weak[..., :2]
    return cat((s_weak, blk_0, blk_cur - blk_0, rel_blk, z), dim = -1)

class ResidualEncoder(Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()
        self.skip = Linear(dim_in, dim_out, bias = False)
        nn.init.orthogonal_(self.skip.weight, gain = 0.5)
        self.mlp = Sequential(Linear(dim_in, dim_hidden), GELU(), Linear(dim_hidden, dim_out))
        self.alpha = nn.Parameter(zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        return self.skip(x) + self.alpha.clamp(-1.0, 1.0) * self.mlp(x)

class PartnerEmpowermentPredictor(Module):
    """Black-box empowerment predictor trained solely from watching Weak A's observed (s_0, s_fut) rollouts.
    Fits p(delta_block | s_0) = N(mu(s_0), diag(var(s_0) + noise_floor^2)) via NLL;
    Gaussian channel capacity 0.5 * sum log(1 + var / noise_floor^2) estimates Weak A's empowerment in nats.
    """

    def __init__(self, dim_state: int = WEAK_STATE_DIM, dim_obj: int = BLOCK_DIM, dim_hidden: int = 64, noise_floor: float = 0.05):
        super().__init__()
        self.noise_floor_sq = noise_floor ** 2
        self.net = Sequential(
            Linear(dim_state + 2, dim_hidden),
            GELU(),
            Linear(dim_hidden, dim_hidden),
            GELU(),
            Linear(dim_hidden, dim_obj * 2),
        )

    def _stats(self, s_weak: Tensor) -> tuple[Tensor, Tensor]:
        feat = cat((s_weak, s_weak[..., 4:6] - s_weak[..., 0:2]), dim = -1)
        mu, raw_lv = self.net(feat).chunk(2, dim = -1)
        var = (raw_lv - 6.0).clamp(-12.0, 2.0).exp()
        return mu, var

    def loss(self, s_weak_0: Tensor, s_weak_fut: Tensor) -> Tensor:
        delta = extract_block_obs(s_weak_fut) - extract_block_obs(s_weak_0)
        mu, var = self._stats(s_weak_0)
        tot_var = var + self.noise_floor_sq
        return 0.5 * ((delta - mu).square() / tot_var + tot_var.log()).sum(dim = -1).mean()

    @torch.no_grad()
    def forward(self, s_weak: Tensor) -> Tensor:
        _, var = self._stats(s_weak)
        return 0.5 * torch.log1p(var / self.noise_floor_sq).sum(dim = -1)

def make_sac(
    dim_state: int,
    *,
    discount: float,
    reward_scale: float,
    lr: float,
    dim_hidden: int,
    mlp_depth: int,
    init_alpha: float,
    tau_ema: float,
) -> SAC:
    net_kw = dict(dim_state = dim_state, num_cont_actions = ACTION_DIM, mlp_depth = mlp_depth, dim_hidden = dim_hidden, simplicial_embed = False)

    sac = SAC(
        actor = Actor(**net_kw, use_beta = False, target_range = (-1.0, 1.0)),
        critics = [Critic(**net_kw, dim_out = 1) for _ in range(2)],
        reward_discount_rate = discount,
        reward_scale = reward_scale,
        actor_learning_rate = lr,
        critics_learning_rate = lr,
        temperature_learning_rate = 3e-4,
        actor_regen_reg_rate = 0.0,
        critics_regen_reg_rate = 0.0,
        critic_target_ema_decay = 1.0 - tau_ema,
        actor_update_freq = 1,
        critic_max_grad_norm = 10.0,
        target_continuous_entropy = -float(ACTION_DIM),
    )

    with torch.no_grad():
        sac.learned_entropy_temperature.log_alpha.fill_(math.log(init_alpha))

    return sac

@torch.no_grad()
def act_sac(sac: SAC, x: Tensor, *, deterministic: bool = False) -> Tensor:
    return sac.actor.deterministic_action(x) if deterministic else sac.actor(x, sample = True).continuous

def weak_skill_q(critics, s: Tensor, s0: Tensor, z: Tensor, a: Tensor) -> Tensor:
    return critics(skill_obs(s, s0, z), cont_actions = a)[0].mean(dim = -1)

def build_probe_skills(num_skills: int, dim_skill: int) -> Tensor:
    eye = torch.eye(dim_skill)
    vecs = [v for d in range(dim_skill) for v in (eye[d], -eye[d])]

    if len(vecs) < num_skills:
        gen = torch.Generator().manual_seed(12345)
        vecs.extend(l2norm(torch.randn(num_skills - len(vecs), dim_skill, generator = gen)).unbind(0))

    return stack(vecs[:num_skills], dim = 0)

# replay & HER helpers

def add_episodes(buf: ReplayBuffer, states: Tensor, actions: Tensor, *, next_weak: Tensor | None = None, **meta):
    with buf.batched_episode(batch_size = actions.shape[0], **meta):
        for t in range(actions.shape[1]):
            step_kw = dict(state = states[:, t], action = actions[:, t], next_state = states[:, t + 1])
            if exists(next_weak):
                step_kw['next_weak'] = next_weak[:, t]
            buf.store_batch(**step_kw)

def sample_recent_pairs(buf: ReplayBuffer, batch_size: int, horizon: int, max_episodes: int, discount: float, *, include_skill: bool):
    names = ('anchor', 'next_state', 'skill') if include_skill else ('anchor', 'next_state')
    batch = buf.sample(
        batch_size,
        horizon = horizon,
        max_age = max_episodes * horizon,
        fallback_on_empty = True,
        replacement = min(buf.num_episodes, max_episodes) < batch_size,
        fields = ('next_state',),
        to_named_tuple = names,
    )
    dt = sample_truncated_geometric(full((batch_size,), horizon, dtype = torch.long), discount)
    s_fut = batch.next_state[arange(batch_size), dt - 1]
    return (batch.anchor, s_fut, batch.skill) if include_skill else (batch.anchor, s_fut)

@torch.no_grad()
def sample_her_future_t(t_i: Tensor, horizon: int) -> Tensor:
    rem = horizon - t_i
    b = t_i.shape[0]
    rand_fut = t_i + (rand(b) * rem.float()).long().clamp(max = rem - 1)
    return einx.where('b, , b -> b', rand(b) < 0.5, horizon - 1, rand_fut)

@torch.no_grad()
def relabel_skills_her(buf: ReplayBuffer, classifier: SkillClassifier, batch, skill_len: int, her_prob: float) -> Tensor:
    if her_prob <= 0.0:
        return batch.skill

    ep_i, t_i = batch.indices[:, 0], batch.indices[:, 1]
    t_fut = sample_her_future_t(t_i, skill_len)
    s_fut = torch.from_numpy(buf.data['next_state'][ep_i.numpy(), t_fut.numpy()].copy())
    dh = classifier.encode_state(s_fut) - classifier.encode_state(batch.anchor)
    do_her = (rand(ep_i.shape[0]) < her_prob) & (dh.norm(dim = -1) > 0.25 * classifier.temperature)

    return einx.where('b, b d, b d -> b d', do_her, l2norm(dh), batch.skill) if do_her.any() else batch.skill

@torch.no_grad()
def relabel_partner_her(buf: ReplayBuffer, batch, pot_steps: int, her_prob: float) -> tuple[Tensor, Tensor, Tensor]:
    s_b, ns_b, nw_b = batch.state, batch.next_state, batch.next_weak
    if her_prob <= 0.0:
        return s_b, ns_b, nw_b

    ep_i, t_i = batch.indices[:, 0], batch.indices[:, 1]
    t_fut = sample_her_future_t(t_i, pot_steps)
    b_fut = torch.from_numpy(buf.data['next_state'][ep_i.numpy(), t_fut.numpy(), 4:6].copy())
    b_start = torch.from_numpy(buf.data['state'][ep_i.numpy(), 0, 4:6].copy())
    move = b_fut - b_start
    do_her = (rand(ep_i.shape[0]) < her_prob) & (move.norm(dim = -1) > 0.10)

    if not do_her.any():
        return s_b, ns_b, nw_b

    w_new = (b_fut + l2norm(move) * 0.16).clamp(-0.68, 0.68)
    s_b, ns_b, nw_b = s_b.clone(), ns_b.clone(), nw_b.clone()
    s_b[do_her, 8:10] = w_new[do_her]
    s_b[do_her, 12:14] = w_new[do_her] - s_b[do_her, 4:6]
    ns_b[do_her, 8:10] = w_new[do_her]
    ns_b[do_her, 12:14] = w_new[do_her] - ns_b[do_her, 4:6]
    nw_b[do_her, 0:2] = w_new[do_her]
    return s_b, ns_b, nw_b

# rollouts

@torch.no_grad()
def collect_two_agent_rollouts(
    env: TwoAgentPushTEnv,
    strong_sac: SAC,
    weak_sac: SAC,
    dim_skill: int,
    *,
    pot_steps: int,
    skill_len: int,
    use_strong_policy: bool,
    random_strong: bool = False,
    random_weak: bool = False,
    weak_corr_frac: float = 0.0,
    weak_near_frac: float = 0.50,
    strong_near_frac: float = 0.0,
):
    n = env.num_envs
    env.reset(weak_near_frac = 0.0 if use_strong_policy else weak_near_frac, strong_near_frac = strong_near_frac)

    # Phase 1: Strong Agent B acts

    pot_s = torch.empty(n, pot_steps + 1, STRONG_STATE_DIM)
    pot_a = torch.empty(n, pot_steps, ACTION_DIM)
    pot_wn = torch.empty(n, pot_steps, WEAK_STATE_DIM)
    pot_s[:, 0] = env.strong_state()

    rand_b = torch.empty(n, ACTION_DIM).uniform_(-1.0, 1.0)
    for t in range(pot_steps):
        if random_strong or not use_strong_policy:
            if divisible_by(t, 4):
                rand_b.uniform_(-1.0, 1.0)
            a_b = rand_b
        else:
            a_b = act_sac(strong_sac, pot_s[:, t])

        env.step(a_b, active = 'strong')
        pot_a[:, t] = a_b
        pot_s[:, t + 1] = env.strong_state()
        pot_wn[:, t] = env.weak_state()

    # Phase 2 starts: half fresh mixed resets (global empowerment coverage) + half Strong B handoffs

    env.reset_first_k(n // 2 if use_strong_policy else n, weak_near_frac = weak_near_frac)

    # Phase 2: Underpowered Agent A executes skill z

    z = sample_skills(n, dim_skill)
    skill_s = torch.empty(n, skill_len + 1, WEAK_STATE_DIM)
    skill_a = torch.empty(n, skill_len, ACTION_DIM)
    s0 = env.weak_state()
    skill_s[:, 0] = s0

    rand_a = torch.empty(n, ACTION_DIM).uniform_(-1.0, 1.0)
    use_corr = ones(n, dtype = torch.bool) if random_weak else (rand(n) < weak_corr_frac)

    for t in range(skill_len):
        if divisible_by(t, 8):
            rand_a.uniform_(-1.0, 1.0)

        a_a = rand_a if random_weak else act_sac(weak_sac, skill_obs(skill_s[:, t], s0, z))
        if not random_weak and use_corr.any():
            a_a = einx.where('b, b d, b d -> b d', use_corr, rand_a, a_a)

        env.step(a_a, active = 'weak')
        skill_a[:, t] = a_a
        skill_s[:, t + 1] = env.weak_state()

    return pot_s, pot_a, pot_wn, skill_s, skill_a, z

# evaluation

def describe_skill_motion(dx: float, dy: float, dth_deg: float) -> str:
    dirs = (
        (0, 'Right'), (45, 'Down-Right'), (90, 'Down'), (135, 'Down-Left'),
        (180, 'Left'), (-180, 'Left'), (-135, 'Up-Left'), (-90, 'Up'), (-45, 'Up-Right'),
    )
    ang = math.degrees(math.atan2(dy, dx))
    name = min(dirs, key = lambda kv: abs(ang - kv[0]))[1]
    dist = math.hypot(dx, dy)

    parts = []
    if dist >= 28.0:
        parts.append(f'Push {name}')
    elif dist >= 10.0:
        parts.append(f'Nudge {name}')

    if abs(dth_deg) >= 20.0:
        parts.append('Spin CW' if dth_deg > 0 else 'Spin CCW')

    return ' + '.join(parts) if parts else 'Hold / Unreached'

def cum_dtheta_deg(block_pose: Tensor, start: int) -> Tensor:
    diff = block_pose[:, start + 1:, 2] - block_pose[:, start:-1, 2]
    return torch.rad2deg(torch.atan2(diff.sin(), diff.cos()).sum(dim = -1))

def count_octants(dv: Tensor, mask: Tensor) -> int:
    if mask.sum() < 2:
        return int(mask.sum().item())
    angles = torch.remainder(torch.atan2(dv[mask, 1], dv[mask, 0]), 2.0 * math.pi)
    return int(torch.floor(angles / (math.pi / 4.0)).long().clamp(0, 7).unique().numel())

@torch.no_grad()
def eval_weak_empowerment(weak_states: Tensor, weak_sac: SAC, dim_skill: int, discount: float, num_mc: int = 32) -> Tensor:
    return estimate_empowerment(
        weak_states,
        partial(weak_skill_q, weak_sac.critics),
        dim_skill,
        policy = lambda st, st0, z: act_sac(weak_sac, skill_obs(st, st0, z), deterministic = True),
        skills = build_probe_skills(num_mc, dim_skill),
        discount = discount,
        clamp_min = 0.0,
    )

@torch.no_grad()
def rollout_eval(
    fp_mod,
    strong_sac: SAC,
    weak_sac: SAC,
    skills: Tensor,
    init_states: np.ndarray,
    *,
    pot_steps: int,
    skill_len: int,
    env_kwargs: dict,
) -> dict:
    k = skills.shape[0]
    total = pot_steps + skill_len
    env = TwoAgentPushTEnv(fp_mod, k, **env_kwargs)
    env.reset(reset_to_state = init_states)

    weak_states = torch.empty(k, total + 1, WEAK_STATE_DIM)
    wp = torch.empty(k, total + 1, 2)
    sp = torch.empty(k, total + 1, 2)
    bp = torch.empty(k, total + 1, 3)
    cog = torch.empty(k, total + 1, 2)

    s0 = env.weak_state()
    for t in range(total + 1):
        weak_states[:, t] = env.weak_state()
        wp[:, t], sp[:, t], bp[:, t], cog[:, t] = env.weak_pos, env.strong_pos, env.block_pose, env.block_cog()

        if t == total:
            break

        if t < pot_steps:
            env.step(act_sac(strong_sac, env.strong_state(), deterministic = True), active = 'strong')
            if t + 1 == pot_steps:
                s0 = env.weak_state()
        else:
            env.step(act_sac(weak_sac, skill_obs(env.weak_state(), s0, skills), deterministic = True), active = 'weak')

    return dict(weak_states = weak_states, weak_pos = wp, strong_pos = sp, block_pose = bp, block_cog = cog)

def make_far_eval_starts(num: int, seed: int = 999) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []

    while len(rows) < num:
        w = rng.uniform(90.0, 422.0, size = 2)
        b = rng.uniform(125.0, 387.0, size = 2)
        if not (230.0 <= float(np.linalg.norm(w - b)) <= 320.0):
            continue

        s = rng.uniform(75.0, 437.0, size = 2)
        if np.linalg.norm(s - b) < 85.0:
            continue

        rows.append([w[0], w[1], s[0], s[1], b[0], b[1], rng.uniform(-math.pi, math.pi)])

    return np.asarray(rows, dtype = np.float32)

@torch.no_grad()
def evaluate(
    fp_mod,
    classifier: SkillClassifier,
    strong_sac: SAC,
    weak_sac: SAC,
    *,
    dim_skill: int,
    pot_steps: int,
    skill_len: int,
    discount: float,
    env_kwargs: dict,
    emp_predictor: PartnerEmpowermentPredictor | None = None,
    num_eval_skills: int = 16,
) -> dict:
    with eval_mode(classifier, strong_sac, weak_sac, emp_predictor):
        far32 = make_far_eval_starts(32, seed = 999)
        z32 = build_probe_skills(32, dim_skill)
        tr_with = rollout_eval(fp_mod, strong_sac, weak_sac, z32, far32, pot_steps = pot_steps, skill_len = skill_len, env_kwargs = env_kwargs)
        tr_solo = rollout_eval(fp_mod, strong_sac, weak_sac, z32, far32, pot_steps = 0, skill_len = skill_len, env_kwargs = env_kwargs)

        d_init = (tr_with['block_cog'][:, 0] - tr_with['weak_pos'][:, 0]).norm(dim = -1)
        d_hand = (tr_with['block_cog'][:, pot_steps] - tr_with['weak_pos'][:, pot_steps]).norm(dim = -1)
        emp_init = eval_weak_empowerment(tr_with['weak_states'][:, 0], weak_sac, dim_skill, discount)
        emp_hand = eval_weak_empowerment(tr_with['weak_states'][:, pot_steps], weak_sac, dim_skill, discount)

        canon = np.tile(np.array(CANONICAL_START, np.float32), (num_eval_skills, 1))
        skills = build_probe_skills(num_eval_skills, dim_skill)
        tr_c_with = rollout_eval(fp_mod, strong_sac, weak_sac, skills, canon, pot_steps = pot_steps, skill_len = skill_len, env_kwargs = env_kwargs)
        tr_c_solo = rollout_eval(fp_mod, strong_sac, weak_sac, skills, canon, pot_steps = 0, skill_len = skill_len, env_kwargs = env_kwargs)

        dv_w = tr_c_with['block_cog'][:, -1] - tr_c_with['block_cog'][:, pot_steps]
        disp_w, dth_w = dv_w.norm(dim = -1), cum_dtheta_deg(tr_c_with['block_pose'], pot_steps)

        dv_s = tr_c_solo['block_cog'][:, -1] - tr_c_solo['block_cog'][:, 0]
        disp_s, dth_s = dv_s.norm(dim = -1), cum_dtheta_deg(tr_c_solo['block_pose'], 0)

        s_sw, ep_ar = tr_c_with['weak_states'][:, pot_steps], arange(num_eval_skills)
        accs, mis = [], []
        for _ in range(4):
            dt = sample_truncated_geometric(full((num_eval_skills,), skill_len, dtype = torch.long), discount)
            s_fut = tr_c_with['weak_states'][ep_ar, pot_steps + dt]
            logits = (classifier.encode_state(s_fut) - classifier.encode_state(s_sw)) @ skills.T
            accs.append((logits.argmax(dim = -1) == ep_ar).float().mean().item())
            mis.append(classifier.reward(s_sw, s_fut, skills, partition_weight = 1.0, increment_step = False).mean().item())

        b_probe = tensor([[320, 256], [192, 256], [256, 320], [256, 192], [440, 440], [72, 72], [440, 72], [72, 440]], dtype = torch.float32)
        emp_probe = eval_weak_empowerment(make_weak_probe_states(tensor([[256.0, 256.0]]), b_probe), weak_sac, dim_skill, discount)

        pred_corr = None
        if exists(emp_predictor):
            coords = torch.linspace(60.0, 452.0, 11)
            gy, gx = torch.meshgrid(coords, coords, indexing = 'ij')
            ws_grid = make_weak_probe_states(tensor([[256.0, 256.0]]), stack((gx, gy), dim = -1).reshape(-1, 2))
            pred_corr = pearson_r(eval_weak_empowerment(ws_grid, weak_sac, dim_skill, discount), emp_predictor(ws_grid))

    return dict(
        d_init = d_init.mean().item(),
        d_hand = d_hand.mean().item(),
        closer_pct = (d_hand < d_init - 20.0).float().mean().item() * 100.0,
        emp_init = emp_init.mean().item(),
        emp_hand = emp_hand.mean().item(),
        emp_near = emp_probe[:4].mean().item(),
        emp_far = emp_probe[4:].mean().item(),
        pred_corr = pred_corr,
        solo_touch = ((disp_s > 5.0) | (dth_s.abs() > 5.0)).float().mean().item(),
        solo_disp = disp_s.mean().item(),
        solo_oct = count_octants(dv_s, disp_s > 8.0),
        with_touch = ((disp_w > 5.0) | (dth_w.abs() > 5.0)).float().mean().item(),
        with_disp = disp_w.mean().item(),
        with_max_disp = disp_w.max().item(),
        with_dth = dth_w.abs().mean().item(),
        with_cw = int((dth_w < -15.0).sum().item()),
        with_ccw = int((dth_w > 15.0).sum().item()),
        with_oct = count_octants(dv_w, disp_w > 8.0),
        acc = float(np.mean(accs)),
        mi = float(np.mean(mis)),
        far32_solo_disp = (tr_solo['block_cog'][:, -1] - tr_solo['block_cog'][:, 0]).norm(dim = -1).mean().item(),
        far32_with_disp = (tr_with['block_cog'][:, -1] - tr_with['block_cog'][:, pot_steps]).norm(dim = -1).mean().item(),
    )

# rendering

def load_font(size: int):
    from PIL import ImageFont

    for path in ('/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/SFNS.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size = size)
            except OSError:
                continue

    return ImageFont.load_default()

def render_arena(fp_mod, weak_pos: Tensor, strong_pos: Tensor, block_pose: Tensor, sz: int, *, strong_active: bool, weak_radius: float, strong_radius: float):
    from PIL import Image, ImageDraw

    scale = 2
    xs, ys = fp_mod._coords(sz, sz, scale, block_pose.device)
    h, w = sz * scale, sz * scale
    bp = block_pose[None].float()
    img = full((1, h, w, 3), 255.0, dtype = torch.float32)

    wall_mask = zeros((h, w), dtype = torch.bool)
    for a, b in (((5.0, 506.0), (5.0, 5.0)), ((5.0, 5.0), (506.0, 5.0)), ((506.0, 5.0), (506.0, 506.0)), ((5.0, 506.0), (506.0, 506.0))):
        wall_mask |= fp_mod._segment_mask(xs, ys, a, b, 2.0)
    img = where(wall_mask[None, :, :, None], tensor(fp_mod.WALL_GRAY), img)

    for i in range(2):
        mask, lx, ly = fp_mod._rect_mask(xs, ys, bp, fp_mod.BLOCK_CENTERS[i], fp_mod.BLOCK_HALFS[i])
        hx, hy = fp_mod.BLOCK_HALFS[i]
        edge = mask & ((hx - lx.abs() <= 2.5) | (hy - ly.abs() <= 2.5))
        img = where(mask[..., None], tensor(fp_mod.BLOCK_FILL), img)
        img = where(edge[..., None], tensor(fp_mod.BLOCK_EDGE), img)

    raw = img.reshape(1, sz, scale, sz, scale, 3).mean(dim = (2, 4))[0].clamp(0.0, 255.0).to(torch.uint8).numpy()
    im = Image.fromarray(raw)
    dr = ImageDraw.Draw(im)
    sc = sz / 512.0

    sx, sy, sr = float(strong_pos[0]) * sc, float(strong_pos[1]) * sc, strong_radius * sc
    if sx > -50.0 and sy > -50.0:
        if strong_active:
            dr.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill = (245, 145, 35), outline = (175, 80, 10), width = 2)
            dr.ellipse([sx - sr * 0.55, sy - sr * 0.55, sx + sr * 0.55, sy + sr * 0.55], fill = (255, 190, 90))
        else:
            dr.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], outline = (230, 150, 65), width = 2)

    wx, wy, wr = float(weak_pos[0]) * sc, float(weak_pos[1]) * sc, weak_radius * sc
    dr.ellipse([wx - wr, wy - wr, wx + wr, wy + wr], fill = (60, 125, 250), outline = (25, 70, 175), width = 2)
    return im

def draw_trail(draw, pts: Tensor, start: int, end: int, scale: float, color: tuple[int, int, int], width: int):
    if end <= start:
        return
    xy = [(float(pts[j, 0]) * scale, float(pts[j, 1]) * scale) for j in range(start, end + 1)]
    if len(xy) >= 2:
        draw.line(xy, fill = color, width = width)

def paste_tile(canv, draw, tile, x0: int, y0: int, hdr: int, title: str, title_col, footer: str, border_col, font):
    from PIL import ImageDraw

    sz = tile.size[0]
    ImageDraw.Draw(tile).rectangle([0, 0, sz - 1, sz - 1], outline = border_col, width = 3)
    draw.text((x0, y0 + 4), title, fill = title_col, font = font)
    canv.paste(tile, (x0, y0 + hdr))
    draw.rectangle([x0 + 4, y0 + hdr + sz - 22, x0 + sz - 4, y0 + hdr + sz - 4], fill = (20, 22, 28))
    draw.text((x0 + 8, y0 + hdr + sz - 20), footer, fill = (225, 230, 240), font = font)

def render_assets(
    fp_mod,
    strong_sac: SAC,
    weak_sac: SAC,
    *,
    dim_skill: int,
    pot_steps: int,
    skill_len: int,
    discount: float,
    env_kwargs: dict,
    out_dir: Path,
    emp_predictor: PartnerEmpowermentPredictor | None = None,
):
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    weak_r = env_kwargs.get('weak_radius', 11.0)
    strong_r = env_kwargs.get('strong_radius', 22.0)
    font_title, font_sub, font_lbl = load_font(15), load_font(12), load_font(12)

    # 1. empowerment_heatmaps.png

    coords = torch.linspace(60.0, 452.0, 15)
    gy, gx = torch.meshgrid(coords, coords, indexing = 'ij')
    pts = stack((gx, gy), dim = -1).reshape(-1, 2)

    hm_rows = [('Oracle', lambda ws: eval_weak_empowerment(ws, weak_sac, dim_skill, discount))]
    if exists(emp_predictor):
        hm_rows.append(('Predicted', emp_predictor))

    s_hm, row_h = 256, 286
    hm_img = Image.new('RGB', (s_hm * 2 + 48, 56 + len(hm_rows) * row_h + 8), (18, 20, 26))
    hd = ImageDraw.Draw(hm_img)
    title_hm = "Oracle vs Strong B's Predicted Empowerment of Weak Agent A" if exists(emp_predictor) else "Weak Agent A's Learned Empowerment E_pot^A(T-block position)"
    hd.text((16, 10), title_hm, fill = (240, 244, 252), font = font_title)
    hd.text((16, 30), 'Blue dot = fixed Weak Agent A   |   White ring = peak empowerment (delivery target)', fill = (175, 185, 200), font = font_sub)

    for r_i, (row_tag, score_fn) in enumerate(hm_rows):
        y_base = 54 + r_i * row_h
        for c_i, (wxy, loc_label, col) in enumerate(HEATMAP_LOCATIONS):
            hm = score_fn(make_weak_probe_states(tensor([wxy]), pts)).reshape(15, 15).numpy()
            hn = np.asarray(Image.fromarray(((hm - hm.min()) / max(hm.max() - hm.min(), 1e-6) * 255).astype(np.uint8)).resize((s_hm, s_hm), Image.BILINEAR), dtype = np.float32) / 255.0
            heat = np.stack((np.clip(hn * 1.8, 0, 1) * 255, np.clip((hn - 0.35) * 1.7, 0, 1) * 255, np.clip(0.55 - np.abs(hn - 0.28) * 1.5, 0, 1) * 220), axis = -1)

            tile = Image.fromarray(np.clip(255 * 0.20 + heat * 0.80, 0, 255).astype(np.uint8))
            td = ImageDraw.Draw(tile)
            sc = s_hm / 512.0

            wx, wy = wxy[0] * sc, wxy[1] * sc
            td.ellipse([wx - 6, wy - 6, wx + 6, wy + 6], fill = (60, 125, 250), outline = (255, 255, 255), width = 2)

            pi, pj = np.unravel_index(hm.argmax(), hm.shape)
            px, py = float(coords[pj]) * sc, float(coords[pi]) * sc
            td.ellipse([px - 9, py - 9, px + 9, py + 9], outline = (255, 255, 255), width = 2)

            td.rectangle([2, s_hm - 22, s_hm - 2, s_hm - 2], fill = (18, 20, 26))
            td.text((8, s_hm - 19), f'peak = {hm.max():.2f} nats   min = {hm.min():.2f} nats', fill = (255, 220, 110), font = font_lbl)

            x0 = 16 + c_i * (s_hm + 16)
            hd.text((x0, y_base), f'{row_tag}  |  {loc_label}', fill = col, font = font_sub)
            hm_img.paste(tile, (x0, y_base + 18))

    hm_img.save(out_dir / 'empowerment_heatmaps.png')

    # shared 2x4 mp4 canvas geometry

    sz, cols, rows, top_h, hdr, pad = 224, 4, 2, 50, 24, 8
    sc = sz / 512.0
    gw = ((cols * sz + (cols + 1) * pad + 15) // 16) * 16
    gh = ((top_h + rows * (hdr + sz + pad) + pad + 15) // 16) * 16
    total_frames = pot_steps + skill_len + 1

    # 2. skills_grid.mp4

    gen = torch.Generator().manual_seed(7)
    z80 = cat((build_probe_skills(16, dim_skill), l2norm(torch.randn(64, dim_skill, generator = gen))), dim = 0)
    tr80 = rollout_eval(fp_mod, strong_sac, weak_sac, z80, np.tile(np.array(CANONICAL_START, np.float32), (80, 1)), pot_steps = pot_steps, skill_len = skill_len, env_kwargs = env_kwargs)

    dv80 = tr80['block_cog'][:, -1] - tr80['block_cog'][:, pot_steps]
    dth80 = cum_dtheta_deg(tr80['block_pose'], pot_steps)
    feats = stack((dv80[:, 0] / 80.0, dv80[:, 1] / 80.0, dth80 / 45.0), dim = -1)
    norms = feats.norm(dim = -1)

    sel = [int(norms.argmax().item())]
    while len(sel) < 8:
        score = torch.cdist(feats, feats[sel]).min(dim = -1).values + 0.15 * norms
        score[sel] = -1.0
        sel.append(int(score.argmax().item()))
    sel.sort(key = lambda i: math.atan2(dv80[i, 1].item(), dv80[i, 0].item()))

    wp, sp, bp, cog = tr80['weak_pos'][sel], tr80['strong_pos'][sel], tr80['block_pose'][sel], tr80['block_cog'][sel]
    dths = cum_dtheta_deg(bp, pot_steps).tolist()
    summaries = [
        (sel[k], dx := float((cog[k, -1, 0] - cog[k, pot_steps, 0]).item()), dy := float((cog[k, -1, 1] - cog[k, pot_steps, 1]).item()), dths[k], describe_skill_motion(dx, dy, dths[k]))
        for k in range(8)
    ]

    skill_frames = []
    for t in range(total_frames):
        canv = Image.new('RGB', (gw, gh), (18, 20, 26))
        dr = ImageDraw.Draw(canv)
        in_pot = t < pot_steps
        phase = f'Phase 1: Strong B delivers block to Weak A ({t}/{pot_steps})' if in_pot else f'Phase 2: Weak A executes skill z ({t - pot_steps}/{skill_len})'

        dr.text((12, 8), f'Two-Agent SEA PushT  |  {phase}', fill = (255, 195, 95) if in_pot else (110, 220, 255), font = font_title)
        dr.text((12, 28), 'Orange = Strong B (pi_pot)   |   Blue = Underpowered Weak A (pi_eff)   |   Red trail = T-block moved by Weak A', fill = (180, 188, 204), font = font_sub)

        for k, (oid, _, _, dth_d, label) in enumerate(summaries):
            r_i, c_i = divmod(k, cols)
            x0, y0 = pad + c_i * (sz + pad), top_h + r_i * (hdr + sz + pad)

            tile = render_arena(fp_mod, wp[k, t], sp[k, t], bp[k, t], sz, strong_active = in_pot, weak_radius = weak_r, strong_radius = strong_r)
            td = ImageDraw.Draw(tile)
            draw_trail(td, sp[k], 0, min(t, pot_steps), sc, (245, 145, 35), 3)
            draw_trail(td, cog[k], 0, min(t, pot_steps), sc, (230, 185, 90), 2)
            draw_trail(td, wp[k], pot_steps, t, sc, (50, 115, 245), 3)
            draw_trail(td, cog[k], pot_steps, t, sc, (225, 60, 60), 3)

            footer = (
                f'dist(block, Weak A) = {float((cog[k, t] - wp[k, t]).norm().item()):3.0f} px'
                if in_pot else
                f'Weak A disp = {float((cog[k, t] - cog[k, pot_steps]).norm().item()):3.0f} px   dth = {dth_d:+4.0f} deg'
            )
            paste_tile(canv, dr, tile, x0, y0, hdr, f'z#{oid:02d}: {label}', (232, 236, 246), footer, (255, 180, 65) if in_pot else (90, 200, 255), font_lbl)

        frame = np.asarray(canv)
        skill_frames.append(frame)
        if t == pot_steps:
            skill_frames.extend([frame] * 10)
    skill_frames.extend([skill_frames[-1]] * 18)

    imageio.mimsave(str(out_dir / 'skills_grid.mp4'), skill_frames, fps = 12, codec = 'libx264', quality = 8)

    # 3. empowerment_comparison.mp4

    z16 = build_probe_skills(16, dim_skill)
    sc_init = np.array([row for _, row in COMPARISON_SCENARIOS], dtype = np.float32)

    def best_probe_skill(row):
        tr_p = rollout_eval(fp_mod, strong_sac, weak_sac, z16, np.tile(row, (16, 1)), pot_steps = pot_steps, skill_len = skill_len, env_kwargs = env_kwargs)
        return z16[int((tr_p['block_cog'][:, -1] - tr_p['block_cog'][:, pot_steps]).norm(dim = -1).argmax().item())]

    z4 = stack([best_probe_skill(row) for row in sc_init], dim = 0)

    tr_solo = rollout_eval(fp_mod, strong_sac, weak_sac, z4, sc_init, pot_steps = 0, skill_len = skill_len, env_kwargs = env_kwargs)
    tr_with = rollout_eval(fp_mod, strong_sac, weak_sac, z4, sc_init, pot_steps = pot_steps, skill_len = skill_len, env_kwargs = env_kwargs)

    cmp_frames = []
    for t in range(total_frames):
        canv = Image.new('RGB', (gw, gh), (18, 20, 26))
        dr = ImageDraw.Draw(canv)
        in_pot = t < pot_steps
        phase = f'Phase 1: Strong B delivers ({t}/{pot_steps})' if in_pot else f'Phase 2: Weak A executes skill ({t - pot_steps}/{skill_len})'

        dr.text((12, 8), f'Top: Underpowered Weak A ALONE  |  Bottom: Strong B Empowers Weak A ({phase})', fill = (240, 244, 252), font = font_title)
        dr.text((12, 28), 'Identical start & skill z per column   |   Orange = Strong B (pi_pot)   |   Blue = Weak A (pi_eff)   |   Red = T-block moved by A', fill = (180, 188, 204), font = font_sub)

        for col, (name, _) in enumerate(COMPARISON_SCENARIOS):
            x0 = pad + col * (sz + pad)

            ts = min(t, skill_len)
            tile_s = render_arena(fp_mod, tr_solo['weak_pos'][col, ts], tensor([-200.0, -200.0]), tr_solo['block_pose'][col, ts], sz, strong_active = False, weak_radius = weak_r, strong_radius = strong_r)
            td_s = ImageDraw.Draw(tile_s)
            draw_trail(td_s, tr_solo['weak_pos'][col], 0, ts, sc, (50, 115, 245), 3)
            draw_trail(td_s, tr_solo['block_cog'][col], 0, ts, sc, (225, 60, 60), 3)
            disp_s = float((tr_solo['block_cog'][col, ts] - tr_solo['block_cog'][col, 0]).norm().item())
            paste_tile(canv, dr, tile_s, x0, top_h, hdr, f'Solo Weak A  |  {name}', (195, 200, 214), f'Weak A block disp = {disp_s:3.0f} px', (140, 145, 160), font_lbl)

            tile_w = render_arena(fp_mod, tr_with['weak_pos'][col, t], tr_with['strong_pos'][col, t], tr_with['block_pose'][col, t], sz, strong_active = in_pot, weak_radius = weak_r, strong_radius = strong_r)
            td_w = ImageDraw.Draw(tile_w)
            draw_trail(td_w, tr_with['strong_pos'][col], 0, min(t, pot_steps), sc, (245, 145, 35), 3)
            draw_trail(td_w, tr_with['block_cog'][col], 0, min(t, pot_steps), sc, (230, 185, 90), 2)
            draw_trail(td_w, tr_with['weak_pos'][col], pot_steps, t, sc, (50, 115, 245), 3)
            draw_trail(td_w, tr_with['block_cog'][col], pot_steps, t, sc, (225, 60, 60), 3)
            footer_w = (
                f'B delivering: dist = {float((tr_with["block_cog"][col, t] - tr_with["weak_pos"][col, t]).norm().item()):3.0f} px'
                if in_pot else
                f'Weak A block disp = {float((tr_with["block_cog"][col, t] - tr_with["block_cog"][col, pot_steps]).norm().item()):3.0f} px'
            )
            paste_tile(canv, dr, tile_w, x0, top_h + hdr + sz + pad, hdr, f'Strong B -> Weak A  |  {name}', (255, 195, 95) if in_pot else (110, 220, 255), footer_w, (255, 180, 65) if in_pot else (90, 200, 255), font_lbl)

        cframe = np.asarray(canv)
        cmp_frames.append(cframe)
        if t == pot_steps:
            cmp_frames.extend([cframe] * 10)
    cmp_frames.extend([cmp_frames[-1]] * 18)

    imageio.mimsave(str(out_dir / 'empowerment_comparison.mp4'), cmp_frames, fps = 12, codec = 'libx264', quality = 8)
    return summaries

# training

def train(
    x_jepa_dir: str = '../x-jepa',
    out_dir: str = 'runs/sea_two_agent_pusht',
    predict_empowerment: bool = False,
    rounds: int = 160,
    num_envs: int = 64,
    pot_steps: int = 25,
    skill_len: int = 40,
    weak_radius: float = 11.0,
    weak_step_scale: float = 18.0,
    weak_kp: float = 65.0,
    weak_kv: float = 20.0,
    weak_mass: float = 5.0,
    strong_radius: float = 22.0,
    strong_step_scale: float = 65.0,
    strong_kp: float = 130.0,
    strong_kv: float = 20.0,
    dim_skill: int = 4,
    temperature: float = 0.10,
    noise_std: float = 0.05,
    sigreg_weight: float = 0.03,
    discount: float = 0.97,
    pot_discount: float = 0.95,
    pot_reward_scale: float = 10.0,
    pred_noise_floor: float = 0.05,
    pred_updates: int = 16,
    pred_lr: float = 3e-3,
    batch_size: int = 256,
    encoder_updates: int = 10,
    weak_updates: int = 64,
    strong_updates: int = 48,
    pot_start_round: int = 18,
    warmup_rounds: int = 10,
    replay_episodes: int = 4096,
    encoder_buffer_episodes: int = 512,
    her_frac: float = 0.50,
    pot_her_frac: float = 0.50,
    emp_num_skills: int = 8,
    lr: float = 1e-3,
    clf_lr: float = 3e-4,
    dim_hidden: int = 64,
    mlp_depth: int = 2,
    init_alpha: float = 0.05,
    tau_ema: float = 0.02,
    eval_every: int = 20,
    seed: int = 42,
    num_threads: int = 4,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    fp_mod = load_faster_pusht(x_jepa_dir)
    env_kwargs = dict(
        weak_radius = weak_radius,
        weak_step_scale = weak_step_scale,
        weak_kp = weak_kp,
        weak_kv = weak_kv,
        weak_mass = weak_mass,
        strong_radius = strong_radius,
        strong_step_scale = strong_step_scale,
        strong_kp = strong_kp,
        strong_kv = strong_kv,
    )
    env = TwoAgentPushTEnv(fp_mod, num_envs, seed = seed, **env_kwargs)
    out_path = Path(out_dir)
    out_path.mkdir(parents = True, exist_ok = True)

    # 1. SkillClassifier over T-block pose (SEA_pytorch)

    classifier = SkillClassifier(
        ResidualEncoder(BLOCK_DIM, 64, dim_skill),
        total_steps = rounds,
        anneal_fracs = (0.30, 0.75),
        temperature = temperature,
        noise_std = noise_std,
        num_samples = 512,
        sigreg_weight = sigreg_weight,
        sigreg_scale = 2.0 * temperature,
        extract_obs = extract_block_obs,
    )
    opt_clf = torch.optim.Adam(classifier.parameters(), lr = clf_lr)

    # 2. Dual SAC agents + optional black-box PartnerEmpowermentPredictor

    weak_obs_dim = WEAK_STATE_DIM + 2 * BLOCK_DIM + 2 + dim_skill
    sac_kw = dict(lr = lr, dim_hidden = dim_hidden, mlp_depth = mlp_depth, init_alpha = init_alpha, tau_ema = tau_ema)
    weak_sac = make_sac(weak_obs_dim, discount = discount, reward_scale = 1.0, **sac_kw)
    strong_sac = make_sac(STRONG_STATE_DIM, discount = pot_discount, reward_scale = 1.0 if predict_empowerment else pot_reward_scale, **sac_kw)

    emp_predictor = PartnerEmpowermentPredictor(WEAK_STATE_DIM, BLOCK_DIM, dim_hidden = dim_hidden, noise_floor = pred_noise_floor) if predict_empowerment else None
    opt_pred = torch.optim.Adam(emp_predictor.parameters(), lr = pred_lr) if exists(emp_predictor) else None

    mode_str = "predicted E_hat(s') from watching Weak A rollouts (black-box)" if predict_empowerment else "oracle E_pot^A(s') via estimate_empowerment on Weak A critic"
    print(
        '\nSEA Two-Agent Asymmetric PushT (arXiv:2610.13002)\n'
        f'  Weak Agent A   (pi_eff) : r={weak_radius:.0f}px, step={weak_step_scale:.0f}px, mass={weak_mass:.1f}  (underpowered skill agent)\n'
        f'  Strong Agent B (pi_pot) : r={strong_radius:.0f}px, step={strong_step_scale:.0f}px, mass=inf  (empowering agent)\n'
        f'  Empowerment signal      : {mode_str}\n'
        f'  Schedule                : {pot_steps} steps (Strong B) -> {skill_len} steps (Weak A) | {rounds} rounds x {num_envs} envs\n'
    )
    corr_hdr = ' |  corr' if predict_empowerment else ''
    print(f'{"round":>5} | {"steps":>7} | {"beta":>4} | {"InfoNCE":>7} | {"E_A near/far":>12} | {"B->A dist (32 far)":>18} | {"E_A before->after":>17} | {"Weak A disp (solo->+B)":>22} | {"oct":>3} | {"spin":>4} | {"MI":>5}{corr_hdr}')
    print('-' * (132 + len(corr_hdr)))

    t0, env_steps, zeros_done, history, last_ev = time.perf_counter(), 0, zeros(batch_size, dtype = torch.bool), [], None

    with tempfile.TemporaryDirectory(prefix = 'sea_two_agent_') as tmp_dir:
        tmp_path = Path(tmp_dir)
        skill_buf = ReplayBuffer(
            tmp_path / 'replay_skill',
            max_episodes = replay_episodes,
            max_timesteps = skill_len,
            fields = dict(state = ('float', WEAK_STATE_DIM), action = ('float', ACTION_DIM), next_state = ('float', WEAK_STATE_DIM)),
            meta_fields = dict(anchor = ('float', WEAK_STATE_DIM), skill = ('float', dim_skill)),
            circular = True, overwrite = True, track_write_step = True, flush_every_store_step = skill_len,
        )
        pot_buf = ReplayBuffer(
            tmp_path / 'replay_pot',
            max_episodes = replay_episodes,
            max_timesteps = pot_steps,
            fields = dict(state = ('float', STRONG_STATE_DIM), action = ('float', ACTION_DIM), next_state = ('float', STRONG_STATE_DIM), next_weak = ('float', WEAK_STATE_DIM)),
            circular = True, overwrite = True, flush_every_store_step = pot_steps,
        )

        # 3. Correlated random warmup

        for _ in range(warmup_rounds):
            ps, pa, pwn, ss, sa, z = collect_two_agent_rollouts(
                env, strong_sac, weak_sac, dim_skill,
                pot_steps = pot_steps, skill_len = skill_len,
                use_strong_policy = False, random_strong = True, random_weak = True,
                weak_near_frac = 0.60, strong_near_frac = 0.35,
            )
            add_episodes(pot_buf, ps, pa, next_weak = pwn)
            add_episodes(skill_buf, ss, sa, anchor = ss[:, 0], skill = z)

        # 4. Main two-agent SEA loop

        for rnd in range(1, rounds + 1):
            weak_corr = max(0.0, 0.35 * (1.0 - (rnd - 1) / max(1, pot_start_round)))
            ps, pa, pwn, ss, sa, z = collect_two_agent_rollouts(
                env, strong_sac, weak_sac, dim_skill,
                pot_steps = pot_steps, skill_len = skill_len,
                use_strong_policy = rnd > pot_start_round,
                random_strong = rnd <= pot_start_round,
                weak_corr_frac = weak_corr,
            )
            add_episodes(pot_buf, ps, pa, next_weak = pwn)
            add_episodes(skill_buf, ss, sa, anchor = ss[:, 0], skill = z)
            env_steps += num_envs * (pot_steps + skill_len)
            classifier.set_step_(rnd)

            # (a) SkillClassifier on Weak A's T-block trajectories

            nce_val = 0.0
            for _ in range(encoder_updates):
                anchor, s_fut, z_c = sample_recent_pairs(skill_buf, batch_size, skill_len, encoder_buffer_episodes, discount, include_skill = True)
                nce_val += step_opt(opt_clf, classifier(anchor, s_fut, z_c), classifier.parameters())
            nce_val /= encoder_updates

            # (b) Weak Agent A skill policy pi_eff & critic Q_eff

            her_now = her_frac * (1.0 - (rnd - 1) / max(1, rounds - 1))
            for _ in range(weak_updates):
                b = skill_buf.sample(batch_size, timestep_level = True, fields = ('state', 'action', 'next_state'), return_indices = True, to_named_tuple = True)
                z_b = relabel_skills_her(skill_buf, classifier, b, skill_len, her_now)
                r_eff = classifier.reward(b.anchor, b.next_state, z_b, increment_step = False)
                weak_sac(
                    states = skill_obs(b.state, b.anchor, z_b),
                    cont_actions = b.action,
                    discrete_actions = None,
                    rewards = r_eff,
                    done = zeros_done,
                    next_states = skill_obs(b.next_state, b.anchor, z_b),
                )

            # (c) Optional black-box PartnerEmpowermentPredictor watching only Weak A's recent (s_0, s_fut)

            if exists(emp_predictor):
                for _ in range(pred_updates):
                    anchor_p, s_fut_p = sample_recent_pairs(skill_buf, batch_size, skill_len, encoder_buffer_episodes, discount, include_skill = False)
                    step_opt(opt_pred, emp_predictor.loss(anchor_p, s_fut_p), emp_predictor.parameters())

            # (d) Strong Agent B potential policy pi_pot maximizing Weak A's empowerment (oracle or predicted)

            if rnd >= pot_start_round:
                pot_her_now = pot_her_frac * (1.0 - 0.8 * (rnd - pot_start_round) / max(1, rounds - pot_start_round))
                for _ in range(strong_updates):
                    pb = pot_buf.sample(batch_size, timestep_level = True, fields = ('state', 'action', 'next_state', 'next_weak'), return_indices = True, to_named_tuple = True)
                    s_b, ns_b, nw_b = relabel_partner_her(pot_buf, pb, pot_steps, pot_her_now)

                    with torch.no_grad():
                        if exists(emp_predictor):
                            with eval_mode(emp_predictor):
                                r_pot = emp_predictor(nw_b)
                        else:
                            weak_sac.critics_target.eval()
                            r_pot = estimate_empowerment(
                                nw_b,
                                partial(weak_skill_q, weak_sac.critics_target),
                                dim_skill,
                                num_skills = emp_num_skills,
                                policy = lambda st, st0, z_: act_sac(weak_sac, skill_obs(st, st0, z_), deterministic = True),
                                discount = discount,
                                clamp_min = 0.0,
                            )

                    strong_sac(
                        states = s_b,
                        cont_actions = pb.action,
                        discrete_actions = None,
                        rewards = r_pot,
                        done = zeros_done,
                        next_states = ns_b,
                    )

            if rnd == 1 or divisible_by(rnd, eval_every) or rnd == rounds:
                ev = evaluate(
                    fp_mod, classifier, strong_sac, weak_sac,
                    dim_skill = dim_skill, pot_steps = pot_steps, skill_len = skill_len,
                    discount = discount, env_kwargs = env_kwargs, emp_predictor = emp_predictor,
                )
                last_ev = ev
                corr_col = f' | {ev["pred_corr"]:5.2f}' if exists(emp_predictor) else ''
                print(
                    f'{rnd:5d} | {env_steps:7d} | {classifier.partition_weight:4.2f} | {nce_val:7.2f} | '
                    f'{ev["emp_near"]:5.2f} / {ev["emp_far"]:4.2f} | '
                    f'{ev["d_init"]:5.0f} -> {ev["d_hand"]:4.0f} px   | '
                    f'  {ev["emp_init"]:5.2f} -> {ev["emp_hand"]:4.2f}   | '
                    f'   {ev["solo_disp"]:5.1f} -> {ev["with_disp"]:5.1f} px   | '
                    f'{ev["with_oct"]:1d}/8 | {ev["with_dth"]:3.0f}d | {ev["mi"]:5.2f}{corr_col}',
                    flush = True,
                )
                history.append(dict(round = rnd, env_steps = env_steps, nce = nce_val, **ev))

    # 5. Render assets & save checkpoint/summary

    with eval_mode(strong_sac, weak_sac, emp_predictor):
        summaries = render_assets(
            fp_mod, strong_sac, weak_sac,
            dim_skill = dim_skill, pot_steps = pot_steps, skill_len = skill_len,
            discount = discount, env_kwargs = env_kwargs, out_dir = out_path, emp_predictor = emp_predictor,
        )

    ckpt = dict(
        predict_empowerment = predict_empowerment,
        env_kwargs = env_kwargs,
        dim_skill = dim_skill,
        classifier = classifier.state_dict(),
        weak_sac = weak_sac.state_dict(),
        strong_sac = strong_sac.state_dict(),
        history = history,
    )
    if exists(emp_predictor):
        ckpt['emp_predictor'] = emp_predictor.state_dict()
    torch.save(ckpt, out_path / 'checkpoint.pt')

    summary_payload = dict(
        predict_empowerment = predict_empowerment,
        env_kwargs = env_kwargs,
        final_eval = last_ev,
        selected_skills = [dict(id = s[0], dx = round(s[1], 1), dy = round(s[2], 1), dth = round(s[3], 1), label = s[4]) for s in summaries],
    )
    (out_path / 'summary.json').write_text(json.dumps(summary_payload, indent = 2))

    gain_str = f' ({last_ev["far32_with_disp"] / last_ev["far32_solo_disp"]:.1f}x)' if last_ev['far32_solo_disp'] >= 0.5 else ''
    corr_line = f'  Predictor vs Oracle correlation   : r = {last_ev["pred_corr"]:.3f} (grid Pearson r)\n' if exists(emp_predictor) else ''
    print(
        f'\nFinished in {time.perf_counter() - t0:.1f}s ({env_steps:,} env steps)\n\n'
        f'Summary ({mode_str}):\n'
        f'  Strong B delivery (32 far starts) : dist(block, Weak A) {last_ev["d_init"]:.1f} -> {last_ev["d_hand"]:.1f} px ({last_ev["closer_pct"]:.0f}% closer)\n'
        f'  Weak A empowerment gain E_pot^A   : {last_ev["emp_init"]:.2f} -> {last_ev["emp_hand"]:.2f} nats (near/far probe: {last_ev["emp_near"]:.2f} / {last_ev["emp_far"]:.2f})\n'
        f'{corr_line}'
        f'  Weak A skill disp (32 far starts) : {last_ev["far32_solo_disp"]:.1f} px solo -> {last_ev["far32_with_disp"]:.1f} px empowered{gain_str}\n'
        f'  Weak A skills (canonical 250 px)  : {last_ev["solo_disp"]:.1f} px solo -> {last_ev["with_disp"]:.1f} px empowered '
        f'(max {last_ev["with_max_disp"]:.1f} px, {last_ev["with_oct"]}/8 octants, {last_ev["with_cw"]} CW / {last_ev["with_ccw"]} CCW, MI = {last_ev["mi"]:.2f} nats)\n\n'
        f'Saved assets in {out_path}:\n'
        '  empowerment_comparison.mp4  — 4 far partner locations: Solo Weak A (top) vs Strong B -> Weak A (bottom), same skill z\n'
        '  skills_grid.mp4             — Strong B delivers block, then Weak A executes 8 distinct push/spin skills\n'
        f'  empowerment_heatmaps.png    — Weak A learned empowerment across two locations{" (Oracle vs Predicted)" if predict_empowerment else ""}\n'
        '  summary.json                — full quantitative metrics\n'
        '  checkpoint.pt               — trained weights\n'
    )


if __name__ == '__main__':
    fire.Fire(train)
