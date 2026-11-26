from __future__ import annotations

# Mode-Conditioned Unified SAC + SEA Empowerment + JEPA+SIGReg World Model + CEM + BC
# Supports 2D Faster PushT (default) and 3D ManiSkill PushT-v1 (`--pusht3d` or `--env pusht3d`):
#   - single unified SAC-pytorch agent conditioned on mode in {EXPLORE [1, 0], TASK [0, 1]}
#       * EXPLORE: SEA empowerment skills z ~ S^{k-1}, intrinsic skill reward r_eff, done = False
#       * TASK:    pure binary sparse reward (2D: cov > 0.95, 3D: ManiSkill exact cov >= 0.90)
#   - cycled with JEPA+SIGReg world model (StateCompressor) + Policy+Q-Guided CEM (CEMPlanner)
#   - occasional Q-filtered behavior cloning from winning CEM planning rollouts into pi(a | s, TASK)
#
# Train on 3D PushT (ManiSkill PushT-v1):
#   uv run --with "mani-skill==3.0.1" --with "sapien==3.0.3" --with pin --with "sac-pytorch>=0.3.7" --with "memmap-replay-buffer>=0.3.0" --with fire --with pillow --with matplotlib --with "imageio[ffmpeg]" \
#       python train_sea_sparse_sac.py --pusht3d
#
# Compare all arms on 3D PushT (Task-Only SAC vs SEA+WM+CEM vs No-SEA vs Unified SEA):
#   uv run --with "mani-skill==3.0.1" --with "sapien==3.0.3" --with pin --with "sac-pytorch>=0.3.7" --with "memmap-replay-buffer>=0.3.0" --with fire --with pillow --with matplotlib --with "imageio[ffmpeg]" \
#       python train_sea_sparse_sac.py compare --pusht3d
#
# Train / compare / pilot on 2D Faster PushT:
#   uv run --with "sac-pytorch>=0.3.7" --with "memmap-replay-buffer>=0.3.0" --with fire --with pillow --with matplotlib --with "imageio[ffmpeg]" \
#       python train_sea_sparse_sac.py [train|compare|pilot]

from functools import partial
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

from einops import rearrange
import einx
from fire import Fire
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import arange, cat, nn, rand, randint, Tensor
from torch.nn import GELU, LayerNorm, Linear, Module, Sequential
import torch.nn.functional as F
from torch_einops_utils import trainable_parameters

from memmap_replay_buffer import ReplayBuffer
from SAC_pytorch import SAC, Actor, Critic
from SEA_pytorch import (
    CEMPlanner,
    SkillClassifier,
    StateCompressor,
    estimate_empowerment,
    sample_skills,
    sample_truncated_geometric,
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def resolve_env(env: str, pusht3d: bool) -> str:
    return 'pusht3d' if pusht3d else env

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clone_sd(module: Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

def step_opt(opt, loss: Tensor, params = None, max_norm: float | None = None):
    opt.zero_grad()
    loss.backward()
    if exists(params) and exists(max_norm):
        nn.utils.clip_grad_norm_(params, max_norm)
    opt.step()
    opt.zero_grad()

def ridge_r2(x: Tensor, y: Tensor) -> float:
    x, y = x.detach().cpu().double(), y.detach().cpu().double()
    xa = cat((x, torch.ones(x.shape[0], 1, dtype = torch.float64)), dim = -1)
    reg = 1e-3 * torch.eye(xa.shape[1], dtype = torch.float64)
    reg[-1, -1] = 0.
    w = torch.linalg.solve(xa.T @ xa + reg, xa.T @ y)
    return float(1. - (y - xa @ w).square().sum().item() / ((y - y.mean(0, keepdim = True)).square().sum().item() + 1e-12))

def act_sac(sac: SAC, x: Tensor, *, deterministic: bool = False) -> Tensor:
    return sac.actor.deterministic_action(x) if deterministic else sac.actor(x, sample = True).continuous

def add_episodes(buf: ReplayBuffer, states: Tensor, actions: Tensor, **meta):
    with buf.batched_episode(batch_size = actions.shape[0], **meta):
        for t in range(actions.shape[1]):
            buf.store_batch(state = states[:, t], action = actions[:, t], next_state = states[:, t + 1])

def pick_indices(max_cov: Tensor, count: int = 6, max_quantile: float = 1.) -> list[int]:
    order = np.argsort(-max_cov.detach().cpu().numpy())
    n = len(order)
    hi = max(0, int(round(max_quantile * (n - 1))))
    return [int(order[i]) for i in np.unique(np.linspace(0, hi, min(count, n)).round().astype(int))]

# neural blocks

class SpectralClip(Module):
    def __init__(self, weight: Tensor, max_norm: float = 0.8, n_iters: int = 2):
        super().__init__()
        self.max_norm, self.n_iters = float(max_norm), n_iters
        w = weight.reshape(weight.shape[0], -1)
        self.register_buffer('_u', F.normalize(torch.randn(w.shape[0], device = weight.device, dtype = weight.dtype), dim = 0))
        self.register_buffer('_v', F.normalize(torch.randn(w.shape[1], device = weight.device, dtype = weight.dtype), dim = 0))

    def forward(self, weight: Tensor) -> Tensor:
        w = weight.reshape(weight.shape[0], -1)
        if self.training:
            with torch.no_grad():
                u, v = self._u, self._v
                for _ in range(self.n_iters):
                    v = F.normalize(w.T @ u, dim = 0, eps = 1e-12)
                    u = F.normalize(w @ v, dim = 0, eps = 1e-12)
                self._u.copy_(u); self._v.copy_(v)
        sigma = (self._u.detach().clone() @ (w @ self._v.detach().clone())).abs()
        return weight / sigma.clamp(min = self.max_norm) * self.max_norm

def clip_sn(mod: Module, max_norm: float = 0.8) -> Module:
    nn.utils.parametrize.register_parametrization(mod, 'weight', SpectralClip(mod.weight, max_norm = max_norm))
    return mod

class ReZeroMLP(Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, *, depth: int = 2, max_spectral_norm: float | None = None, layernorm: bool = False, skip_bias: bool = False, orthogonal_skip_gain: float | None = None, init_gate: float = 0., learnable_gate: bool = True, zero_init_mlp: bool = False):
        super().__init__()
        make = lambda di, do, bias = True: clip_sn(Linear(di, do, bias = bias), max_spectral_norm) if exists(max_spectral_norm) else Linear(di, do, bias = bias)
        self.skip = make(dim_in, dim_out, bias = skip_bias)
        if exists(orthogonal_skip_gain) and not exists(max_spectral_norm):
            nn.init.orthogonal_(self.skip.weight, gain = orthogonal_skip_gain)

        layers, d = [], dim_in
        for _ in range(depth):
            layers.append(make(d, dim_hidden))
            if layernorm:
                layers.append(LayerNorm(dim_hidden))
            layers.append(GELU())
            d = dim_hidden
        out = make(d, dim_out)
        if zero_init_mlp and not exists(max_spectral_norm):
            nn.init.zeros_(out.weight); nn.init.zeros_(out.bias)
        layers.append(out)
        self.mlp = Sequential(*layers)

        self.learnable_gate = learnable_gate
        if learnable_gate:
            self.alpha = nn.Parameter(torch.tensor(float(init_gate)))
        else:
            self.register_buffer('alpha', torch.tensor(float(init_gate)), persistent = False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.alpha.clamp(-1., 1.) if self.learnable_gate else self.alpha
        return self.skip(x) + gate * self.mlp(x)

class AffineProject(Module):
    def __init__(self, mean: Tensor, proj: Tensor):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('proj', proj.clone())

    def forward(self, h: Tensor) -> Tensor:
        return F.linear(h - self.mean, self.proj)

class GeneralStateEncoder(Module):
    def __init__(self, mean: Tensor, std: Tensor, dim_in: int, dim_latent: int = 8, dim_hidden: int = 128):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clamp(min = 1e-4).clone())
        self.head = ReZeroMLP(dim_in, dim_hidden, dim_latent, depth = 2, max_spectral_norm = 0.8)

    def forward(self, x: Tensor) -> Tensor:
        return self.head((x - self.mean) / self.std)

def make_sac(dim_state: int, dim_action: int, *, discount: float, lr: float, dim_hidden: int, mlp_depth: int = 2, use_beta: bool = False, init_alpha: float = 0.05, tau_ema: float = 0.02, actor_update_freq: int = 2) -> SAC:
    net_kw = dict(dim_state = dim_state, num_cont_actions = dim_action, mlp_depth = mlp_depth, dim_hidden = dim_hidden, simplicial_embed = False)
    sac = SAC(
        actor = Actor(**net_kw, use_beta = use_beta, target_range = (-1., 1.)),
        critics = [Critic(**net_kw, dim_out = 1) for _ in range(2)],
        reward_discount_rate = discount,
        reward_scale = 1.,
        actor_learning_rate = lr,
        critics_learning_rate = lr,
        temperature_learning_rate = 3e-4,
        actor_regen_reg_rate = 0.,
        critics_regen_reg_rate = 0.,
        critic_target_ema_decay = 1. - tau_ema,
        actor_update_freq = actor_update_freq,
        critic_max_grad_norm = 10.,
        target_continuous_entropy = -float(dim_action),
    )
    with torch.no_grad():
        sac.learned_entropy_temperature.log_alpha.fill_(math.log(init_alpha))
    return sac

def build_jepa_sigreg(state_dim: int, action_dim: int, *, dim_latent: int = 16, dim_hidden: int = 128, pred_hidden: int = 256, reg_weight: float = 0.10, rollout_discount: float = 0.95) -> StateCompressor:
    enc = ReZeroMLP(state_dim, dim_hidden, dim_latent, depth = 1, layernorm = True, skip_bias = True, init_gate = 0.5, learnable_gate = False, zero_init_mlp = True)
    return StateCompressor(
        enc, dim_latent = dim_latent, dim_action = action_dim, predictor_hidden_dim = pred_hidden, predictor_depth = 2,
        predictor_layernorm = True, obs_ndim = 1, regularizer = 'sigreg', dynamics_loss_weight = 1.,
        rollout_discount = rollout_discount, reg_weight = reg_weight, l2norm_latent = False,
    )

def build_probe(dim_latent: int, state_dim: int) -> Sequential:
    return Sequential(Linear(dim_latent, 128), GELU(), Linear(128, 128), GELU(), Linear(128, state_dim))

# 2D Faster PushT backend (PushT3DBackend lives in pusht3d_env.py)

class PushT2DBackend:
    name = 'pusht2d'
    state_dim = 8
    action_dim = 2
    win_cov = 0.95
    win_label = '>0.95'
    dist_unit = 'px'
    blk_xy_slice = slice(2, 4)
    blk_rot_slice = slice(4, 6)
    dist_scale = 256.
    contact_step_thresh = 0.5
    touch_disp_thresh = 5.0
    wm_dim_latent = 16
    wm_dim_hidden = 128
    default_max_sw = 15
    default_skill_len = 45
    default_dim_hidden = 128
    has_trajs = True

    def __init__(self, x_jepa_dir: str = '../x-jepa', step_scale: float = 40.):
        import train_sea_pusht as sp2
        import train_sea_world_model as wm2

        self.sp2, self.wm2 = sp2, wm2
        self.fp = sp2.load_faster_pusht(x_jepa_dir)
        self.step_scale = float(step_scale)
        self.pool_size = 64

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        pass

    def is_won(self, cov: Tensor) -> Tensor:
        return cov > self.win_cov

    def sample_starts(self, num_tasks: int, seed: int, *, start_mode: str = 'docking') -> np.ndarray | None:
        return None if start_mode == 'native' else self.wm2.build_docking_tasks(num_tasks = num_tasks, seed = seed)

    def build_eval_tasks(self, num_tasks: int, seed: int, *, start_mode: str = 'docking') -> np.ndarray:
        if start_mode == 'native':
            w = self.sp2.RawPushTWrapper(self.fp, num_tasks, step_scale = self.step_scale, seed = seed)
            w.reset(seed = seed)
            return cat((w.env.agent_pos, w.env.block_pose), dim = -1).cpu().numpy().astype(np.float32)
        return self.wm2.build_docking_tasks(num_tasks = num_tasks, seed = seed)

    def build_heldout_push_tasks(self, *, num_tasks: int = 64, rollout_len: int = 16, seed: int = 999) -> dict:
        return self.wm2.build_heldout_push_tasks(self.fp, num_tasks = num_tasks, rollout_len = rollout_len, step_scale = self.step_scale, seed = seed)

    def docking_probe_score(self, probe: Module, z_seq: Tensor) -> Tensor:
        s = probe(z_seq)
        goal_norm = s.new_tensor([self.wm2.GOAL_COM[0] / 256. - 1., self.wm2.GOAL_COM[1] / 256. - 1.])
        dist = (s[..., 2:4] - goal_norm).norm(dim = -1) * 256.
        th = torch.atan2(s[..., 5], s[..., 4])
        yaw = torch.atan2(torch.sin(th - self.wm2.GOAL_TH), torch.cos(th - self.wm2.GOAL_TH)).abs()
        step_score = -dist - 60. * yaw
        return 0.7 * step_score[..., -1] + 0.3 * einx.mean('... [h]', step_score)

    def goal_dist(self, raw_obs: Tensor) -> Tensor:
        goal_norm = raw_obs.new_tensor([self.wm2.GOAL_COM[0] / 256. - 1., self.wm2.GOAL_COM[1] / 256. - 1.])
        return (raw_obs[..., 2:4] - goal_norm).norm(dim = -1) * 256.

    def fit_state_compressor(self, comp_dev: torch.device, *, skill_len: int = 45, dim_latent: int = 8, horizon: int = 4, warmup_rounds: int = 16, steps: int = 1000, seed: int = 42):
        w_warm = self.sp2.RawPushTWrapper(self.fp, 16, step_scale = self.step_scale, seed = seed)
        comp, dim_eff, warm_obs, warm_act, _, _, stats = self.sp2.fit_state_compressor(
            w_warm, comp_dev, input_mode = 'state', img_size = 48, skill_len = skill_len, dim_latent = dim_latent, horizon = horizon,
            warmup_rounds = warmup_rounds, steps = steps, epx_eta = 20., reg_weight = 0.10, slow_weight = 20., energy_weight = 0.20, rank_rel_thresh = 0.25, target_std = 0.40,
        )
        return comp, dim_eff, warm_obs, warm_act, stats

    @torch.no_grad()
    def collect_random_steps(self, wm_buf: 'WorldModelBuffer', target_steps: int, *, ep_len: int = 60, hold: int = 4, seed: int = 0) -> int:
        need = target_steps - wm_buf.total_env_steps
        if need <= 0:
            return 0
        n_eps = max(1, math.ceil(need / ep_len))
        w = self.sp2.RawPushTWrapper(self.fp, n_eps, step_scale = self.step_scale, seed = seed)
        w.reset(seed = seed)
        gen = torch.Generator().manual_seed(seed + 17)
        obs, act = torch.empty(n_eps, ep_len + 1, self.state_dim), torch.empty(n_eps, ep_len, self.action_dim)
        ra = torch.empty(n_eps, self.action_dim).uniform_(-1., 1., generator = gen)
        obs[:, 0] = w.raw_state()
        for t in range(ep_len):
            if t % hold == 0:
                ra.uniform_(-1., 1., generator = gen)
            a = ra.clone()
            w.step(a)
            act[:, t], obs[:, t + 1] = a, w.raw_state()
        wm_buf.add_batch(obs, act)
        return int(n_eps * ep_len)

    @torch.no_grad()
    def rollout_explore_batch(self, n: int, total_steps: int, action_fn, *, seed: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w = self.sp2.RawPushTWrapper(self.fp, n, step_scale = self.step_scale, seed = seed)
        w.reset(seed = seed)
        raw_seq, act_seq = torch.empty(n, total_steps + 1, self.state_dim), torch.empty(n, total_steps, self.action_dim)
        pot_dist_seq, blk_pos_seq = torch.empty(n, total_steps + 1), torch.empty(n, total_steps + 1, 2)
        raw = w.raw_state()
        raw_seq[:, 0], pot_dist_seq[:, 0], blk_pos_seq[:, 0] = raw, (w.env.agent_pos - w.env._cog()).norm(dim = -1), w.env._cog()
        for t in range(total_steps):
            a = action_fn(t, raw)
            w.step(a)
            raw = w.raw_state()
            act_seq[:, t], raw_seq[:, t + 1] = a, raw
            pot_dist_seq[:, t + 1], blk_pos_seq[:, t + 1] = (w.env.agent_pos - w.env._cog()).norm(dim = -1), w.env._cog()
        return raw_seq, act_seq, pot_dist_seq, blk_pos_seq

    def reset_chunk(self, init_states: np.ndarray | None, n: int, seed: int):
        w = self.sp2.RawPushTWrapper(self.fp, n, step_scale = self.step_scale, seed = seed)
        w.reset(seed = seed, reset_to_state = init_states)
        return w.raw_state(), self.fp.coverage(w.env.block_pose), w

    def step_chunk(self, w, actions: Tensor, alive: Tensor):
        w.step(actions)
        return w.raw_state(), self.fp.coverage(w.env.block_pose)

    def init_trajs(self, w, n: int, max_steps: int) -> dict:
        trajs = dict(agent_pos = torch.zeros(n, max_steps + 1, 2), block_pose = torch.zeros(n, max_steps + 1, 3), block_cog = torch.zeros(n, max_steps + 1, 2))
        trajs['agent_pos'][:, 0], trajs['block_pose'][:, 0], trajs['block_cog'][:, 0] = w.env.agent_pos, w.env.block_pose, w.env._cog()
        return trajs

    def record_traj(self, w, trajs: dict, step: int, alive: Tensor):
        trajs['agent_pos'][:, step + 1] = einx.where('b, b d, b d -> b d', alive, w.env.agent_pos, trajs['agent_pos'][:, step])
        trajs['block_pose'][:, step + 1] = einx.where('b, b d, b d -> b d', alive, w.env.block_pose, trajs['block_pose'][:, step])
        trajs['block_cog'][:, step + 1] = einx.where('b, b d, b d -> b d', alive, w.env._cog(), trajs['block_cog'][:, step])

    def pad_trajs(self, trajs: dict, step: int, max_steps: int):
        for v in trajs.values():
            v[:, step + 1:] = v[:, step:step + 1]

    def render_showcase(self, init_states: np.ndarray, eval_res: dict, title: str, out_dir: Path, stem: str, *, seed: int = 0):
        pick = pick_indices(eval_res['max_cov'], count = 6)
        self.wm2.render_trajectory_showcase(self.fp, eval_res, pick, None, title, out_dir / f'{stem}_final.png', out_dir / f'{stem}.gif', out_dir / f'{stem}.mp4')

def make_backend(env: str, *, x_jepa_dir: str = '../x-jepa', step_scale: float = 40., num_workers: int = 8, envs_per_worker: int = 2, action_scale: tuple[float, float, float] = (0.45, 0.45, 0.05)):
    if env == 'pusht2d':
        return PushT2DBackend(x_jepa_dir = x_jepa_dir, step_scale = step_scale)
    if env == 'pusht3d':
        from pusht3d_env import PushT3DBackend
        return PushT3DBackend(num_workers = num_workers, envs_per_worker = envs_per_worker, action_scale = action_scale)
    raise ValueError(f'Unknown env {env!r}; expected pusht2d or pusht3d')

def fit_state_compressor_generic(backend, comp_dev: torch.device, *, skill_len: int = 45, dim_latent: int = 8, horizon: int = 4, warmup_rounds: int = 16, steps: int = 1000, seed: int = 42):
    """Fit epiplexity + slowness StateCompressor with empirical SVD whitening onto active subspace."""
    if hasattr(backend, 'fit_state_compressor'):
        return backend.fit_state_compressor(comp_dev, skill_len = skill_len, dim_latent = dim_latent, horizon = horizon, warmup_rounds = warmup_rounds, steps = steps, seed = seed)

    t0 = time.perf_counter()
    lag_min, lag_max = max(1, int(round(0.30 * skill_len))), max(1, int(round(0.70 * skill_len)))
    w_len = max(60, lag_max + horizon + 20)
    warm_obs, warm_act = backend.collect_random_episodes(warmup_rounds * backend.pool_size, ep_len = w_len, hold = 6, seed = seed)

    flat = warm_obs.reshape(-1, backend.state_dim)
    comp = StateCompressor(
        GeneralStateEncoder(flat.mean(0), flat.std(0), backend.state_dim, dim_latent),
        dim_latent = dim_latent, dim_action = backend.action_dim, obs_ndim = 1, regularizer = 'epiplexity',
        dynamics_loss_weight = 1., rollout_discount = 0.8, reg_weight = 0.10, l2norm_latent = False, eta = 20., mlp_reservoir_layers = 2,
    ).to(comp_dev)

    opt = torch.optim.AdamW(trainable_parameters(comp), lr = 3e-3, weight_decay = 1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = steps, eta_min = 1e-5)
    max_ts, wts = w_len - max(horizon, lag_max), comp.rollout_discount ** arange(horizon, device = comp_dev)

    for _ in range(steps):
        ep = randint(0, warm_obs.shape[0], (256,))
        ts = randint(0, max_ts + 1, (256,))
        lags = randint(lag_min, lag_max + 1, (256,))
        o_b = warm_obs[ep[:, None], ts[:, None] + arange(horizon + 1)].to(comp_dev)
        o_lag = warm_obs[ep, ts + lags].to(comp_dev)
        a_b = warm_act[ep[:, None], ts[:, None] + arange(horizon)].to(comp_dev)

        h_j, h_lag = comp.encode(o_b), comp.encode(o_lag)
        pred_loss = ((comp.rollout(h_j[:, 0], a_b) - h_j[:, 1:]).square().mean(-1) * wts).sum(-1).mean() / wts.sum()
        slow_loss = (h_lag - h_j[:, 0]).square().sum(-1).mean()
        energy_loss = 0.5 * (h_j[:, 0].square().sum(-1).mean() + h_lag.square().sum(-1).mean())
        step_opt(opt, pred_loss - 0.10 * comp.regularizer(h_j[:, 0], h_lag) + 20. * slow_loss + 0.20 * energy_loss)
        sch.step()
    comp.eval()

    idx = torch.randperm(flat.shape[0])[:1500]
    lat_raw = comp.encode_detached(flat[idx].to(comp_dev)).cpu()
    mu_raw = lat_raw.mean(0)
    _, sv_raw, vh = torch.linalg.svd(lat_raw - mu_raw, full_matrices = False)
    sv_raw = sv_raw / math.sqrt(lat_raw.shape[0])
    dim_eff = max(2, min(dim_latent, int((sv_raw >= 0.25 * sv_raw[0]).sum().item())))
    comp.encoder[1] = AffineProject(mu_raw, (0.40 / sv_raw[:dim_eff].clamp(min = 1e-4))[:, None] * vh[:dim_eff]).to(comp_dev)

    lat = comp.encode_detached(flat[idx].to(comp_dev)).cpu()
    r2_lin = dict(blk_xy = round(ridge_r2(lat, flat[idx, backend.blk_xy_slice]), 3), blk_rot = round(ridge_r2(lat, flat[idx, backend.blk_rot_slice]), 3))
    dh = (comp.encode_detached(warm_obs[:, -1].to(comp_dev)) - comp.encode_detached(warm_obs[:, 0].to(comp_dev))).norm(dim = -1).cpu()
    db = (warm_obs[:, -1, backend.blk_xy_slice] - warm_obs[:, 0, backend.blk_xy_slice]).norm(dim = -1) * backend.dist_scale
    dh_ratio = float(dh[db > 1.5].mean().item()) / max(float(dh[db < 0.2].mean().item()), 1e-6) if ((db > 1.5).any() and (db < 0.2).any()) else 1.

    print(f'=== StateCompressor ({backend.name}, d_lat={dim_latent} -> auto k={dim_eff}, {steps} steps in {time.perf_counter() - t0:.1f}s) ===')
    print(f'  raw_sv={[round(x, 2) for x in sv_raw.tolist()]} | ||dh|| contrast={dh_ratio:.1f}x | R2 blk_xy={r2_lin["blk_xy"]:.3f} blk_rot={r2_lin["blk_rot"]:.3f}', flush = True)
    return comp, dim_eff, warm_obs, warm_act, dict(lat_sv = sv_raw.tolist(), r2_lin = r2_lin, dh_ratio = round(dh_ratio, 1))

# world model buffer & training

class WorldModelBuffer:
    def __init__(self, path: Path, backend, *, max_episodes: int = 12_000, max_len: int = 80):
        self.backend, self.max_len = backend, max_len
        self.state_dim, self.action_dim = backend.state_dim, backend.action_dim
        self.buf = ReplayBuffer(str(path), max_episodes = max_episodes, max_timesteps = max_len + 1, fields = dict(obs = ('float', (self.state_dim,)), action = ('float', (self.action_dim,))), overwrite = True)
        self.ep_lens: list[int] = []
        self.total_env_steps = self.contact_steps = 0

    def add_batch(self, obs_seq: Tensor, act_seq: Tensor):
        b, t = act_seq.shape[:2]
        assert obs_seq.shape[:2] == (b, t + 1) and t <= self.max_len
        pad_act = cat((act_seq, torch.zeros(b, 1, self.action_dim)), dim = 1)
        with self.buf.batched_episode(batch_size = b):
            for step in range(t + 1):
                self.buf.store_batch(obs = obs_seq[:, step], action = pad_act[:, step])
        self.ep_lens.extend([t] * b)
        self.total_env_steps += b * t
        step_move = (obs_seq[:, 1:, self.backend.blk_xy_slice] - obs_seq[:, :-1, self.backend.blk_xy_slice]).norm(dim = -1) * self.backend.dist_scale
        self.contact_steps += int((step_move > self.backend.contact_step_thresh).sum().item())

    @property
    def contact_step_pct(self) -> float:
        return 100. * self.contact_steps / max(1, self.total_env_steps)

    def clone_to(self, new_path: Path) -> 'WorldModelBuffer':
        other = WorldModelBuffer(new_path, self.backend, max_episodes = self.buf.max_episodes, max_len = self.max_len)
        n = self.buf.num_episodes
        if n == 0:
            return other
        obs_np, act_np, i = self.buf.data['obs'][:n], self.buf.data['action'][:n], 0
        while i < n:
            j = i + 1
            while j < n and self.ep_lens[j] == self.ep_lens[i]:
                j += 1
            length = self.ep_lens[i]
            other.add_batch(torch.from_numpy(obs_np[i:j, :length + 1].copy()), torch.from_numpy(act_np[i:j, :length].copy()))
            i = j
        return other

def update_world_model(backend, wm: StateCompressor, wm_buf: WorldModelBuffer, device: torch.device, *, train_horizon: int, steps: int, batch_size: int, lr_start: float, lr_end: float, seed: int) -> tuple[Module, dict]:
    torch.manual_seed(seed)
    wm.train()
    opt = torch.optim.AdamW(wm.parameters(), lr = lr_start, weight_decay = 1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = steps, eta_min = lr_end)

    n_ep = wm_buf.buf.num_episodes
    obs_d = torch.from_numpy(wm_buf.buf.data['obs'][:n_ep].copy()).to(device)
    act_d = torch.from_numpy(wm_buf.buf.data['action'][:n_ep].copy()).to(device)
    lens_d = torch.tensor(wm_buf.ep_lens, dtype = torch.long, device = device)
    max_ts = (lens_d - train_horizon).clamp(min = 0)
    ah1, ah = arange(train_horizon + 1, device = device), arange(train_horizon, device = device)

    t0 = time.perf_counter()
    for _ in range(steps):
        ep = randint(0, n_ep, (batch_size,), device = device)
        ts = (rand(batch_size, device = device) * (max_ts[ep] + 1).float()).long()
        loss, bd = wm.loss(obs_d[ep[:, None], ts[:, None] + ah1], act_d[ep[:, None], ts[:, None] + ah], return_breakdown = True)
        step_opt(opt, loss, wm.parameters(), max_norm = 5.)
        sch.step()
    wm.eval()

    flat_x = obs_d[arange(obs_d.shape[1], device = device)[None, :] <= lens_d[:, None]]
    with torch.no_grad():
        x_tr = flat_x[torch.randperm(flat_x.shape[0], device = device)[:min(14_000, flat_x.shape[0])]]
        z_tr = wm.encode_detached(x_tr)
        r2_blk, r2_rot = round(ridge_r2(z_tr, x_tr[:, backend.blk_xy_slice]), 3), round(ridge_r2(z_tr, x_tr[:, backend.blk_rot_slice]), 3)

    probe = build_probe(wm.dim_latent, backend.state_dim).to(device)
    p_opt = torch.optim.Adam(probe.parameters(), lr = 4e-3)
    for _ in range(250):
        bi = randint(0, z_tr.shape[0], (1024,), device = device)
        step_opt(p_opt, (probe(z_tr[bi]) - x_tr[bi]).square().mean())
    probe.eval()
    return probe, dict(wm_sec = round(time.perf_counter() - t0, 1), pred_loss = round(float(bd.pred), 5), sigreg = round(float(bd.reg), 3), r2_blk = r2_blk, r2_rot = r2_rot)

# mode conditioning & metrics

def explore_obs(s: Tensor, s0: Tensor, z: Tensor, dim_eff: int) -> Tensor:
    h_cur, h0 = s[..., -dim_eff:], s0[..., -dim_eff:]
    mode = s.new_tensor((1., 0.)).expand(*s.shape[:-1], 2)
    return cat((s, h0, h_cur - h0, z, mode), dim = -1)

def task_obs(s: Tensor, dim_eff: int, dim_skill: int) -> Tensor:
    pad = s.new_zeros(*s.shape[:-1], 2 * dim_eff + dim_skill)
    mode = s.new_tensor((0., 1.)).expand(*s.shape[:-1], 2)
    return cat((s, pad, mode), dim = -1)

def explore_q(critics, s: Tensor, s0: Tensor, z: Tensor, a: Tensor, dim_eff: int) -> Tensor:
    return critics(explore_obs(s, s0, z, dim_eff), cont_actions = a)[0].mean(dim = -1)

@torch.no_grad()
def eval_h10_error(backend, wm: StateCompressor, probe: Module, test_push: dict, device: torch.device) -> tuple[float, float]:
    obs_te, act_te = test_push['obs'], test_push['act']
    pred = probe(wm.rollout(wm.encode_detached(obs_te[:, 0].to(device)), act_te.to(device))).cpu()
    blk_err = float((pred[:, 9, backend.blk_xy_slice] - obs_te[:, 10, backend.blk_xy_slice]).norm(dim = -1).mean() * backend.dist_scale)
    th_p = torch.atan2(pred[:, 9, backend.blk_rot_slice][..., 1], pred[:, 9, backend.blk_rot_slice][..., 0])
    th_t = torch.atan2(obs_te[:, 10, backend.blk_rot_slice][..., 1], obs_te[:, 10, backend.blk_rot_slice][..., 0])
    rot_err = float(torch.atan2(torch.sin(th_p - th_t), torch.cos(th_p - th_t)).abs().mean() * 180. / math.pi)
    return round(blk_err, 2), round(rot_err, 1)

def summarize_docking(res: dict) -> dict:
    cov, won = res['max_cov'], res['won']
    win_lens = res['lengths'][won].float()
    return dict(
        win_pct = round(float(won.float().mean() * 100.), 1),
        ge80_pct = round(float((cov >= 0.80).float().mean() * 100.), 1),
        ge60_pct = round(float((cov >= 0.60).float().mean() * 100.), 1),
        mean_cov = round(float(cov.mean()), 3),
        fin_dist = round(float(res['fin_dist'].mean()), 2),
        win_steps = round(float(win_lens.mean()), 1) if win_lens.numel() > 0 else 0.,
    )

# flat off-policy transition buffer

class TransitionBuffer:
    def __init__(self, path: Path, dim_state: int, dim_action: int, *, max_transitions: int = 250_000, store_reward_done: bool = True):
        self.store_reward_done = store_reward_done
        fields = dict(state = ('float', dim_state), action = ('float', dim_action))
        if store_reward_done:
            fields.update(next_state = ('float', dim_state), reward = 'float', done = 'bool')
        self.buf = ReplayBuffer(str(path), max_episodes = max_transitions, max_timesteps = 1, fields = fields, circular = True, overwrite = True, flush_every_store_step = 4096)

    @property
    def size(self) -> int:
        return int(self.buf.num_episodes)

    def add(self, state: Tensor, action: Tensor, next_state: Tensor | None = None, reward: Tensor | None = None, done: Tensor | None = None):
        n = int(state.shape[0])
        if n == 0:
            return
        kw = dict(state = state.detach().cpu(), action = action.detach().cpu())
        if self.store_reward_done:
            assert exists(next_state) and exists(reward) and exists(done)
            kw.update(next_state = next_state.detach().cpu(), reward = reward.detach().cpu().float(), done = done.detach().cpu().bool())
        with self.buf.batched_episode(batch_size = n):
            self.buf.store_batch(**kw)

    def sample(self, batch_size: int):
        return self.buf.sample(batch_size, timestep_level = True, replacement = self.size < batch_size, to_named_tuple = True)

def add_episodes_to_wm(wm_buf: WorldModelBuffer, obs_seq: Tensor, act_seq: Tensor, lengths: Tensor, *, min_len: int = 10):
    by_len: dict[int, list[int]] = {}
    for i, ep_len in enumerate(lengths.detach().cpu().tolist()):
        if ep_len >= min_len:
            by_len.setdefault(ep_len, []).append(i)
    obs_cpu, act_cpu = obs_seq.detach().cpu(), act_seq.detach().cpu()
    for ep_len, idxs in by_len.items():
        idx = torch.tensor(idxs, dtype = torch.long)
        wm_buf.add_batch(obs_cpu[idx, :ep_len + 1], act_cpu[idx, :ep_len])

def add_episodes_to_buffer(buf: TransitionBuffer, state_seq: Tensor, act_seq: Tensor, rew_seq: Tensor, done_seq: Tensor, lengths: Tensor):
    mask = rearrange(arange(act_seq.shape[1]), 't -> 1 t') < rearrange(lengths.cpu(), 'b -> b 1')
    if not mask.any():
        return
    buf.add(
        state_seq[:, :-1][mask], act_seq[mask],
        state_seq[:, 1:][mask] if buf.store_reward_done else None,
        rew_seq[mask] if buf.store_reward_done else None,
        done_seq[mask] if buf.store_reward_done else None,
    )

# unified mode-conditioned SEA + sparse-task SAC agent

class ModeConditionedSEA:
    def __init__(self, backend, comp: StateCompressor | None, dim_eff: int, warm_obs: Tensor | None, warm_act: Tensor | None, tmp_dir: Path, *, tag: str = 'u', num_envs: int = 16, skill_len: int = 45, max_sw: int | None = None, total_rounds: int = 120, pot_start_round: int = 10, dim_hidden: int = 128, discount: float = 0.97, task_reward_scale: float = 10., bc_lr: float = 1e-3, seed: int = 42):
        self.backend, self.comp = backend, comp
        self.dim_eff = self.dim_skill = dim_eff if exists(comp) else 0
        self.state_dim, self.action_dim = backend.state_dim, backend.action_dim
        self.dim_state = self.state_dim + self.dim_eff
        self.dim_unified = self.dim_state + 2 * self.dim_eff + self.dim_skill + 2 if exists(comp) else self.state_dim
        self.num_envs, self.skill_len = num_envs, skill_len
        self.max_sw = default(max_sw, backend.default_max_sw)
        self.total_rounds, self.pot_start_round, self.discount = total_rounds, pot_start_round, discount
        self.task_reward_scale = float(task_reward_scale)
        self.comp_dev, self.seed = torch.device('cpu'), seed
        self.rnd = self.win_cem_eps = self.win_task_eps = 0
        self._seeded = False

        self.sac = make_sac(self.dim_unified, self.action_dim, discount = discount, lr = 1e-3, dim_hidden = dim_hidden)
        self.bc_opt = torch.optim.Adam(self.sac.actor.parameters(), lr = bc_lr)

        if exists(comp):
            self.classifier = SkillClassifier(
                ReZeroMLP(self.dim_eff, 64, self.dim_skill, depth = 1, orthogonal_skip_gain = 0.5),
                total_steps = total_rounds, anneal_fracs = (0.35, 0.90), temperature = 0.10, noise_std = 0.05,
                num_samples = 512, sigreg_weight = 0.03, sigreg_scale = 0.20, extract_obs = lambda s: s[..., -self.dim_eff:],
            )
            self.opt_clf = torch.optim.Adam(self.classifier.parameters(), lr = 3e-4)
            self.pot_sac = make_sac(self.dim_state, self.action_dim, discount = 0.95, lr = 1e-3, dim_hidden = 64, actor_update_freq = 1)

            fields = dict(state = ('float', self.dim_state), action = ('float', self.action_dim), next_state = ('float', self.dim_state))
            self.skill_buf = ReplayBuffer(tmp_dir / f'{tag}_skill', max_episodes = 4096, max_timesteps = skill_len, fields = fields, meta_fields = dict(anchor = ('float', self.dim_state), skill = ('float', self.dim_skill)), circular = True, overwrite = True, track_write_step = True, flush_every_store_step = skill_len)
            max_pot_t = max(60, skill_len + self.max_sw, int(warm_act.shape[1]) if exists(warm_act) else 0)
            self.pot_buf = ReplayBuffer(tmp_dir / f'{tag}_pot', max_episodes = 4096, max_timesteps = max_pot_t, fields = fields, circular = True, overwrite = True, flush_every_store_step = max_pot_t)
            if exists(warm_obs) and exists(warm_act):
                add_episodes(self.pot_buf, self.encode_raw(warm_obs), warm_act)
        else:
            self.classifier = self.opt_clf = self.pot_sac = self.skill_buf = self.pot_buf = None

        self.task_buf = TransitionBuffer(tmp_dir / f'{tag}_task', self.dim_state, self.action_dim, max_transitions = 400_000)
        self.win_buf = TransitionBuffer(tmp_dir / f'{tag}_win', self.dim_state, self.action_dim, max_transitions = 200_000)
        self.bc_win_buf = TransitionBuffer(tmp_dir / f'{tag}_bc_win', self.dim_state, self.action_dim, max_transitions = 120_000, store_reward_done = False)
        self.bc_plan_buf = TransitionBuffer(tmp_dir / f'{tag}_bc_plan', self.dim_state, self.action_dim, max_transitions = 40_000, store_reward_done = False)

    @torch.no_grad()
    def encode_raw(self, raw_obs: Tensor, *, device: torch.device | None = None) -> Tensor:
        raw_d = raw_obs.to(default(device, self.comp_dev))
        if not exists(self.comp):
            return raw_d
        h = self.comp.encode_detached(raw_d.reshape(-1, self.state_dim)).reshape(*raw_d.shape[:-1], self.dim_eff)
        return cat((raw_d, h), dim = -1)

    @torch.no_grad()
    def encode_task_obs(self, raw_obs: Tensor, *, device: torch.device | None = None) -> Tensor:
        s = self.encode_raw(raw_obs, device = device)
        return task_obs(s, self.dim_eff, self.dim_skill) if exists(self.comp) else s

    @torch.no_grad()
    def rollout_explore_eval(self, *, num_envs: int = 32, seed: int = 777) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        assert exists(self.comp)
        self.pot_sac.eval(); self.sac.eval()
        z = sample_skills(num_envs, self.dim_skill, generator = torch.Generator().manual_seed(seed))
        s0_holder, max_sw = {}, self.max_sw

        def act_fn(t: int, raw_t: Tensor, slc: slice | None = None) -> Tensor:
            key, s = slc.start if exists(slc) else 0, self.encode_raw(raw_t)
            if t < max_sw:
                return act_sac(self.pot_sac, s, deterministic = True)
            if t == max_sw:
                s0_holder[key] = s.clone()
            return act_sac(self.sac, explore_obs(s, s0_holder[key], z[slc] if exists(slc) else z, self.dim_eff), deterministic = True)

        return self.backend.rollout_explore_batch(num_envs, max_sw + self.skill_len, act_fn, seed = seed)

    @torch.no_grad()
    def collect_explore(self, *, use_potential: bool):
        assert exists(self.comp)
        n, z = self.num_envs, sample_skills(self.num_envs, self.dim_skill)
        if not use_potential or self.max_sw <= 0:
            switch_t, total_steps = torch.zeros(n, dtype = torch.long), self.skill_len
        else:
            switch_t = randint(1, self.max_sw + 1, (n,))
            switch_t[:int(round(n * 0.25))] = 0
            total_steps = self.max_sw + self.skill_len

        s0_holder = {}

        def act_fn(t: int, raw_t: Tensor, slc: slice | None = None) -> Tensor:
            key, s = slc.start if exists(slc) else 0, self.encode_raw(raw_t)
            sw_b = switch_t[slc] if exists(slc) else switch_t
            z_b = z[slc] if exists(slc) else z
            if t == 0:
                s0_holder[key] = s.clone()
            at_sw = sw_b == t
            if at_sw.any():
                s0_holder[key] = einx.where('b, b d, b d -> b d', at_sw, s, s0_holder[key])
            a = act_sac(self.sac, explore_obs(s, s0_holder[key], z_b, self.dim_eff))
            if use_potential and self.max_sw > 0:
                a = einx.where('b, b d, b d -> b d', t >= sw_b, a, act_sac(self.pot_sac, s))
            return a

        raw_seq, all_a, _, _ = self.backend.rollout_explore_batch(n, total_steps, act_fn)
        all_s = self.encode_raw(raw_seq)
        t_idx = rearrange(switch_t, 'b -> b 1') + rearrange(arange(self.skill_len + 1), 't -> 1 t')
        env_idx = rearrange(arange(n), 'b -> b 1')
        return all_s[env_idx, t_idx], all_a[env_idx, t_idx[:, :-1]], z, all_s, all_a

    def store_explore(self, wm_buf: WorldModelBuffer, *, use_potential: bool) -> int:
        ss, sa, z, all_s, all_a = self.collect_explore(use_potential = use_potential)
        add_episodes(self.skill_buf, ss, sa, anchor = ss[:, 0], skill = z)
        add_episodes(self.pot_buf, all_s, all_a)
        wm_buf.add_batch(all_s[:, :, :self.state_dim], all_a)
        return int(all_a.shape[0] * all_a.shape[1])

    def _update_classifier(self, steps: int, batch_size: int):
        for _ in range(steps):
            cb = self.skill_buf.sample(
                batch_size, horizon = self.skill_len, max_age = 256 * self.skill_len, fallback_on_empty = True,
                replacement = (min(self.skill_buf.num_episodes, batch_size) < batch_size) or (self.rnd < 20),
                fields = ('next_state',), to_named_tuple = ('anchor', 'next_state', 'skill'),
            )
            lag = sample_truncated_geometric(torch.full((batch_size,), self.skill_len, dtype = torch.long), 0.96) - 1
            step_opt(self.opt_clf, self.classifier(cb.anchor, cb.next_state[arange(batch_size), lag], cb.skill), self.classifier.parameters(), max_norm = 10.)

    def _update_potential(self, steps: int, batch_size: int):
        if self.pot_buf.num_episodes == 0:
            return
        zeros_done = torch.zeros(batch_size, dtype = torch.bool)
        for _ in range(steps):
            pb = self.pot_buf.sample(batch_size, timestep_level = True, fields = ('state', 'action', 'next_state'), to_named_tuple = True)
            with torch.no_grad():
                self.sac.critics_target.eval()
                r_pot = estimate_empowerment(
                    pb.next_state, partial(explore_q, self.sac.critics_target, dim_eff = self.dim_eff), self.dim_skill,
                    num_skills = 8, policy = lambda st, st0, z_: act_sac(self.sac, explore_obs(st, st0, z_, self.dim_eff)),
                    discount = self.discount, clamp_min = 0.,
                )
            self.pot_sac(states = pb.state, cont_actions = pb.action, discrete_actions = None, rewards = r_pot, done = zeros_done, next_states = pb.next_state)

    def _sample_explore_batch(self, batch_size: int, her_prob: float):
        b = self.skill_buf.sample(batch_size, timestep_level = True, fields = ('state', 'action', 'next_state'), return_indices = True, to_named_tuple = True)
        z = b.skill
        if her_prob > 0.:
            with torch.no_grad():
                ep_i, t_i = b.indices[:, 0], b.indices[:, 1]
                rem = self.skill_len - t_i
                t_fut = einx.where('b, , b -> b', rand(batch_size) < 0.5, self.skill_len - 1, t_i + (rand(batch_size) * rem.float()).long().clamp(max = rem - 1))
                s_fut = torch.from_numpy(self.skill_buf.data['next_state'][ep_i.numpy(), t_fut.numpy()].copy())
                dh = self.classifier.encode_state(s_fut) - self.classifier.encode_state(b.anchor)
                do_her = (rand(batch_size) < her_prob) & (dh.norm(dim = -1) > self.classifier.temperature)
                if do_her.any():
                    z = einx.where('b, b d, b d -> b d', do_her, F.normalize(dh, dim = -1), z)
        r_eff = self.classifier.reward(b.anchor, b.next_state, z, increment_step = False)
        return explore_obs(b.state, b.anchor, z, self.dim_eff), b.action, r_eff, torch.zeros(batch_size, dtype = torch.bool), explore_obs(b.next_state, b.anchor, z, self.dim_eff)

    def _sample_task_batch(self, batch_size: int):
        if self.win_buf.size > 0:
            n_win = batch_size // 2
            b1, b2 = self.task_buf.sample(batch_size - n_win), self.win_buf.sample(n_win)
            s, a, ns = cat((b1.state, b2.state)), cat((b1.action, b2.action)), cat((b1.next_state, b2.next_state))
            r, d = cat((b1.reward, b2.reward)) * self.task_reward_scale, cat((b1.done, b2.done))
        else:
            b = self.task_buf.sample(batch_size)
            s, a, ns, r, d = b.state, b.action, b.next_state, b.reward * self.task_reward_scale, b.done
        return (task_obs(s, self.dim_eff, self.dim_skill), a, r, d, task_obs(ns, self.dim_eff, self.dim_skill)) if exists(self.comp) else (s, a, r, d, ns)

    def _sample_bc_batch(self, batch_size: int, min_win_eps: int) -> tuple[Tensor, Tensor] | None:
        if self.win_cem_eps >= min_win_eps and self.bc_win_buf.size >= 32:
            b = self.bc_win_buf.sample(min(batch_size, self.bc_win_buf.size))
            return b.state, b.action
        if self.bc_win_buf.size >= 16 and self.bc_plan_buf.size >= 16:
            half = batch_size // 2
            bw, bp = self.bc_win_buf.sample(min(half, self.bc_win_buf.size)), self.bc_plan_buf.sample(min(half, self.bc_plan_buf.size))
            return cat((bw.state, bp.state)), cat((bw.action, bp.action))
        if self.bc_plan_buf.size >= 16:
            b = self.bc_plan_buf.sample(min(batch_size, self.bc_plan_buf.size))
            return b.state, b.action
        return None

    def train_explore_until(self, wm_buf: WorldModelBuffer, target_steps: int):
        assert exists(self.comp)
        self.pot_sac.train(); self.sac.train(); self.classifier.train()
        if not self._seeded:
            for _ in range(4):
                self.store_explore(wm_buf, use_potential = False)
            self._seeded = True

        while wm_buf.total_env_steps < target_steps:
            self.rnd += 1
            self.store_explore(wm_buf, use_potential = self.rnd > self.pot_start_round)
            self.classifier.set_step_(min(self.rnd, self.total_rounds))
            self._update_classifier(steps = 10, batch_size = 256)

            her_prob = 0.50 * max(0., 1. - (self.rnd - 1) / max(1, self.total_rounds - 1))
            for _ in range(64):
                s, a, r, d, ns = self._sample_explore_batch(256, her_prob)
                self.sac(states = s, cont_actions = a, discrete_actions = None, rewards = r, done = d, next_states = ns)
            if self.rnd >= self.pot_start_round:
                self._update_potential(steps = 24, batch_size = 256)
            if self.rnd % 15 == 0:
                ex = evaluate_explore_mode(self, num_envs = 16, seed = 777)
                print(f'    [SEA rnd={self.rnd:3d}] steps={wm_buf.total_env_steps:7,d} | wm_contact={wm_buf.contact_step_pct:5.2f}% | pot_d={ex["exp_pot_dist"]:4.1f}{self.backend.dist_unit} touch={ex["exp_touch_pct"]:5.1f}% disp={ex["exp_disp"]:4.1f}{self.backend.dist_unit}', flush = True)
                self.pot_sac.train(); self.sac.train()

    def update_mixed(self, *, sac_steps: int = 600, bc_every: int = 3, batch_size: int = 256, explore_frac: float = 0.30, min_win_eps_for_pure_bc: int = 64, q_filter_bc: bool = True, use_bc: bool = True) -> dict:
        self.sac.train()
        has_explore = exists(self.comp) and self.skill_buf.num_episodes >= 8
        if has_explore:
            self.pot_sac.train(); self.classifier.train()
            self._update_classifier(steps = 6, batch_size = batch_size)
        assert self.task_buf.size >= 32

        n_exp = int(round(batch_size * explore_frac)) if has_explore else 0
        n_tsk = batch_size - n_exp
        bc_losses, bc_keeps, q_mean, q_max = [], [], 0., 0.

        for step_i in range(sac_steps):
            s_t, a_t, r_t, d_t, ns_t = self._sample_task_batch(n_tsk)
            if n_exp > 0:
                s_e, a_e, r_e, d_e, ns_e = self._sample_explore_batch(n_exp, her_prob = 0.10)
                s_m, a_m, r_m, d_m, ns_m = cat((s_e, s_t)), cat((a_e, a_t)), cat((r_e, r_t)), cat((d_e, d_t)), cat((ns_e, ns_t))
            else:
                s_m, a_m, r_m, d_m, ns_m = s_t, a_t, r_t, d_t, ns_t
            self.sac(states = s_m, cont_actions = a_m, discrete_actions = None, rewards = r_m, done = d_m, next_states = ns_m)

            if step_i == sac_steps - 1:
                with torch.no_grad():
                    qt = self.sac.critics(s_t, cont_actions = a_t)[0].mean(dim = -1)
                    q_mean, q_max = float(qt.mean()), float(qt.amax())

            if use_bc and bc_every > 0 and (step_i + 1) % bc_every == 0:
                bc_batch = self._sample_bc_batch(128, min_win_eps_for_pure_bc)
                if exists(bc_batch):
                    s_bc, a_bc = bc_batch
                    obs_bc = task_obs(s_bc, self.dim_eff, self.dim_skill) if exists(self.comp) else s_bc
                    a_pred = self.sac.actor.deterministic_action(obs_bc)
                    per_ex = (a_pred - a_bc).square().mean(dim = -1)

                    keep = torch.ones(per_ex.shape[0], dtype = torch.bool)
                    if q_filter_bc and (self.win_cem_eps + self.win_task_eps) >= 16:
                        with torch.no_grad():
                            q_bc = self.sac.critics(obs_bc, cont_actions = a_bc)[0].mean(dim = -1)
                            q_pi = self.sac.critics(obs_bc, cont_actions = a_pred.detach())[0].mean(dim = -1)
                            if float(q_bc.amax()) > 2.:
                                keep = q_bc >= (q_pi - 0.25)

                    bc_keeps.append(float(keep.float().mean()))
                    if keep.any():
                        w = keep.float()
                        loss_bc = (per_ex * w).sum() / w.sum().clamp(min = 1.)
                        step_opt(self.bc_opt, loss_bc, self.sac.actor.parameters(), max_norm = 5.)
                        bc_losses.append(float(loss_bc.detach()))

        if has_explore:
            self._update_potential(steps = 12, batch_size = batch_size)
        return dict(
            q_task_mean = round(q_mean, 3), q_task_max = round(q_max, 3),
            bc_loss = round(float(np.mean(bc_losses)), 4) if bc_losses else 0.,
            bc_keep = round(float(np.mean(bc_keeps)), 3) if bc_keeps else 0.,
            alpha = round(float(self.sac.learned_entropy_temperature.alpha.detach()), 4),
        )

    def ingest_rollouts(self, wm_buf: WorldModelBuffer, res: dict, *, from_cem: bool = False, train_horizon: int = 10, top_plan_frac: float = 0.20, max_plan_boot_wins: int = 64):
        obs_seq, act_seq, rew_seq, done_seq, lengths = res['obs_seq'], res['act_seq'], res['rew_seq'], res['done_seq'], res['lengths']
        add_episodes_to_wm(wm_buf, obs_seq, act_seq, lengths, min_len = train_horizon)
        state_seq = self.encode_raw(obs_seq)
        add_episodes_to_buffer(self.task_buf, state_seq, act_seq, rew_seq, done_seq, lengths)

        won = res['won']
        if won.any():
            n_won = int(won.sum().item())
            add_episodes_to_buffer(self.win_buf, state_seq[won], act_seq[won], rew_seq[won], done_seq[won], lengths[won])
            if from_cem:
                self.win_cem_eps += n_won
                add_episodes_to_buffer(self.bc_win_buf, state_seq[won], act_seq[won], rew_seq[won], done_seq[won], lengths[won])
            else:
                self.win_task_eps += n_won

        if from_cem and self.win_cem_eps < max_plan_boot_wins and 'pred_score_seq' in res:
            pred_sc = res['pred_score_seq']
            k = max(2, int(round(obs_seq.shape[0] * top_plan_frac)))
            top = pred_sc.amax(dim = 1).topk(k).indices
            trunc = torch.minimum(lengths[top], pred_sc.argmax(dim = 1)[top].clamp(min = 1))
            add_episodes_to_buffer(self.bc_plan_buf, state_seq[top], act_seq[top], rew_seq[top], done_seq[top], trunc)

# rollouts, policy+Q-guided CEM, and evaluation

@torch.no_grad()
def run_task_chunk(backend, action_fn, init_states: np.ndarray | None, *, num_envs: int, max_steps: int, seed: int, score_obs_fn = None, return_trajs: bool = False) -> dict:
    n = len(init_states) if exists(init_states) else num_envs
    raw0, cov0, ctx = backend.reset_chunk(init_states, n, seed)

    obs_seq = torch.zeros(n, max_steps + 1, backend.state_dim)
    act_seq = torch.zeros(n, max_steps, backend.action_dim)
    rew_seq = torch.zeros(n, max_steps)
    done_seq = torch.zeros(n, max_steps, dtype = torch.bool)
    cov_seq = torch.zeros(n, max_steps + 1)
    pred_score_seq = torch.full((n, max_steps + 1), -1e9) if exists(score_obs_fn) else None

    obs_seq[:, 0], cov_seq[:, 0] = raw0, cov0
    if exists(score_obs_fn):
        pred_score_seq[:, 0] = score_obs_fn(obs_seq[:, 0])
    trajs = backend.init_trajs(ctx, n, max_steps) if (return_trajs and backend.has_trajs) else None

    alive, lengths, step = ~backend.is_won(cov_seq[:, 0]), torch.zeros(n, dtype = torch.long), 0
    while step < max_steps and alive.any():
        a_exec = action_fn(step, obs_seq[:, step])
        nxt_raw, nxt_cov = backend.step_chunk(ctx, a_exec, alive)
        just_won = alive & backend.is_won(nxt_cov)

        act_seq[:, step] = einx.where('b, b d, b d -> b d', alive, a_exec, torch.zeros_like(a_exec))
        obs_seq[:, step + 1] = einx.where('b, b d, b d -> b d', alive, nxt_raw, obs_seq[:, step])
        cov_seq[:, step + 1] = einx.where('b, b, b -> b', alive, nxt_cov, cov_seq[:, step])
        rew_seq[:, step], done_seq[:, step] = just_won.float(), just_won
        if exists(score_obs_fn):
            pred_score_seq[:, step + 1] = einx.where('b, b, b -> b', alive, score_obs_fn(obs_seq[:, step + 1]), pred_score_seq[:, step])
        if exists(trajs):
            backend.record_traj(ctx, trajs, step, alive)

        lengths = lengths + alive.long()
        alive = alive & (~just_won)
        step += 1

    if exists(trajs) and step < max_steps:
        backend.pad_trajs(trajs, step, max_steps)
        cov_seq[:, step + 1:] = cov_seq[:, step:step + 1]

    out = dict(obs_seq = obs_seq, act_seq = act_seq, rew_seq = rew_seq, done_seq = done_seq, lengths = lengths, won = done_seq.any(dim = 1), max_cov = cov_seq.amax(dim = 1), fin_dist = backend.goal_dist(obs_seq[arange(n), lengths]), env_steps = int(lengths.sum().item()))
    if exists(pred_score_seq):
        out['pred_score_seq'] = pred_score_seq
    if exists(trajs):
        out.update(trajs)
    return out

@torch.no_grad()
def run_task_episodes(backend, action_fn_factory, init_states: np.ndarray | None, *, num_envs: int, max_steps: int = 80, seed: int = 0, score_obs_fn = None, return_trajs: bool = False) -> dict:
    n = len(init_states) if exists(init_states) else num_envs
    parts = [
        run_task_chunk(
            backend, action_fn_factory(min(n, start + backend.pool_size) - start, start),
            init_states[start:min(n, start + backend.pool_size)] if exists(init_states) else None,
            num_envs = min(n, start + backend.pool_size) - start, max_steps = max_steps,
            seed = seed + start, score_obs_fn = score_obs_fn, return_trajs = return_trajs,
        )
        for start in range(0, n, backend.pool_size)
    ]
    if len(parts) == 1:
        return parts[0]
    return {k: sum(p[k] for p in parts) if k == 'env_steps' else cat([p[k] for p in parts], dim = 0) for k in parts[0]}

@torch.no_grad()
def imagine_task_policy(agent: ModeConditionedSEA, wm: StateCompressor, probe: Module, z_cur: Tensor, horizon: int) -> tuple[Tensor, Tensor]:
    device = z_cur.device
    acts = torch.empty(z_cur.shape[0], horizon, agent.action_dim, device = device, dtype = z_cur.dtype)
    zs = torch.empty(z_cur.shape[0], horizon, z_cur.shape[-1], device = device, dtype = z_cur.dtype)
    z = z_cur
    agent.sac.eval()
    for t in range(horizon):
        a_t = act_sac(agent.sac, agent.encode_task_obs(probe(z), device = device), deterministic = True)
        z = wm.predict_next(z, a_t)
        acts[:, t], zs[:, t] = a_t, z
    return acts, zs

@torch.no_grad()
def run_policy_task(backend, sac: SAC, encode_fn, init_states: np.ndarray | None, *, num_envs: int, max_steps: int = 80, deterministic: bool = False, det_frac: float = 0., seed: int = 0, return_trajs: bool = False) -> dict:
    n = len(init_states) if exists(init_states) else num_envs
    sac.eval()
    n_det = int(round(n * det_frac))

    def factory(b: int, offset: int):
        det_mask = (arange(b) + offset) < n_det

        def policy_action(_step: int, raw_obs: Tensor) -> Tensor:
            x = encode_fn(raw_obs)
            if deterministic:
                return act_sac(sac, x, deterministic = True)
            if det_frac > 0.:
                return einx.where('b, b d, b d -> b d', det_mask, act_sac(sac, x, deterministic = True), act_sac(sac, x, deterministic = False))
            return act_sac(sac, x, deterministic = False)

        return policy_action

    return run_task_episodes(backend, factory, init_states, num_envs = num_envs, max_steps = max_steps, seed = seed, return_trajs = return_trajs)

@torch.no_grad()
def run_guided_cem(backend, wm: StateCompressor, probe: Module, agent: ModeConditionedSEA | None, init_states: np.ndarray | None, *, num_envs: int, device: torch.device, max_steps: int = 80, cem_horizon: int = 12, cem_pop: int = 256, cem_gens: int = 4, commit_k: int = 1, action_noise: float = 0., use_policy_guide: bool = False, q_cem_weight: float = 0., seed: int = 0, return_trajs: bool = False) -> dict:
    planner = CEMPlanner(wm, backend.action_dim, horizon = cem_horizon, pop_size = cem_pop, generations = cem_gens, elite_frac = 0.10)
    moved = exists(agent) and (use_policy_guide or q_cem_weight > 0.)
    if moved:
        if exists(agent.comp):
            agent.comp.to(device)
        agent.sac.actor.to(device); agent.sac.critics.to(device); agent.sac.eval()

    try:
        def score_fn(z_seq: Tensor) -> Tensor:
            base = backend.docking_probe_score(probe, z_seq)
            if q_cem_weight > 0. and exists(agent):
                b_sz, p_sz = z_seq.shape[:2]
                t_end = agent.encode_task_obs(probe(rearrange(z_seq[:, :, -1, :], 'b p d -> (b p) d')), device = device)
                q_end = agent.sac.critics(t_end, cont_actions = act_sac(agent.sac, t_end, deterministic = True))[0].mean(dim = -1)
                base = base + q_cem_weight * rearrange(q_end, '(b p) -> b p', b = b_sz, p = p_sz)
            return base

        def score_obs_fn(raw_obs: Tensor) -> Tensor:
            return backend.docking_probe_score(probe, rearrange(wm.encode_detached(raw_obs.to(device)), 'b d -> b 1 1 d'))[:, 0].cpu()

        def factory(b: int, offset: int):
            gen, cpu_gen = torch.Generator(device = device).manual_seed(seed + 13 + offset), torch.Generator().manual_seed(seed + 29 + offset)
            state = dict(warm = None, chosen = None)

            def cem_action(step: int, raw_obs: Tensor) -> Tensor:
                j = step % commit_k
                if j == 0:
                    z_cur = wm.encode_detached(raw_obs.to(device))
                    plan_warm, a_pi, sc_pi = state['warm'], None, None
                    if use_policy_guide and exists(agent):
                        a_pi, z_pi = imagine_task_policy(agent, wm, probe, z_cur, cem_horizon)
                        sc_pi = score_fn(rearrange(z_pi, 'b h d -> b 1 h d'))[:, 0]
                        if exists(plan_warm):
                            sc_w = score_fn(rearrange(wm.rollout(z_cur, plan_warm), 'b h d -> b 1 h d'))[:, 0]
                            plan_warm = einx.where('b, b h a, b h a -> b h a', sc_pi > sc_w, a_pi, plan_warm)
                        else:
                            plan_warm = a_pi

                    chosen = planner.plan(z_cur, score_fn, warm_mean = plan_warm, generator = gen).mean
                    if use_policy_guide and exists(a_pi) and exists(sc_pi):
                        sc_cm = score_fn(rearrange(wm.rollout(z_cur, chosen), 'b h d -> b 1 h d'))[:, 0]
                        chosen = einx.where('b, b h a, b h a -> b h a', sc_pi > sc_cm, a_pi, chosen)
                    state['chosen'], state['warm'] = chosen, planner.shift_warm_start(chosen, commit_k = commit_k)

                a_exec = state['chosen'][:, j].cpu()
                return (a_exec + action_noise * torch.randn(a_exec.shape, generator = cpu_gen)).clamp(-1., 1.) if action_noise > 0. else a_exec

            return cem_action

        return run_task_episodes(backend, factory, init_states, num_envs = num_envs, max_steps = max_steps, seed = seed, score_obs_fn = score_obs_fn, return_trajs = return_trajs)
    finally:
        if moved:
            if exists(agent.comp):
                agent.comp.to('cpu')
            agent.sac.actor.to('cpu'); agent.sac.critics.to('cpu')

@torch.no_grad()
def evaluate_explore_mode(agent: ModeConditionedSEA, *, num_envs: int = 32, seed: int = 777) -> dict:
    if not exists(agent.comp):
        return dict(exp_pot_dist = 0., exp_touch_pct = 0., exp_disp = 0.)
    _, _, pot_dist_seq, blk_pos_seq = agent.rollout_explore_eval(num_envs = num_envs, seed = seed)
    disp = (blk_pos_seq[:, -1] - blk_pos_seq[:, agent.max_sw]).norm(dim = -1)
    return dict(
        exp_pot_dist = round(float(pot_dist_seq[:, agent.max_sw].mean()), 1),
        exp_touch_pct = round(float((disp > agent.backend.touch_disp_thresh).float().mean() * 100.), 1),
        exp_disp = round(float(disp.mean()), 1),
    )

def evaluate_unified(backend, agent: ModeConditionedSEA, wm: StateCompressor, probe: Module, test_push: dict, eval_init: np.ndarray, device: torch.device, *, env_steps: int, max_steps: int, cem_horizon: int, cem_gens: int, seed: int, use_guide: bool, q_cem_weight: float, extra: dict | None = None, return_trajs: bool = False):
    ev_pi = run_policy_task(backend, agent.sac, agent.encode_task_obs, eval_init, num_envs = len(eval_init), max_steps = max_steps, deterministic = True, seed = seed, return_trajs = return_trajs)
    ev_cem = run_guided_cem(backend, wm, probe, agent if use_guide else None, eval_init, num_envs = len(eval_init), device = device, max_steps = max_steps, cem_horizon = cem_horizon, cem_pop = 384, cem_gens = cem_gens, use_policy_guide = use_guide, q_cem_weight = q_cem_weight if (use_guide and agent.win_buf.size >= 128) else 0., seed = seed, return_trajs = return_trajs)
    h10_blk, h10_rot = eval_h10_error(backend, wm, probe, test_push, device)
    row = dict(
        env_steps = env_steps, h10_blk = h10_blk, h10_rot = h10_rot,
        **{f'pi_{k}': v for k, v in summarize_docking(ev_pi).items()},
        **{f'cem_{k}': v for k, v in summarize_docking(ev_cem).items()},
        **evaluate_explore_mode(agent),
        win_cem_eps = agent.win_cem_eps, win_task_eps = agent.win_task_eps, **default(extra, {}),
    )
    return (row, ev_pi, ev_cem) if return_trajs else row

# shared training phases

def build_milestones(sea_phase_k: int, total_k: int, cycle_step_k: int) -> list[int]:
    milestones = [m * 1000 for m in range(sea_phase_k, total_k + 1, cycle_step_k)]
    if milestones[-1] < total_k * 1000:
        milestones.append(total_k * 1000)
    return milestones

def run_sea_pretrain(backend, tmp_dir: Path, device: torch.device, *, sea_phase_k: int, max_buf_len: int, max_wm_episodes: int, init_wm_steps: int, seed: int):
    comp, dim_eff, warm_obs, warm_act, comp_stats = fit_state_compressor_generic(backend, torch.device('cpu'), skill_len = backend.default_skill_len, dim_latent = 8, horizon = 4, warmup_rounds = 16, steps = 1000, seed = seed)
    wm_buf = WorldModelBuffer(tmp_dir / 'wm_unified', backend, max_episodes = max_wm_episodes, max_len = max_buf_len)
    wm_buf.add_batch(warm_obs, warm_act)
    agent = ModeConditionedSEA(backend, comp, dim_eff, warm_obs, warm_act, tmp_dir, tag = 'u', skill_len = backend.default_skill_len, max_sw = backend.default_max_sw, total_rounds = 90, dim_hidden = backend.default_dim_hidden, seed = seed)
    agent.train_explore_until(wm_buf, sea_phase_k * 1000)
    sea_contact_pct = round(wm_buf.contact_step_pct, 2)
    print(f'  [SEA Phase Done] steps={wm_buf.total_env_steps:,d} | WM buffer contact steps={sea_contact_pct:.2f}% | rounds={agent.rnd}', flush = True)

    wm = build_jepa_sigreg(backend.state_dim, backend.action_dim, dim_latent = backend.wm_dim_latent, dim_hidden = backend.wm_dim_hidden, reg_weight = 0.10).to(device)
    probe, wm_stats = update_world_model(backend, wm, wm_buf, device, train_horizon = 10, steps = init_wm_steps, batch_size = 256, lr_start = 2e-3, lr_end = 1e-4, seed = seed)
    print(f'  [SEA Init WM] {wm_stats}', flush = True)
    return agent, wm, probe, wm_buf, comp, comp_stats, sea_contact_pct

def run_unified_cycles(backend, agent: ModeConditionedSEA, wm: StateCompressor, probe: Module, wm_buf: WorldModelBuffer, test_push: dict, eval_init: np.ndarray, milestones: list[int], device: torch.device, t0: float, *, prefix: str = 'Unified', start_mode: str = 'docking', cem_eps_per_cycle: int = 48, task_eps_per_cycle: int = 48, sac_steps_per_cycle: int = 500, bc_every: int = 3, q_cem_weight: float = 5., cycle_wm_steps: int = 800, max_steps: int = 80, cem_horizon: int = 12, cem_pop: int = 256, cem_gens: int = 4, seed: int = 42, seed_offset: int = 10_000, on_milestone = None):
    row0 = evaluate_unified(backend, agent, wm, probe, test_push, eval_init, device, env_steps = wm_buf.total_env_steps, max_steps = max_steps, cem_horizon = cem_horizon, cem_gens = cem_gens, seed = seed, use_guide = False, q_cem_weight = 0., extra = dict(contact_pct = round(wm_buf.contact_step_pct, 2)))
    rows = [row0]
    log_row(backend, prefix, row0, time.perf_counter() - t0)
    if exists(on_milestone):
        on_milestone(rows, probe)

    total_steps, last_pi_ev, last_cem_ev = wm_buf.total_env_steps, None, None
    for cyc_i, target in enumerate(milestones[1:], start = 1):
        sub_i, up = 0, {}
        while total_steps < target:
            sub_i += 1
            if exists(agent.comp):
                total_steps += agent.store_explore(wm_buf, use_potential = True)

            use_guide = cyc_i >= 2 or sub_i >= 2
            cem_res = run_guided_cem(
                backend, wm, probe, agent if use_guide else None,
                backend.sample_starts(cem_eps_per_cycle, seed + seed_offset + cyc_i * 100 + sub_i * 7, start_mode = start_mode),
                num_envs = cem_eps_per_cycle, device = device, max_steps = max_steps,
                cem_horizon = cem_horizon, cem_pop = cem_pop, cem_gens = cem_gens, action_noise = 0.04,
                use_policy_guide = use_guide, q_cem_weight = q_cem_weight if (use_guide and agent.win_buf.size >= 128) else 0.,
                seed = seed + seed_offset + 1000 + cyc_i * 100 + sub_i,
            )
            agent.ingest_rollouts(wm_buf, cem_res, from_cem = True)
            total_steps += cem_res['env_steps']

            tsk_res = run_policy_task(
                backend, agent.sac, agent.encode_task_obs,
                backend.sample_starts(task_eps_per_cycle, seed + seed_offset + 2000 + cyc_i * 100 + sub_i * 11, start_mode = start_mode),
                num_envs = task_eps_per_cycle, max_steps = max_steps, det_frac = 0.25,
                seed = seed + seed_offset + 3000 + cyc_i * 100 + sub_i,
            )
            agent.ingest_rollouts(wm_buf, tsk_res, from_cem = False)
            total_steps += tsk_res['env_steps']
            up = agent.update_mixed(sac_steps = sac_steps_per_cycle, bc_every = bc_every, q_filter_bc = True, use_bc = True)

        probe, _ = update_world_model(backend, wm, wm_buf, device, train_horizon = 10, steps = cycle_wm_steps, batch_size = 256, lr_start = 1.2e-3, lr_end = 1e-4, seed = seed + cyc_i)
        is_last = cyc_i == len(milestones) - 1
        ev = evaluate_unified(backend, agent, wm, probe, test_push, eval_init, device, env_steps = total_steps, max_steps = max_steps, cem_horizon = cem_horizon, cem_gens = cem_gens, seed = seed, use_guide = True, q_cem_weight = q_cem_weight, extra = dict(contact_pct = round(wm_buf.contact_step_pct, 2), **up), return_trajs = is_last)
        row, last_pi_ev, last_cem_ev = ev if is_last else (ev, last_pi_ev, last_cem_ev)
        rows.append(row); log_row(backend, prefix, row, time.perf_counter() - t0)
        if exists(on_milestone):
            on_milestone(rows, probe)

    return rows, probe, last_pi_ev, last_cem_ev

def run_task_only_sac(backend, eval_init: np.ndarray, milestones: list[int], tmp_dir: Path, *, start_mode: str = 'docking', max_steps: int = 80, dim_hidden: int = 128, discount: float = 0.97, task_reward_scale: float = 10., seed: int = 42):
    sac = make_sac(backend.state_dim, backend.action_dim, discount = discount, lr = 1e-3, dim_hidden = dim_hidden)
    task_buf = TransitionBuffer(tmp_dir / 'base_task', backend.state_dim, backend.action_dim, max_transitions = 250_000)
    win_buf = TransitionBuffer(tmp_dir / 'base_win', backend.state_dim, backend.action_dim, max_transitions = 120_000)
    total_steps = total_wins = rnd = 0
    rows, last_ev = [], None

    for mi, target in enumerate(milestones):
        while total_steps < target:
            rnd += 1
            res = run_policy_task(backend, sac, lambda x: x, backend.sample_starts(32, seed + 9000 + rnd * 17, start_mode = start_mode), num_envs = 32, max_steps = max_steps, det_frac = 0.125, seed = seed + 9500 + rnd)
            total_steps += res['env_steps']
            add_episodes_to_buffer(task_buf, res['obs_seq'], res['act_seq'], res['rew_seq'], res['done_seq'], res['lengths'])
            if res['won'].any():
                w = res['won']; total_wins += int(w.sum().item())
                add_episodes_to_buffer(win_buf, res['obs_seq'][w], res['act_seq'][w], res['rew_seq'][w], res['done_seq'][w], res['lengths'][w])

            if task_buf.size >= 256:
                sac.train()
                for _ in range(96):
                    if win_buf.size > 0:
                        b1, b2 = task_buf.sample(128), win_buf.sample(128)
                        s, a, ns = cat((b1.state, b2.state)), cat((b1.action, b2.action)), cat((b1.next_state, b2.next_state))
                        r, d = cat((b1.reward, b2.reward)) * task_reward_scale, cat((b1.done, b2.done))
                    else:
                        b = task_buf.sample(256)
                        s, a, ns, r, d = b.state, b.action, b.next_state, b.reward * task_reward_scale, b.done
                    sac(states = s, cont_actions = a, discrete_actions = None, rewards = r, done = d, next_states = ns)

        is_last = mi == len(milestones) - 1
        ev = run_policy_task(backend, sac, lambda x: x, eval_init, num_envs = len(eval_init), max_steps = max_steps, deterministic = True, seed = seed, return_trajs = is_last)
        if is_last:
            last_ev = ev
        row = dict(env_steps = total_steps, train_wins = total_wins, **summarize_docking(ev))
        rows.append(row)
        print(f'  [Task-Only SAC] steps={total_steps:7,d} | wins={total_wins:4d} | {backend.win_label}={row["win_pct"]:5.1f}% >=.80={row["ge80_pct"]:5.1f}% cov={row["mean_cov"]:.3f} d={row["fin_dist"]:5.1f}{backend.dist_unit}', flush = True)
    return rows, last_ev

# plotting & logging

def plot_results(backend, out_path: Path, title: str, unified_rows: list[dict], *, baseline_sac_rows: list[dict] | None = None, sea_cem_rows: list[dict] | None = None, no_sea_rows: list[dict] | None = None, sea_phase_k: float | None = None):
    fig, axes = plt.subplots(2, 2, figsize = (13.6, 9.0), dpi = 150)
    fig.patch.set_facecolor('white')
    for ax in axes.ravel():
        if exists(sea_phase_k):
            ax.axvspan(0, sea_phase_k, color = '#dbeafe', alpha = 0.35); ax.axvline(sea_phase_k, color = '#3b82f6', ls = ':', lw = 1.2)
        ax.grid(True, alpha = 0.28); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlabel('Total Env Steps (k)', fontsize = 9.5)

    def plot_metric(ax, rows, key, *, scale = 1., **kw):
        if exists(rows):
            ax.plot([r['env_steps'] / 1000. for r in rows], [r[key] * scale for r in rows], **kw)

    ax = axes[0, 0]
    plot_metric(ax, baseline_sac_rows, 'win_pct', color = '#9ca3af', ls = '--', marker = 'x', lw = 2., label = f'Task-Only SAC (sparse {backend.win_label})')
    plot_metric(ax, no_sea_rows, 'cem_win_pct', color = '#dc2626', ls = ':', marker = 's', lw = 2., label = 'No-SEA (Rand) Guided CEM')
    plot_metric(ax, no_sea_rows, 'pi_win_pct', color = '#f97316', ls = ':', marker = 'v', lw = 1.8, label = 'No-SEA (Rand) pi(TASK)')
    plot_metric(ax, sea_cem_rows, 'win_pct', color = '#f59e0b', ls = '-.', marker = '^', lw = 2.2, label = 'SEA + WM + CEM Only')
    plot_metric(ax, unified_rows, 'pi_win_pct', color = '#059669', marker = 'o', lw = 2.6, label = 'Unified SEA SAC pi(a|s, TASK) [Zero CEM]')
    plot_metric(ax, unified_rows, 'cem_win_pct', color = '#2563eb', marker = 'D', lw = 2.4, label = 'Unified SEA Policy+Q-Guided CEM')
    ax.set_title(f'Exact Binary Sparse Win Rate (cov {backend.win_label})', fontweight = 'bold'); ax.set_ylabel('Win Rate (%)'); ax.set_ylim(-2, 102); ax.legend(fontsize = 7.8)

    ax = axes[0, 1]
    plot_metric(ax, baseline_sac_rows, 'mean_cov', scale = 100., color = '#9ca3af', ls = '--', marker = 'x', lw = 2., label = 'Task-Only SAC')
    plot_metric(ax, no_sea_rows, 'cem_mean_cov', scale = 100., color = '#dc2626', ls = ':', marker = 's', lw = 2., label = 'No-SEA Guided CEM')
    plot_metric(ax, no_sea_rows, 'pi_mean_cov', scale = 100., color = '#f97316', ls = ':', marker = 'v', lw = 1.8, label = 'No-SEA pi(TASK)')
    plot_metric(ax, sea_cem_rows, 'mean_cov', scale = 100., color = '#f59e0b', ls = '-.', marker = '^', lw = 2.2, label = 'SEA + WM + CEM Only')
    plot_metric(ax, unified_rows, 'pi_mean_cov', scale = 100., color = '#059669', marker = 'o', lw = 2.6, label = 'Unified SEA pi(TASK)')
    plot_metric(ax, unified_rows, 'cem_mean_cov', scale = 100., color = '#2563eb', marker = 'D', lw = 2.4, label = 'Unified SEA Guided CEM')
    ax.set_title('Mean Max Goal T-Block Coverage (%)', fontweight = 'bold'); ax.set_ylabel('Mean Max Coverage (%)'); ax.set_ylim(0, 102); ax.legend(fontsize = 7.8)

    ax = axes[1, 0]
    plot_metric(ax, unified_rows, 'pi_ge60_pct', color = '#0284c7', marker = 's', lw = 2.2, label = 'cov >= 0.60 (%)')
    plot_metric(ax, unified_rows, 'pi_ge80_pct', color = '#7c3aed', marker = '^', lw = 2.2, label = 'cov >= 0.80 (%)')
    plot_metric(ax, unified_rows, 'pi_win_pct', color = '#059669', marker = 'o', lw = 2.6, label = f'cov {backend.win_label} Exact Win (%)')
    ax.set_title('Reactive pi(a|s, mode=TASK) Precision Breakdown', fontweight = 'bold'); ax.set_ylabel('Success Rate (%)'); ax.set_ylim(-2, 102); ax.legend(fontsize = 8.5)

    ax = axes[1, 1]
    plot_metric(ax, unified_rows, 'h10_blk', color = '#2563eb', marker = 's', lw = 2.4, label = f'With-SEA WM H10 Err ({backend.dist_unit})')
    plot_metric(ax, no_sea_rows, 'h10_blk', color = '#dc2626', ls = ':', marker = 'x', lw = 2.2, label = f'No-SEA WM H10 Err ({backend.dist_unit})')
    ax2 = ax.twinx()
    plot_metric(ax2, unified_rows, 'exp_touch_pct', color = '#ea580c', ls = '--', marker = 'o', lw = 2., label = 'SEA EXPLORE touch (%)')
    ax2.set_ylabel('EXPLORE Touch (%)', color = '#ea580c', fontsize = 9.5); ax2.set_ylim(0, 105)
    ax.set_title('World Model H10 Contact Error & EXPLORE Retention', fontweight = 'bold'); ax.set_ylabel(f'H10 Block Err ({backend.dist_unit})')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize = 8., loc = 'best')

    fig.suptitle(title, fontsize = 13., fontweight = 'bold', y = 0.98)
    fig.tight_layout(rect = [0, 0, 1, 0.95]); fig.savefig(out_path); plt.close(fig)

def log_row(backend, prefix: str, row: dict, elapsed: float):
    print(
        f'  [{prefix}] steps={row["env_steps"]:7,d} ({elapsed:4.0f}s) | '
        f'pi(TASK): {backend.win_label}={row["pi_win_pct"]:5.1f}% >=.80={row["pi_ge80_pct"]:5.1f}% cov={row["pi_mean_cov"]:.3f} d={row["pi_fin_dist"]:4.1f}{backend.dist_unit} | '
        f'GuidedCEM: {backend.win_label}={row["cem_win_pct"]:5.1f}% >=.80={row["cem_ge80_pct"]:5.1f}% cov={row["cem_mean_cov"]:.3f} d={row["cem_fin_dist"]:4.1f}{backend.dist_unit} | '
        f'wins={row["win_cem_eps"]}/{row["win_task_eps"]} EXP={row["exp_touch_pct"]:4.1f}% H10={row["h10_blk"]:4.1f}{backend.dist_unit}',
        flush = True,
    )

# cli commands

def train(
    pusht3d: bool = False,
    env: str = 'pusht2d',
    x_jepa_dir: str = '../x-jepa',
    out_dir: str = '',
    baseline_json: str = '',
    sea_phase_k: int = 80,
    total_k: int = 400,
    cycle_step_k: int = 25,
    cem_eps_per_cycle: int = 48,
    task_eps_per_cycle: int = 48,
    sac_steps_per_cycle: int = 500,
    bc_every: int = 3,
    q_cem_weight: float = 5.,
    init_wm_steps: int = 1600,
    cycle_wm_steps: int = 800,
    max_steps: int = 80,
    cem_horizon: int = 12,
    cem_pop: int = 256,
    cem_gens: int = 4,
    start_mode: str = 'docking',
    eval_tasks: int = 32,
    step_scale: float = 40.,
    action_scale: tuple[float, float, float] = (0.45, 0.45, 0.05),
    num_workers: int = 8,
    envs_per_worker: int = 2,
    seed: int = 42,
):
    """Train Unified Mode-Conditioned SEA + WM + Policy+Q-Guided CEM + Binary-Sparse SAC + BC."""
    env, t0 = resolve_env(env, pusht3d), time.perf_counter()
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed)
    out_path = Path(out_dir or f'runs/sea_sparse_sac_{env}')
    out_path.mkdir(parents = True, exist_ok = True)
    tmp_dir, device = Path(tempfile.mkdtemp(prefix = f'sea_sparse_{env}_')), get_device()

    ref = json.loads(Path(baseline_json).read_text()) if (baseline_json and Path(baseline_json).exists()) else {}
    base_sac_rows, sea_cem_rows, no_sea_rows = ref.get('task_only_sac'), ref.get('sea_wm_cem_only'), ref.get('no_sea_unified')

    try:
        with make_backend(env, x_jepa_dir = x_jepa_dir, step_scale = step_scale, num_workers = num_workers, envs_per_worker = envs_per_worker, action_scale = action_scale) as backend:
            test_push = backend.build_heldout_push_tasks(num_tasks = 48 if env == 'pusht3d' else 64, rollout_len = 16, seed = 999)
            eval_init = backend.build_eval_tasks(num_tasks = eval_tasks, seed = 2025, start_mode = start_mode)
            milestones = build_milestones(sea_phase_k, total_k, cycle_step_k)
            max_buf_len = max(60, backend.default_skill_len + backend.default_max_sw, max_steps)

            print(f'\n=== 1. Unsupervised SEA Empowerment Phase ({env}, 0 -> {sea_phase_k}k, 0 task rewards) ===', flush = True)
            agent, wm, probe, wm_buf, comp, comp_stats, sea_contact_pct = run_sea_pretrain(backend, tmp_dir, device, sea_phase_k = sea_phase_k, max_buf_len = max_buf_len, max_wm_episodes = 24_000, init_wm_steps = init_wm_steps, seed = seed)

            best_score = (-1., -1.)

            def on_milestone(rows: list[dict], cur_probe: Module):
                nonlocal best_score
                plot_results(backend, out_path / 'training_curves.png', f'{env.upper()} ({start_mode}): Unified SEA + WM + Policy+Q-Guided CEM + Sparse SAC + BC (0 -> {total_k}k)', rows, baseline_sac_rows = base_sac_rows, sea_cem_rows = sea_cem_rows, no_sea_rows = no_sea_rows, sea_phase_k = float(sea_phase_k))
                ckpt = dict(env = env, start_mode = start_mode, env_steps = rows[-1]['env_steps'], comp = comp.state_dict(), sac = agent.sac.state_dict(), pot_sac = agent.pot_sac.state_dict(), classifier = agent.classifier.state_dict(), wm = wm.state_dict(), probe = cur_probe.state_dict())
                torch.save(ckpt, out_path / 'unified_agent.pt')
                score = (rows[-1]['cem_win_pct'], rows[-1]['cem_mean_cov'])
                if score > best_score:
                    best_score = score
                    torch.save(ckpt, out_path / 'best_unified_agent.pt')
                (out_path / 'results.json').write_text(json.dumps(dict(env = env, start_mode = start_mode, sea_contact_pct = sea_contact_pct, comp_stats = comp_stats, unified = rows, elapsed_sec = round(time.perf_counter() - t0, 1)), indent = 2))

            print(f'\n=== 2. Unified SEA + WM + Policy+Q-Guided CEM + Binary-Sparse SAC + BC ({sea_phase_k}k -> {total_k}k) ===', flush = True)
            _, _, last_pi_ev, last_cem_ev = run_unified_cycles(
                backend, agent, wm, probe, wm_buf, test_push, eval_init, milestones, device, t0,
                prefix = 'Unified', start_mode = start_mode, cem_eps_per_cycle = cem_eps_per_cycle, task_eps_per_cycle = task_eps_per_cycle,
                sac_steps_per_cycle = sac_steps_per_cycle, bc_every = bc_every, q_cem_weight = q_cem_weight, cycle_wm_steps = cycle_wm_steps,
                max_steps = max_steps, cem_horizon = cem_horizon, cem_pop = cem_pop, cem_gens = cem_gens, seed = seed, on_milestone = on_milestone,
            )

            if exists(last_pi_ev):
                backend.render_showcase(eval_init, last_pi_ev, f'{env.upper()} Unified SEA SAC pi(a|s, TASK) at {total_k}k [Zero CEM]', out_path, 'pi_task', seed = seed)
            if exists(last_cem_ev):
                backend.render_showcase(eval_init, last_cem_ev, f'{env.upper()} Unified SEA Policy+Q-Guided CEM at {total_k}k', out_path, 'guided_cem', seed = seed)
            if env == 'pusht3d' and exists(last_cem_ev) and exists(last_pi_ev):
                pick4 = pick_indices(last_cem_ev['max_cov'], count = 4, max_quantile = 0.80)
                backend.render_side_by_side(eval_init, [('Unified pi(TASK) [Zero CEM]', last_pi_ev), ('Unified Policy+Q-Guided CEM', last_cem_ev)], pick4, f'PushT3D Unified at {total_k}k ({start_mode})', out_path, 'pi_vs_cem', seed = seed)
                backend.render_explore_showcase(agent, f'PushT3D Unsupervised SEA Skills at {total_k}k', out_path, 'explore_skills', seed = 777)

            print(f'\nSaved curves, checkpoints & videos to {out_path} in {time.perf_counter() - t0:.1f}s', flush = True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors = True)

def compare(
    pusht3d: bool = False,
    env: str = 'pusht2d',
    x_jepa_dir: str = '../x-jepa',
    out_dir: str = '',
    sea_phase_k: int = 80,
    total_k: int = 240,
    cycle_step_k: int = 20,
    cem_eps_per_cycle: int = 48,
    task_eps_per_cycle: int = 48,
    sac_steps_per_cycle: int = 600,
    bc_every: int = 3,
    q_cem_weight: float = 5.,
    init_wm_steps: int = 1800,
    cycle_wm_steps: int = 900,
    max_steps: int = 80,
    cem_horizon: int = 12,
    cem_pop: int = 256,
    cem_gens: int = 4,
    start_mode: str = 'docking',
    eval_tasks: int = 32,
    step_scale: float = 40.,
    action_scale: tuple[float, float, float] = (0.45, 0.45, 0.05),
    num_workers: int = 8,
    envs_per_worker: int = 2,
    include_no_sea: bool = True,
    seed: int = 42,
):
    """From-scratch comparison: Task-Only SAC vs SEA+WM+CEM vs No-SEA vs Unified SEA."""
    env, t0 = resolve_env(env, pusht3d), time.perf_counter()
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed)
    out_path = Path(out_dir or (f'runs/sea_sparse_sac_compare_{env}' if env != 'pusht2d' else 'runs/sea_sparse_sac_compare'))
    out_path.mkdir(parents = True, exist_ok = True)
    tmp_dir, device = Path(tempfile.mkdtemp(prefix = f'sea_sparse_cmp_{env}_')), get_device()

    try:
        with make_backend(env, x_jepa_dir = x_jepa_dir, step_scale = step_scale, num_workers = num_workers, envs_per_worker = envs_per_worker, action_scale = action_scale) as backend:
            test_push = backend.build_heldout_push_tasks(num_tasks = 48 if env == 'pusht3d' else 64, rollout_len = 16, seed = 999)
            eval_init = backend.build_eval_tasks(num_tasks = eval_tasks, seed = 2025, start_mode = start_mode)
            milestones = build_milestones(sea_phase_k, total_k, cycle_step_k)
            max_buf_len = max(60, backend.default_skill_len + backend.default_max_sw, max_steps)

            print(f'\n=== 1. Baseline: Task-Only SAC ({env}, start={start_mode}, binary sparse {backend.win_label}, 0 -> {total_k}k) ===', flush = True)
            base_milestones = ([min(40_000, milestones[0] // 2)] if milestones[0] >= 40_000 else []) + milestones
            base_sac_rows, base_sac_ev = run_task_only_sac(backend, eval_init, base_milestones, tmp_dir, start_mode = start_mode, max_steps = max_steps, dim_hidden = backend.default_dim_hidden, seed = seed)

            print(f'\n=== 2. Unsupervised SEA Empowerment Phase ({env}, 0 -> {sea_phase_k}k, 0 task rewards) ===', flush = True)
            agent, wm_u, pr_u, wm_buf_u, comp, comp_stats, sea_contact_pct = run_sea_pretrain(backend, tmp_dir, device, sea_phase_k = sea_phase_k, max_buf_len = max_buf_len, max_wm_episodes = 12_000, init_wm_steps = init_wm_steps, seed = seed)

            wm_buf_cem = wm_buf_u.clone_to(tmp_dir / 'wm_sea_cem_only')
            wm_cem = build_jepa_sigreg(backend.state_dim, backend.action_dim, dim_latent = backend.wm_dim_latent, dim_hidden = backend.wm_dim_hidden, reg_weight = 0.10).to(device)
            wm_cem.load_state_dict(clone_sd(wm_u))
            pr_cem = build_probe(backend.wm_dim_latent, backend.state_dim).to(device)
            pr_cem.load_state_dict(clone_sd(pr_u)); pr_cem.eval()

            print(f'\n=== 3. Unified Mode-Conditioned SEA + WM + CEM + Sparse-Task SAC + BC ({sea_phase_k}k -> {total_k}k) ===', flush = True)
            unified_rows, pr_u, last_pi_ev, last_cem_ev = run_unified_cycles(
                backend, agent, wm_u, pr_u, wm_buf_u, test_push, eval_init, milestones, device, t0,
                prefix = 'Unified', start_mode = start_mode, cem_eps_per_cycle = cem_eps_per_cycle, task_eps_per_cycle = task_eps_per_cycle,
                sac_steps_per_cycle = sac_steps_per_cycle, bc_every = bc_every, q_cem_weight = q_cem_weight, cycle_wm_steps = cycle_wm_steps,
                max_steps = max_steps, cem_horizon = cem_horizon, cem_pop = cem_pop, cem_gens = cem_gens, seed = seed, seed_offset = 10_000,
            )

            row0 = unified_rows[0]
            sea_cem_rows = [dict(env_steps = wm_buf_cem.total_env_steps, h10_blk = row0['h10_blk'], h10_rot = row0['h10_rot'], win_pct = row0['cem_win_pct'], ge80_pct = row0['cem_ge80_pct'], ge60_pct = row0['cem_ge60_pct'], mean_cov = row0['cem_mean_cov'], fin_dist = row0['cem_fin_dist'], win_steps = row0['cem_win_steps'])]

            print(f'\n=== 4. Baseline: SEA + WM + CEM Only ({sea_phase_k}k -> {total_k}k) ===', flush = True)
            total_c_steps, last_sea_cem_ev = wm_buf_cem.total_env_steps, None
            for cyc_i, target in enumerate(milestones[1:], start = 1):
                sub_i = 0
                while total_c_steps < target:
                    sub_i += 1
                    cem_res = run_guided_cem(backend, wm_cem, pr_cem, None, backend.sample_starts(64, seed + 20_000 + cyc_i * 100 + sub_i * 7, start_mode = start_mode), num_envs = 64, device = device, max_steps = max_steps, cem_horizon = cem_horizon, cem_pop = cem_pop, cem_gens = cem_gens, action_noise = 0.08, seed = seed + 21_000 + cyc_i * 100 + sub_i)
                    add_episodes_to_wm(wm_buf_cem, cem_res['obs_seq'], cem_res['act_seq'], cem_res['lengths'])
                    total_c_steps += cem_res['env_steps']

                pr_cem, _ = update_world_model(backend, wm_cem, wm_buf_cem, device, train_horizon = 10, steps = cycle_wm_steps, batch_size = 256, lr_start = 1.2e-3, lr_end = 1e-4, seed = seed + cyc_i)
                is_last = cyc_i == len(milestones) - 1
                ev_c = run_guided_cem(backend, wm_cem, pr_cem, None, eval_init, num_envs = len(eval_init), device = device, max_steps = max_steps, cem_horizon = cem_horizon, cem_pop = 384, cem_gens = cem_gens, seed = seed, return_trajs = is_last)
                if is_last:
                    last_sea_cem_ev = ev_c
                h10_blk_c, h10_rot_c = eval_h10_error(backend, wm_cem, pr_cem, test_push, device)
                row_c = dict(env_steps = total_c_steps, h10_blk = h10_blk_c, h10_rot = h10_rot_c, **summarize_docking(ev_c))
                sea_cem_rows.append(row_c)
                print(f'  [SEA+WM+CEM] steps={total_c_steps:7,d} | {backend.win_label}={row_c["win_pct"]:5.1f}% >=.80={row_c["ge80_pct"]:5.1f}% cov={row_c["mean_cov"]:.3f} d={row_c["fin_dist"]:4.1f}{backend.dist_unit} | H10={row_c["h10_blk"]:4.1f}{backend.dist_unit}', flush = True)

            no_sea_rows = last_ns_cem_ev = no_sea_contact_pct = None
            if include_no_sea:
                print(f'\n=== 5. Ablation: No-SEA (Random 0->{sea_phase_k}k -> WM + Guided CEM + Sparse-Task SAC + BC) ===', flush = True)
                wm_buf_ns = WorldModelBuffer(tmp_dir / 'wm_no_sea', backend, max_episodes = 12_000, max_len = max_buf_len)
                backend.collect_random_steps(wm_buf_ns, sea_phase_k * 1000, ep_len = 60, seed = seed + 300)
                no_sea_contact_pct = round(wm_buf_ns.contact_step_pct, 2)
                print(f'  [No-SEA Random Phase] steps={wm_buf_ns.total_env_steps:,d} | WM buffer contact steps={no_sea_contact_pct:.2f}%', flush = True)

                wm_ns = build_jepa_sigreg(backend.state_dim, backend.action_dim, dim_latent = backend.wm_dim_latent, dim_hidden = backend.wm_dim_hidden, reg_weight = 0.10).to(device)
                pr_ns, wm_ns_stats = update_world_model(backend, wm_ns, wm_buf_ns, device, train_horizon = 10, steps = init_wm_steps, batch_size = 256, lr_start = 2e-3, lr_end = 1e-4, seed = seed)
                print(f'  [No-SEA Init WM] {wm_ns_stats}', flush = True)

                agent_ns = ModeConditionedSEA(backend, None, 0, None, None, tmp_dir, tag = 'ns', dim_hidden = backend.default_dim_hidden, seed = seed)
                no_sea_rows, _, _, last_ns_cem_ev = run_unified_cycles(
                    backend, agent_ns, wm_ns, pr_ns, wm_buf_ns, test_push, eval_init, milestones, device, t0,
                    prefix = 'No-SEA ', start_mode = start_mode, cem_eps_per_cycle = cem_eps_per_cycle, task_eps_per_cycle = task_eps_per_cycle,
                    sac_steps_per_cycle = sac_steps_per_cycle, bc_every = bc_every, q_cem_weight = q_cem_weight, cycle_wm_steps = cycle_wm_steps,
                    max_steps = max_steps, cem_horizon = cem_horizon, cem_pop = cem_pop, cem_gens = cem_gens, seed = seed, seed_offset = 30_000,
                )

            plot_results(backend, out_path / 'sparse_sac_comparison.png', f'{env.upper()} ({start_mode}): Mode-Conditioned SEA + WM + CEM + Binary-Sparse SAC + BC (0 -> {total_k}k)', unified_rows, baseline_sac_rows = base_sac_rows, sea_cem_rows = sea_cem_rows, no_sea_rows = no_sea_rows, sea_phase_k = float(sea_phase_k))
            if exists(last_pi_ev):
                backend.render_showcase(eval_init, last_pi_ev, f'{env.upper()} Unified SEA SAC pi(a|s, TASK) at {total_k}k [Zero CEM]', out_path, 'pi_task', seed = seed)
            if exists(last_cem_ev):
                backend.render_showcase(eval_init, last_cem_ev, f'{env.upper()} Unified SEA Policy+Q-Guided CEM at {total_k}k', out_path, 'guided_cem', seed = seed)

            if env == 'pusht3d' and exists(last_cem_ev):
                method_list = [(l, ev) for l, ev in (('Task-Only SAC', base_sac_ev), ('No-SEA Guided CEM', last_ns_cem_ev), ('SEA+WM+CEM Only', last_sea_cem_ev), ('Unified pi(TASK)', last_pi_ev), ('Unified Guided CEM', last_cem_ev)) if exists(ev)]
                backend.render_side_by_side(eval_init, method_list, pick_indices(last_cem_ev['max_cov'], count = 4, max_quantile = 0.75), f'PushT3D Side-by-Side at {total_k}k ({start_mode})', out_path, 'side_by_side', seed = seed)
                backend.render_explore_showcase(agent, f'PushT3D Unsupervised SEA Skills at {total_k}k (Zero Task Rewards)', out_path, 'explore_skills', seed = 777)

            torch.save(dict(env = env, start_mode = start_mode, comp = comp.state_dict(), sac = agent.sac.state_dict(), pot_sac = agent.pot_sac.state_dict(), classifier = agent.classifier.state_dict(), wm = wm_u.state_dict(), probe = pr_u.state_dict()), out_path / 'unified_agent.pt')
            (out_path / 'results.json').write_text(json.dumps(dict(env = env, start_mode = start_mode, sea_contact_pct = sea_contact_pct, no_sea_contact_pct = no_sea_contact_pct, comp_stats = comp_stats, task_only_sac = base_sac_rows, sea_wm_cem_only = sea_cem_rows, no_sea_unified = no_sea_rows, unified = unified_rows, elapsed_sec = round(time.perf_counter() - t0, 1)), indent = 2))
            print(f'\nSaved from-scratch comparison, curves & videos to {out_path} in {time.perf_counter() - t0:.1f}s', flush = True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors = True)

def pilot(
    sea_ckpt: str = 'runs/see_pusht_sac_state/checkpoint.pt',
    wm_ckpt: str = 'runs/sea_wm_600k/wm_600k.pt',
    x_jepa_dir: str = '../x-jepa',
    out_dir: str = 'runs/sea_sparse_sac_pilot',
    cycles: int = 12,
    cem_eps_per_cycle: int = 48,
    task_eps_per_cycle: int = 48,
    sac_steps_per_cycle: int = 600,
    bc_every: int = 3,
    use_bc: bool = True,
    q_filter_bc: bool = True,
    q_cem_weight: float = 5.,
    wm_steps_per_cycle: int = 300,
    max_steps: int = 80,
    cem_horizon: int = 12,
    cem_pop: int = 256,
    cem_gens: int = 4,
    start_mode: str = 'docking',
    step_scale: float = 40.,
    seed: int = 42,
):
    """2D PushT pilot starting from an existing unsupervised SEA + world model checkpoint."""
    from train_sea_world_model import collect_sea_skill_steps, load_sea_checkpoint

    t0 = time.perf_counter()
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed)
    out_path = Path(out_dir); out_path.mkdir(parents = True, exist_ok = True)
    tmp_dir, device = Path(tempfile.mkdtemp(prefix = 'sea_sparse_pilot_')), get_device()

    try:
        with make_backend('pusht2d', x_jepa_dir = x_jepa_dir, step_scale = step_scale) as backend:
            test_push = backend.build_heldout_push_tasks(num_tasks = 64, rollout_len = 16, seed = 999)
            dock_init = backend.build_eval_tasks(num_tasks = 32, seed = 2025, start_mode = start_mode)

            sea_agent = load_sea_checkpoint(sea_ckpt)
            comp, ck_pot, _, dim_eff, _ = sea_agent
            wm_ck = torch.load(wm_ckpt, map_location = 'cpu', weights_only = False)
            wm = build_jepa_sigreg(backend.state_dim, backend.action_dim, dim_latent = 16, reg_weight = 0.10).to(device)
            wm.load_state_dict(wm_ck['wm']); wm.eval()
            probe = build_probe(16, backend.state_dim).to(device)
            probe.load_state_dict(wm_ck['probe']); probe.eval()

            wm_buf = WorldModelBuffer(tmp_dir / 'wm_buf', backend, max_episodes = 8000, max_len = max(60, max_steps))
            collect_sea_skill_steps(backend.fp, sea_agent, wm_buf, 60_000, ep_len = 60, action_noise = 0.15, step_scale = step_scale, seed = seed + 7)

            agent = ModeConditionedSEA(backend, comp, dim_eff, None, None, tmp_dir, dim_hidden = 128, seed = seed)
            agent.pot_sac.load_state_dict(ck_pot.state_dict())
            for _ in range(6):
                agent.store_explore(wm_buf, use_potential = True)

            extra_steps = 0
            rows = [evaluate_unified(backend, agent, wm, probe, test_push, dock_init, device, env_steps = 0, max_steps = max_steps, cem_horizon = cem_horizon, cem_gens = cem_gens, seed = seed, use_guide = False, q_cem_weight = 0.)]
            print(f'\n=== Pilot: Mode-Conditioned SEA + WM + CEM + Binary-Sparse SAC (use_bc={use_bc}, start_mode={start_mode}) ===', flush = True)
            log_row(backend, 'cyc= 0', rows[0], time.perf_counter() - t0)

            last_pi_ev = last_cem_ev = None
            for cyc in range(1, cycles + 1):
                extra_steps += agent.store_explore(wm_buf, use_potential = True)
                use_guide = cyc >= 2
                cem_res = run_guided_cem(backend, wm, probe, agent if use_guide else None, backend.sample_starts(cem_eps_per_cycle, seed + 1000 + cyc * 37, start_mode = start_mode), num_envs = cem_eps_per_cycle, device = device, max_steps = max_steps, cem_horizon = cem_horizon, cem_pop = cem_pop, cem_gens = cem_gens, action_noise = 0.04, use_policy_guide = use_guide, q_cem_weight = q_cem_weight if (use_guide and agent.win_buf.size >= 128) else 0., seed = seed + 2000 + cyc)
                agent.ingest_rollouts(wm_buf, cem_res, from_cem = True); extra_steps += cem_res['env_steps']

                tsk_res = run_policy_task(backend, agent.sac, agent.encode_task_obs, backend.sample_starts(task_eps_per_cycle, seed + 3000 + cyc * 41, start_mode = start_mode), num_envs = task_eps_per_cycle, max_steps = max_steps, det_frac = 0.25, seed = seed + 4000 + cyc)
                agent.ingest_rollouts(wm_buf, tsk_res, from_cem = False); extra_steps += tsk_res['env_steps']

                up = agent.update_mixed(sac_steps = sac_steps_per_cycle, bc_every = bc_every, q_filter_bc = q_filter_bc, use_bc = use_bc)
                if wm_steps_per_cycle > 0 and wm_buf.buf.num_episodes >= 64:
                    probe, _ = update_world_model(backend, wm, wm_buf, device, train_horizon = 10, steps = wm_steps_per_cycle, batch_size = 256, lr_start = 3e-4, lr_end = 5e-5, seed = seed + cyc)

                is_last = cyc == cycles
                ev = evaluate_unified(backend, agent, wm, probe, test_push, dock_init, device, env_steps = extra_steps, max_steps = max_steps, cem_horizon = cem_horizon, cem_gens = cem_gens, seed = seed, use_guide = True, q_cem_weight = q_cem_weight, extra = up, return_trajs = is_last)
                row, last_pi_ev, last_cem_ev = ev if is_last else (ev, last_pi_ev, last_cem_ev)
                rows.append(row); log_row(backend, f'cyc={cyc:2d}', row, time.perf_counter() - t0)

            plot_results(backend, out_path / 'pilot_curves.png', 'Pilot: Mode-Conditioned SEA + WM + CEM + Binary-Sparse SAC (+ BC)', rows)
            if exists(last_pi_ev):
                backend.render_showcase(dock_init, last_pi_ev, 'Unified SAC pi(a|s, mode=TASK) [Zero CEM]', out_path, 'pilot_pi_task', seed = seed)
            if exists(last_cem_ev):
                backend.render_showcase(dock_init, last_cem_ev, 'Unified Policy+Q-Guided CEM (Binary Sparse)', out_path, 'pilot_guided_cem', seed = seed)
            (out_path / 'pilot.json').write_text(json.dumps(dict(rows = rows, elapsed_sec = round(time.perf_counter() - t0, 1)), indent = 2))
            print(f'\nSaved pilot curves & videos to {out_path} in {time.perf_counter() - t0:.1f}s', flush = True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors = True)

if __name__ == '__main__':
    commands = dict(train = train, compare = compare, pilot = pilot)
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        Fire(commands)
    else:
        Fire(train)
