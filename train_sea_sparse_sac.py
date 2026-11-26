from __future__ import annotations

# Mode-Conditioned Unified SAC + SEA Empowerment + JEPA+SIGReg World Model + CEM + BC on Faster PushT:
#   - single unified SAC-pytorch agent conditioned on mode in {EXPLORE [1, 0], TASK [0, 1]}
#       * EXPLORE: SEA empowerment skills z ~ S^{k-1}, intrinsic skill reward r_eff, done = False
#       * TASK:    pure binary sparse reward r = 1{coverage > 0.95}, done = (coverage > 0.95)
#   - cycled with JEPA+SIGReg world model (StateCompressor) + Policy+Q-Guided CEM (CEMPlanner)
#   - occasional Q-filtered behavior cloning from the best CEM planning rollouts into pi(a | s, TASK)
#
# Run pilot (from a trained unsupervised SEA + WM checkpoint):
#   uv run --with "sac-pytorch>=0.3.7" --with "memmap-replay-buffer>=0.3.0" --with fire --with pillow --with matplotlib --with "imageio[ffmpeg]" \
#       python train_sea_sparse_sac.py pilot
#
# Run from-scratch comparison (Task-Only SAC vs SEA+WM+CEM vs Unified Mode-Conditioned SEA+WM+CEM+Sparse-SAC+BC):
#   uv run --with "sac-pytorch>=0.3.7" --with "memmap-replay-buffer>=0.3.0" --with fire --with pillow --with matplotlib --with "imageio[ffmpeg]" \
#       python train_sea_sparse_sac.py compare

from functools import partial
import json
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
import torch.nn.functional as F

from memmap_replay_buffer import ReplayBuffer
from SAC_pytorch import SAC
from SEA_pytorch import (
    CEMPlanner,
    SkillClassifier,
    StateCompressor,
    estimate_empowerment,
    sample_skills,
    sample_truncated_geometric,
)
from train_sea_pusht import (
    ACTION_DIM,
    RawPushTWrapper,
    ReZeroMLP,
    act_sac,
    add_episodes,
    encode_policy_state,
    fit_state_compressor,
    load_faster_pusht,
    make_sac,
)
from train_sea_world_model import (
    GOAL_COM,
    GOAL_TH,
    STATE_DIM,
    WorldModelBuffer,
    build_docking_tasks,
    build_heldout_push_tasks,
    build_jepa_sigreg,
    collect_sea_skill_steps,
    load_sea_checkpoint,
    render_trajectory_showcase,
    update_world_model,
)

WIN_COV = 0.95

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clone_sd(module: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

def step_opt(opt, loss: Tensor, params = None, max_norm: float | None = None):
    opt.zero_grad()
    loss.backward()
    if exists(params) and exists(max_norm):
        nn.utils.clip_grad_norm_(params, max_norm)
    opt.step()
    opt.zero_grad()

def build_probe(dim_latent: int = 16) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim_latent, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, STATE_DIM))

def sample_starts(num_tasks: int, seed: int, *, start_mode: str = 'docking') -> np.ndarray | None:
    return None if start_mode == 'native' else build_docking_tasks(num_tasks = num_tasks, seed = seed)

# mode conditioning & scoring

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

def docking_probe_score(probe: nn.Module, z_seq: Tensor, goal_com_norm: Tensor) -> Tensor:
    s = probe(z_seq)
    dist = (s[..., 2:4] - goal_com_norm).norm(dim = -1) * 256.
    th = torch.atan2(s[..., 5], s[..., 4])
    yaw = torch.atan2(torch.sin(th - GOAL_TH), torch.cos(th - GOAL_TH)).abs()
    step_score = -dist - 60. * yaw
    return 0.7 * step_score[..., -1] + 0.3 * einx.mean('... [h]', step_score)

@torch.no_grad()
def eval_h10_error(wm: StateCompressor, probe: nn.Module, test_push: dict, device: torch.device) -> float:
    obs_te, act_te = test_push['obs'], test_push['act']
    pred = probe(wm.rollout(wm.encode_detached(obs_te[:, 0].to(device)), act_te.to(device))).cpu()
    return round(float((pred[:, 9, 2:4] - obs_te[:, 10, 2:4]).norm(dim = -1).mean() * 256.), 1)

def summarize_docking(res: dict) -> dict:
    cov, won = res['max_cov'], res['won']
    win_lens = res['lengths'][won].float()
    return dict(
        win95_pct = round(float(won.float().mean() * 100.), 1),
        ge90_pct = round(float((cov >= 0.90).float().mean() * 100.), 1),
        ge80_pct = round(float((cov >= 0.80).float().mean() * 100.), 1),
        mean_cov = round(float(cov.mean()), 3),
        win_steps = round(float(win_lens.mean()), 1) if win_lens.numel() > 0 else 0.,
    )

# flat off-policy transition buffer

class TransitionBuffer:
    def __init__(self, path: Path, dim_state: int, *, max_transitions: int = 250_000, store_reward_done: bool = True):
        self.store_reward_done = store_reward_done
        fields = dict(state = ('float', dim_state), action = ('float', ACTION_DIM))
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
        state_seq[:, :-1][mask],
        act_seq[mask],
        state_seq[:, 1:][mask] if buf.store_reward_done else None,
        rew_seq[mask] if buf.store_reward_done else None,
        done_seq[mask] if buf.store_reward_done else None,
    )

# unified mode-conditioned SEA + sparse-task SAC agent

class ModeConditionedSEA:
    def __init__(
        self,
        fp_mod,
        comp: StateCompressor,
        dim_eff: int,
        warm_obs: Tensor | None,
        warm_act: Tensor | None,
        tmp_dir: Path,
        *,
        num_envs: int = 16,
        skill_len: int = 45,
        max_sw: int = 15,
        step_scale: float = 40.,
        total_rounds: int = 120,
        pot_start_round: int = 10,
        dim_hidden: int = 128,
        discount: float = 0.97,
        task_reward_scale: float = 10.,
        bc_lr: float = 1e-3,
        seed: int = 42,
    ):
        self.fp, self.comp, self.dim_eff, self.dim_skill = fp_mod, comp, dim_eff, dim_eff
        self.dim_state = STATE_DIM + dim_eff
        self.dim_unified = self.dim_state + 2 * dim_eff + self.dim_skill + 2
        self.skill_len, self.max_sw, self.step_scale = skill_len, max_sw, step_scale
        self.total_rounds, self.pot_start_round, self.discount = total_rounds, pot_start_round, discount
        self.task_reward_scale = float(task_reward_scale)
        self.comp_dev = torch.device('cpu')
        self.rnd = self.win_cem_eps = self.win_task_eps = 0
        self._seeded = False

        self.wrapper = RawPushTWrapper(fp_mod, num_envs, step_scale = step_scale, seed = seed + 500)
        self.classifier = SkillClassifier(
            ReZeroMLP(dim_eff, 64, self.dim_skill, depth = 1, orthogonal_skip_gain = 0.5),
            total_steps = total_rounds, anneal_fracs = (0.35, 0.90), temperature = 0.10, noise_std = 0.05,
            num_samples = 512, sigreg_weight = 0.03, sigreg_scale = 0.20, extract_obs = lambda s: s[..., -dim_eff:],
        )
        self.opt_clf = torch.optim.Adam(self.classifier.parameters(), lr = 3e-4)

        self.sac = make_sac(self.dim_unified, discount = discount, lr = 1e-3, dim_hidden = dim_hidden, mlp_depth = 2, use_beta = False, init_alpha = 0.05, tau_ema = 0.02, actor_update_freq = 2)
        self.pot_sac = make_sac(self.dim_state, discount = 0.95, lr = 1e-3, dim_hidden = 64, mlp_depth = 2, use_beta = False, init_alpha = 0.05, tau_ema = 0.02, actor_update_freq = 1)
        self.bc_opt = torch.optim.Adam(self.sac.actor.parameters(), lr = bc_lr)

        fields = dict(state = ('float', self.dim_state), action = ('float', ACTION_DIM), next_state = ('float', self.dim_state))
        self.skill_buf = ReplayBuffer(
            tmp_dir / 'u_skill', max_episodes = 4096, max_timesteps = skill_len, fields = fields,
            meta_fields = dict(anchor = ('float', self.dim_state), skill = ('float', self.dim_skill)),
            circular = True, overwrite = True, track_write_step = True, flush_every_store_step = skill_len,
        )
        max_pot_t = max(60, skill_len + max_sw, int(warm_act.shape[1]) if exists(warm_act) else 0)
        self.pot_buf = ReplayBuffer(tmp_dir / 'u_pot', max_episodes = 4096, max_timesteps = max_pot_t, fields = fields, circular = True, overwrite = True, flush_every_store_step = max_pot_t)
        if exists(warm_obs) and exists(warm_act):
            add_episodes(self.pot_buf, self.encode_raw(warm_obs), warm_act)

        self.task_buf = TransitionBuffer(tmp_dir / 'u_task', self.dim_state, max_transitions = 250_000)
        self.win_buf = TransitionBuffer(tmp_dir / 'u_win', self.dim_state, max_transitions = 120_000)
        self.bc_win_buf = TransitionBuffer(tmp_dir / 'u_bc_win', self.dim_state, max_transitions = 80_000, store_reward_done = False)
        self.bc_plan_buf = TransitionBuffer(tmp_dir / 'u_bc_plan', self.dim_state, max_transitions = 40_000, store_reward_done = False)

    @torch.no_grad()
    def encode_raw(self, raw_obs: Tensor, *, device: torch.device | None = None) -> Tensor:
        raw_d = raw_obs.to(default(device, self.comp_dev))
        h = self.comp.encode_detached(raw_d.reshape(-1, STATE_DIM)).reshape(*raw_d.shape[:-1], self.dim_eff)
        return cat((raw_d, h), dim = -1)

    @torch.no_grad()
    def encode_task_obs(self, raw_obs: Tensor, *, device: torch.device | None = None) -> Tensor:
        return task_obs(self.encode_raw(raw_obs, device = device), self.dim_eff, self.dim_skill)

    @torch.no_grad()
    def collect_explore(self, *, use_potential: bool):
        n = self.wrapper.num_envs
        self.wrapper.reset()
        s = encode_policy_state(self.wrapper, self.comp, 'state', self.comp_dev)
        z = sample_skills(n, self.dim_skill)

        if not use_potential or self.max_sw <= 0:
            switch_t, total_steps = torch.zeros(n, dtype = torch.long), self.skill_len
        else:
            switch_t = randint(1, self.max_sw + 1, (n,))
            switch_t[:int(round(n * 0.25))] = 0
            total_steps = self.max_sw + self.skill_len

        all_s, all_a = torch.empty(n, total_steps + 1, self.dim_state), torch.empty(n, total_steps, ACTION_DIM)
        all_s[:, 0], s0 = s, s.clone()
        for t in range(total_steps):
            at_sw = switch_t == t
            if at_sw.any():
                s0 = einx.where('b, b d, b d -> b d', at_sw, s, s0)
            a = act_sac(self.sac, explore_obs(s, s0, z, self.dim_eff))
            if use_potential and self.max_sw > 0:
                a = einx.where('b, b d, b d -> b d', t >= switch_t, a, act_sac(self.pot_sac, s))
            self.wrapper.step(a)
            s = encode_policy_state(self.wrapper, self.comp, 'state', self.comp_dev)
            all_a[:, t], all_s[:, t + 1] = a, s

        t_idx = rearrange(switch_t, 'b -> b 1') + rearrange(arange(self.skill_len + 1), 't -> 1 t')
        env_idx = rearrange(arange(n), 'b -> b 1')
        return all_s[env_idx, t_idx], all_a[env_idx, t_idx[:, :-1]], z, all_s, all_a

    def store_explore(self, wm_buf: WorldModelBuffer, *, use_potential: bool) -> int:
        ss, sa, z, all_s, all_a = self.collect_explore(use_potential = use_potential)
        add_episodes(self.skill_buf, ss, sa, anchor = ss[:, 0], skill = z)
        add_episodes(self.pot_buf, all_s, all_a)
        wm_buf.add_batch(all_s[:, :, :STATE_DIM], all_a)
        return int(all_a.shape[0] * all_a.shape[1])

    def _update_classifier(self, steps: int, batch_size: int):
        for _ in range(steps):
            cb = self.skill_buf.sample(
                batch_size, horizon = self.skill_len, max_age = 256 * self.skill_len, fallback_on_empty = True,
                replacement = min(self.skill_buf.num_episodes, batch_size) < batch_size,
                fields = ('next_state',), to_named_tuple = ('anchor', 'next_state', 'skill'),
            )
            lag = sample_truncated_geometric(torch.full((batch_size,), self.skill_len, dtype = torch.long), 0.96) - 1
            loss = self.classifier(cb.anchor, cb.next_state[arange(batch_size), lag], cb.skill)
            step_opt(self.opt_clf, loss, self.classifier.parameters(), max_norm = 10.)

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
        return task_obs(s, self.dim_eff, self.dim_skill), a, r, d, task_obs(ns, self.dim_eff, self.dim_skill)

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

    def update_mixed(self, *, sac_steps: int = 600, bc_every: int = 3, batch_size: int = 256, explore_frac: float = 0.30, min_win_eps_for_pure_bc: int = 64, q_filter_bc: bool = True, use_bc: bool = True) -> dict:
        self.sac.train(); self.pot_sac.train(); self.classifier.train()
        assert self.task_buf.size >= 32
        has_explore = self.skill_buf.num_episodes >= 8
        if has_explore:
            self._update_classifier(steps = 6, batch_size = batch_size)

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
                    obs_bc = task_obs(s_bc, self.dim_eff, self.dim_skill)
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
            q_task_mean = round(q_mean, 3),
            q_task_max = round(q_max, 3),
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

        # zero-dense-reward early BC bootstrap: rank CEM rollouts strictly by world-model predicted goal score
        if from_cem and self.win_cem_eps < max_plan_boot_wins and 'pred_score_seq' in res:
            pred_sc = res['pred_score_seq']
            k = max(2, int(round(obs_seq.shape[0] * top_plan_frac)))
            top = pred_sc.amax(dim = 1).topk(k).indices
            trunc = torch.minimum(lengths[top], pred_sc.argmax(dim = 1)[top].clamp(min = 1))
            add_episodes_to_buffer(self.bc_plan_buf, state_seq[top], act_seq[top], rew_seq[top], done_seq[top], trunc)

# rollout engine, policy+Q-guided CEM, and evaluation

@torch.no_grad()
def run_task_episodes(fp_mod, action_fn, init_states: np.ndarray | None, *, num_envs: int, max_steps: int = 80, step_scale: float = 40., seed: int = 0, score_obs_fn = None, return_trajs: bool = False) -> dict:
    n = len(init_states) if exists(init_states) else num_envs
    w = RawPushTWrapper(fp_mod, n, step_scale = step_scale, seed = seed)
    w.reset(seed = seed, reset_to_state = init_states)

    obs_seq = torch.zeros(n, max_steps + 1, STATE_DIM)
    act_seq = torch.zeros(n, max_steps, ACTION_DIM)
    rew_seq = torch.zeros(n, max_steps)
    done_seq = torch.zeros(n, max_steps, dtype = torch.bool)
    cov_seq = torch.zeros(n, max_steps + 1)
    pred_score_seq = torch.full((n, max_steps + 1), -1e9) if exists(score_obs_fn) else None

    obs_seq[:, 0], cov_seq[:, 0] = w.raw_state(), fp_mod.coverage(w.env.block_pose)
    if exists(score_obs_fn):
        pred_score_seq[:, 0] = score_obs_fn(obs_seq[:, 0])
    if return_trajs:
        ag_tr, pose_tr, cog_tr = torch.zeros(n, max_steps + 1, 2), torch.zeros(n, max_steps + 1, 3), torch.zeros(n, max_steps + 1, 2)
        ag_tr[:, 0], pose_tr[:, 0], cog_tr[:, 0] = w.env.agent_pos, w.env.block_pose, w.env._cog()

    alive, lengths, step = cov_seq[:, 0] <= WIN_COV, torch.zeros(n, dtype = torch.long), 0
    while step < max_steps and alive.any():
        a_exec = action_fn(step, obs_seq[:, step])
        prev_raw = obs_seq[:, step]
        w.step(a_exec)
        nxt_raw, nxt_cov = w.raw_state(), fp_mod.coverage(w.env.block_pose)
        just_won = alive & (nxt_cov > WIN_COV)

        act_seq[:, step] = einx.where('b, b d, b d -> b d', alive, a_exec, torch.zeros_like(a_exec))
        obs_seq[:, step + 1] = einx.where('b, b d, b d -> b d', alive, nxt_raw, prev_raw)
        cov_seq[:, step + 1] = einx.where('b, b, b -> b', alive, nxt_cov, cov_seq[:, step])
        rew_seq[:, step], done_seq[:, step] = just_won.float(), just_won
        if exists(score_obs_fn):
            pred_score_seq[:, step + 1] = einx.where('b, b, b -> b', alive, score_obs_fn(obs_seq[:, step + 1]), pred_score_seq[:, step])
        if return_trajs:
            ag_tr[:, step + 1] = einx.where('b, b d, b d -> b d', alive, w.env.agent_pos, ag_tr[:, step])
            pose_tr[:, step + 1] = einx.where('b, b d, b d -> b d', alive, w.env.block_pose, pose_tr[:, step])
            cog_tr[:, step + 1] = einx.where('b, b d, b d -> b d', alive, w.env._cog(), cog_tr[:, step])

        lengths = lengths + alive.long()
        alive = alive & (~just_won)
        step += 1

    if return_trajs and step < max_steps:
        ag_tr[:, step + 1:], pose_tr[:, step + 1:], cog_tr[:, step + 1:], cov_seq[:, step + 1:] = ag_tr[:, step:step + 1], pose_tr[:, step:step + 1], cog_tr[:, step:step + 1], cov_seq[:, step:step + 1]

    out = dict(obs_seq = obs_seq, act_seq = act_seq, rew_seq = rew_seq, done_seq = done_seq, lengths = lengths, won = done_seq.any(dim = 1), max_cov = cov_seq.amax(dim = 1), env_steps = int(lengths.sum().item()))
    if exists(pred_score_seq):
        out['pred_score_seq'] = pred_score_seq
    if return_trajs:
        out.update(agent_pos = ag_tr, block_pose = pose_tr, block_cog = cog_tr)
    return out

@torch.no_grad()
def imagine_task_policy(agent: ModeConditionedSEA, wm: StateCompressor, probe: nn.Module, z_cur: Tensor, horizon: int) -> tuple[Tensor, Tensor]:
    device = z_cur.device
    acts = torch.empty(z_cur.shape[0], horizon, ACTION_DIM, device = device, dtype = z_cur.dtype)
    zs = torch.empty(z_cur.shape[0], horizon, z_cur.shape[-1], device = device, dtype = z_cur.dtype)
    z = z_cur
    agent.sac.eval()
    for t in range(horizon):
        a_t = act_sac(agent.sac, agent.encode_task_obs(probe(z), device = device), deterministic = True)
        z = wm.predict_next(z, a_t)
        acts[:, t], zs[:, t] = a_t, z
    return acts, zs

@torch.no_grad()
def run_policy_task(fp_mod, sac: SAC, encode_fn, init_states: np.ndarray | None, *, num_envs: int, max_steps: int = 80, deterministic: bool = False, det_frac: float = 0., step_scale: float = 40., seed: int = 0, return_trajs: bool = False) -> dict:
    n = len(init_states) if exists(init_states) else num_envs
    sac.eval()
    det_mask = arange(n) < int(round(n * det_frac))

    def policy_action(_step: int, raw_obs: Tensor) -> Tensor:
        x = encode_fn(raw_obs)
        if deterministic:
            return act_sac(sac, x, deterministic = True)
        if det_frac > 0.:
            return einx.where('b, b d, b d -> b d', det_mask, act_sac(sac, x, deterministic = True), act_sac(sac, x, deterministic = False))
        return act_sac(sac, x, deterministic = False)

    return run_task_episodes(fp_mod, policy_action, init_states, num_envs = num_envs, max_steps = max_steps, step_scale = step_scale, seed = seed, return_trajs = return_trajs)

@torch.no_grad()
def run_guided_cem(fp_mod, wm: StateCompressor, probe: nn.Module, agent: ModeConditionedSEA | None, init_states: np.ndarray | None, *, num_envs: int, device: torch.device, max_steps: int = 80, cem_horizon: int = 12, cem_pop: int = 256, cem_gens: int = 4, commit_k: int = 1, action_noise: float = 0., use_policy_guide: bool = False, q_cem_weight: float = 0., step_scale: float = 40., seed: int = 0, return_trajs: bool = False) -> dict:
    planner = CEMPlanner(wm, ACTION_DIM, horizon = cem_horizon, pop_size = cem_pop, generations = cem_gens, elite_frac = 0.10)
    goal_com_norm = torch.tensor([GOAL_COM[0] / 256. - 1., GOAL_COM[1] / 256. - 1.], device = device)
    gen, cpu_gen = torch.Generator(device = device).manual_seed(seed + 13), torch.Generator().manual_seed(seed + 29)

    moved = exists(agent) and (use_policy_guide or q_cem_weight > 0.)
    if moved:
        agent.comp.to(device); agent.sac.actor.to(device); agent.sac.critics.to(device); agent.sac.eval()

    try:
        def score_fn(z_seq: Tensor) -> Tensor:
            base = docking_probe_score(probe, z_seq, goal_com_norm)
            if q_cem_weight > 0. and exists(agent):
                b_sz, p_sz = z_seq.shape[:2]
                t_end = agent.encode_task_obs(probe(rearrange(z_seq[:, :, -1, :], 'b p d -> (b p) d')), device = device)
                q_end = agent.sac.critics(t_end, cont_actions = act_sac(agent.sac, t_end, deterministic = True))[0].mean(dim = -1)
                base = base + q_cem_weight * rearrange(q_end, '(b p) -> b p', b = b_sz, p = p_sz)
            return base

        def score_obs_fn(raw_obs: Tensor) -> Tensor:
            z = wm.encode_detached(raw_obs.to(device))
            return docking_probe_score(probe, rearrange(z, 'b d -> b 1 1 d'), goal_com_norm)[:, 0].cpu()

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

        return run_task_episodes(fp_mod, cem_action, init_states, num_envs = num_envs, max_steps = max_steps, step_scale = step_scale, seed = seed, score_obs_fn = score_obs_fn, return_trajs = return_trajs)
    finally:
        if moved:
            agent.comp.to('cpu'); agent.sac.actor.to('cpu'); agent.sac.critics.to('cpu')

@torch.no_grad()
def evaluate_explore_mode(agent: ModeConditionedSEA, *, num_envs: int = 32, seed: int = 777) -> dict:
    agent.pot_sac.eval(); agent.sac.eval()
    w = RawPushTWrapper(agent.fp, num_envs, step_scale = agent.step_scale, seed = seed)
    w.reset(seed = seed)
    s = encode_policy_state(w, agent.comp, 'state', agent.comp_dev)
    for _ in range(agent.max_sw):
        w.step(act_sac(agent.pot_sac, s, deterministic = True))
        s = encode_policy_state(w, agent.comp, 'state', agent.comp_dev)

    pot_end = float((w.env.agent_pos - w.env._cog()).norm(dim = -1).mean())
    cog0, s0, z = w.env._cog().clone(), s.clone(), sample_skills(num_envs, agent.dim_skill)
    for _ in range(agent.skill_len):
        w.step(act_sac(agent.sac, explore_obs(s, s0, z, agent.dim_eff), deterministic = True))
        s = encode_policy_state(w, agent.comp, 'state', agent.comp_dev)

    disp = (w.env._cog() - cog0).norm(dim = -1)
    return dict(exp_pot_px = round(pot_end, 1), exp_touch_pct = round(float((disp > 5.).float().mean() * 100.), 1), exp_disp_px = round(float(disp.mean()), 1))

def evaluate_unified(fp_mod, agent: ModeConditionedSEA, wm: StateCompressor, probe: nn.Module, test_push: dict, dock_init: np.ndarray, device: torch.device, *, env_steps: int, max_steps: int, cem_horizon: int, cem_gens: int, step_scale: float, seed: int, use_guide: bool, q_cem_weight: float, extra: dict | None = None, return_trajs: bool = False):
    ev_pi = run_policy_task(fp_mod, agent.sac, agent.encode_task_obs, dock_init, num_envs = len(dock_init), max_steps = max_steps, deterministic = True, step_scale = step_scale, seed = seed, return_trajs = return_trajs)
    ev_cem = run_guided_cem(fp_mod, wm, probe, agent if use_guide else None, dock_init, num_envs = len(dock_init), device = device, max_steps = max_steps, cem_horizon = cem_horizon, cem_pop = 384, cem_gens = cem_gens, use_policy_guide = use_guide, q_cem_weight = q_cem_weight if (use_guide and agent.win_buf.size >= 128) else 0., step_scale = step_scale, seed = seed, return_trajs = return_trajs)
    row = dict(
        env_steps = env_steps,
        h10_blk_px = eval_h10_error(wm, probe, test_push, device),
        **{f'pi_{k}': v for k, v in summarize_docking(ev_pi).items()},
        **{f'cem_{k}': v for k, v in summarize_docking(ev_cem).items()},
        **evaluate_explore_mode(agent),
        win_cem_eps = agent.win_cem_eps,
        win_task_eps = agent.win_task_eps,
        **default(extra, {}),
    )
    return (row, ev_pi, ev_cem) if return_trajs else row

# baseline: task-only SAC on identical binary sparse reward

def run_task_only_sac(fp_mod, dock_init: np.ndarray, milestones: list[int], tmp_dir: Path, *, start_mode: str = 'docking', max_steps: int = 80, dim_hidden: int = 128, discount: float = 0.97, task_reward_scale: float = 10., step_scale: float = 40., seed: int = 42) -> list[dict]:
    sac = make_sac(STATE_DIM, discount = discount, lr = 1e-3, dim_hidden = dim_hidden, mlp_depth = 2, use_beta = False, init_alpha = 0.05, tau_ema = 0.02, actor_update_freq = 2)
    task_buf = TransitionBuffer(tmp_dir / 'base_task', STATE_DIM, max_transitions = 250_000)
    win_buf = TransitionBuffer(tmp_dir / 'base_win', STATE_DIM, max_transitions = 120_000)
    total_steps = total_wins = rnd = 0
    rows = []

    for target in milestones:
        while total_steps < target:
            rnd += 1
            res = run_policy_task(fp_mod, sac, lambda x: x, sample_starts(32, seed + 9000 + rnd * 17, start_mode = start_mode), num_envs = 32, max_steps = max_steps, det_frac = 0.125, step_scale = step_scale, seed = seed + 9500 + rnd)
            total_steps += res['env_steps']
            add_episodes_to_buffer(task_buf, res['obs_seq'], res['act_seq'], res['rew_seq'], res['done_seq'], res['lengths'])
            if res['won'].any():
                w = res['won']
                total_wins += int(w.sum().item())
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

        ev = run_policy_task(fp_mod, sac, lambda x: x, dock_init, num_envs = len(dock_init), max_steps = max_steps, deterministic = True, step_scale = step_scale, seed = seed)
        row = dict(env_steps = total_steps, train_wins = total_wins, **summarize_docking(ev))
        rows.append(row)
        print(f'  [Task-Only SAC] steps={total_steps:7,d} | wins={total_wins:4d} | >0.95={row["win95_pct"]:5.1f}% >=.90={row["ge90_pct"]:5.1f}% cov={row["mean_cov"]:.3f}', flush = True)
    return rows

# plotting & logging

def render_showcase(fp_mod, eval_res: dict, title: str, out_dir: Path, stem: str):
    order = np.argsort(-eval_res['max_cov'].numpy())
    pick = [int(order[k]) for k in (0, 3, 6, 10, 15, 20) if k < len(order)]
    render_trajectory_showcase(fp_mod, eval_res, pick, None, title, out_dir / f'{stem}_final.png', out_dir / f'{stem}.gif', out_dir / f'{stem}.mp4')

def plot_results(out_path: Path, title: str, unified_rows: list[dict], *, baseline_sac_rows: list[dict] | None = None, sea_cem_rows: list[dict] | None = None, sea_phase_k: float | None = None):
    fig, axes = plt.subplots(2, 2, figsize = (13.2, 8.8), dpi = 150)
    fig.patch.set_facecolor('white')
    for ax in axes.ravel():
        if exists(sea_phase_k):
            ax.axvspan(0, sea_phase_k, color = '#dbeafe', alpha = 0.35)
            ax.axvline(sea_phase_k, color = '#3b82f6', ls = ':', lw = 1.2)
        ax.grid(True, alpha = 0.28); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlabel('Total Env Steps (k)', fontsize = 9.5)

    xs_u = [r['env_steps'] / 1000. for r in unified_rows]

    ax = axes[0, 0]
    if exists(baseline_sac_rows):
        ax.plot([r['env_steps'] / 1000. for r in baseline_sac_rows], [r['win95_pct'] for r in baseline_sac_rows], color = '#9ca3af', ls = '--', marker = 'x', lw = 2., label = 'Task-Only SAC (sparse >0.95)')
    if exists(sea_cem_rows):
        ax.plot([r['env_steps'] / 1000. for r in sea_cem_rows], [r['win95_pct'] for r in sea_cem_rows], color = '#f59e0b', ls = '-.', marker = '^', lw = 2.2, label = 'SEA + WM + CEM Only')
    ax.plot(xs_u, [r['pi_win95_pct'] for r in unified_rows], color = '#059669', marker = 'o', lw = 2.6, label = 'Unified SAC pi(a|s, TASK) [Zero CEM]')
    ax.plot(xs_u, [r['cem_win95_pct'] for r in unified_rows], color = '#2563eb', marker = 'D', lw = 2.4, label = 'Unified Policy+Q-Guided CEM')
    ax.set_title('Exact Binary Sparse Win Rate (cov > 0.95)', fontweight = 'bold'); ax.set_ylabel('Win Rate (%)'); ax.set_ylim(-2, 102); ax.legend(fontsize = 8.2)

    ax = axes[0, 1]
    if exists(baseline_sac_rows):
        ax.plot([r['env_steps'] / 1000. for r in baseline_sac_rows], [r['mean_cov'] * 100. for r in baseline_sac_rows], color = '#9ca3af', ls = '--', marker = 'x', lw = 2., label = 'Task-Only SAC')
    if exists(sea_cem_rows):
        ax.plot([r['env_steps'] / 1000. for r in sea_cem_rows], [r['mean_cov'] * 100. for r in sea_cem_rows], color = '#f59e0b', ls = '-.', marker = '^', lw = 2.2, label = 'SEA + WM + CEM Only')
    ax.plot(xs_u, [r['pi_mean_cov'] * 100. for r in unified_rows], color = '#059669', marker = 'o', lw = 2.6, label = 'Unified pi(a|s, TASK)')
    ax.plot(xs_u, [r['cem_mean_cov'] * 100. for r in unified_rows], color = '#2563eb', marker = 'D', lw = 2.4, label = 'Guided CEM')
    ax.set_title('Mean Max PushT Green-T Coverage (%)', fontweight = 'bold'); ax.set_ylabel('Mean Max Coverage (%)'); ax.set_ylim(0, 102); ax.legend(fontsize = 8.2)

    ax = axes[1, 0]
    ax.plot(xs_u, [r['pi_ge80_pct'] for r in unified_rows], color = '#0284c7', marker = 's', lw = 2.2, label = 'cov >= 0.80 (%)')
    ax.plot(xs_u, [r['pi_ge90_pct'] for r in unified_rows], color = '#7c3aed', marker = '^', lw = 2.2, label = 'cov >= 0.90 (%)')
    ax.plot(xs_u, [r['pi_win95_pct'] for r in unified_rows], color = '#059669', marker = 'o', lw = 2.6, label = 'cov > 0.95 Exact Win (%)')
    ax.set_title('Reactive pi(a|s, mode=TASK) Precision Breakdown', fontweight = 'bold'); ax.set_ylabel('Success Rate (%)'); ax.set_ylim(-2, 102); ax.legend(fontsize = 8.5)

    ax = axes[1, 1]
    ax.plot(xs_u, [r['exp_touch_pct'] for r in unified_rows], color = '#ea580c', marker = 'o', lw = 2.4, label = 'EXPLORE mode touch (%)')
    ax.plot(xs_u, [r['h10_blk_px'] for r in unified_rows], color = '#dc2626', marker = 's', lw = 2.2, label = 'WM H10 Block Err (px)')
    ax.set_title('EXPLORE Mode Retention & World Model H10 Error', fontweight = 'bold'); ax.set_ylabel('Touch (%) / Error (px)'); ax.set_ylim(0, 105); ax.legend(fontsize = 8.5)

    fig.suptitle(title, fontsize = 13., fontweight = 'bold', y = 0.98)
    fig.tight_layout(rect = [0, 0, 1, 0.95]); fig.savefig(out_path); plt.close(fig)

def log_row(prefix: str, row: dict, elapsed: float):
    print(
        f'  [{prefix}] steps={row["env_steps"]:7,d} ({elapsed:4.0f}s) | '
        f'pi(TASK): >0.95={row["pi_win95_pct"]:5.1f}% >=.90={row["pi_ge90_pct"]:5.1f}% cov={row["pi_mean_cov"]:.3f} | '
        f'GuidedCEM: >0.95={row["cem_win95_pct"]:5.1f}% >=.90={row["cem_ge90_pct"]:5.1f}% cov={row["cem_mean_cov"]:.3f} | '
        f'wins={row["win_cem_eps"]}/{row["win_task_eps"]} EXP={row["exp_touch_pct"]:4.1f}% H10={row["h10_blk_px"]:4.1f}px',
        flush = True,
    )

# cli commands

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
    t0 = time.perf_counter()
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed)
    out_path = Path(out_dir); out_path.mkdir(parents = True, exist_ok = True)
    tmp_dir, device, fp_mod = Path(tempfile.mkdtemp(prefix = 'sea_sparse_pilot_')), get_device(), load_faster_pusht(x_jepa_dir)

    test_push = build_heldout_push_tasks(fp_mod, num_tasks = 64, rollout_len = 16, step_scale = step_scale, seed = 999)
    dock_init = build_docking_tasks(num_tasks = 32, seed = 2025)

    sea_agent = load_sea_checkpoint(sea_ckpt)
    comp, ck_pot, _, dim_eff, _ = sea_agent
    wm_ck = torch.load(wm_ckpt, map_location = 'cpu', weights_only = False)
    wm = build_jepa_sigreg(dim_latent = 16, reg_weight = 0.10).to(device)
    wm.load_state_dict(wm_ck['wm']); wm.eval()
    probe = build_probe(16).to(device)
    probe.load_state_dict(wm_ck['probe']); probe.eval()

    wm_buf = WorldModelBuffer(tmp_dir / 'wm_buf', max_episodes = 8000, max_len = max(60, max_steps))
    collect_sea_skill_steps(fp_mod, sea_agent, wm_buf, 60_000, ep_len = 60, action_noise = 0.15, step_scale = step_scale, seed = seed + 7)

    agent = ModeConditionedSEA(fp_mod, comp, dim_eff, None, None, tmp_dir, step_scale = step_scale, dim_hidden = 128, seed = seed)
    agent.pot_sac.load_state_dict(ck_pot.state_dict())
    for _ in range(6):
        agent.store_explore(wm_buf, use_potential = True)

    extra_steps = 0
    rows = [evaluate_unified(fp_mod, agent, wm, probe, test_push, dock_init, device, env_steps = 0, max_steps = max_steps, cem_horizon = cem_horizon, cem_gens = cem_gens, step_scale = step_scale, seed = seed, use_guide = False, q_cem_weight = 0.)]
    print(f'\n=== Pilot: Mode-Conditioned SEA + WM + CEM + Binary-Sparse SAC (use_bc={use_bc}, start_mode={start_mode}) ===', flush = True)
    log_row('cyc= 0', rows[0], time.perf_counter() - t0)

    last_pi_ev = last_cem_ev = None
    for cyc in range(1, cycles + 1):
        extra_steps += agent.store_explore(wm_buf, use_potential = True)

        use_guide = cyc >= 2
        cem_res = run_guided_cem(
            fp_mod, wm, probe, agent if use_guide else None,
            sample_starts(cem_eps_per_cycle, seed + 1000 + cyc * 37, start_mode = start_mode),
            num_envs = cem_eps_per_cycle, device = device, max_steps = max_steps,
            cem_horizon = cem_horizon, cem_pop = cem_pop, cem_gens = cem_gens, action_noise = 0.04,
            use_policy_guide = use_guide, q_cem_weight = q_cem_weight if (use_guide and agent.win_buf.size >= 128) else 0.,
            step_scale = step_scale, seed = seed + 2000 + cyc,
        )
        agent.ingest_rollouts(wm_buf, cem_res, from_cem = True)
        extra_steps += cem_res['env_steps']

        tsk_res = run_policy_task(
            fp_mod, agent.sac, agent.encode_task_obs,
            sample_starts(task_eps_per_cycle, seed + 3000 + cyc * 41, start_mode = start_mode),
            num_envs = task_eps_per_cycle, max_steps = max_steps, det_frac = 0.25,
            step_scale = step_scale, seed = seed + 4000 + cyc,
        )
        agent.ingest_rollouts(wm_buf, tsk_res, from_cem = False)
        extra_steps += tsk_res['env_steps']

        up = agent.update_mixed(sac_steps = sac_steps_per_cycle, bc_every = bc_every, q_filter_bc = q_filter_bc, use_bc = use_bc)
        if wm_steps_per_cycle > 0 and wm_buf.buf.num_episodes >= 64:
            probe, _ = update_world_model(wm, wm_buf, device, train_horizon = 10, steps = wm_steps_per_cycle, batch_size = 256, lr_start = 3e-4, lr_end = 5e-5, seed = seed + cyc)

        is_last = cyc == cycles
        ev = evaluate_unified(fp_mod, agent, wm, probe, test_push, dock_init, device, env_steps = extra_steps, max_steps = max_steps, cem_horizon = cem_horizon, cem_gens = cem_gens, step_scale = step_scale, seed = seed, use_guide = True, q_cem_weight = q_cem_weight, extra = up, return_trajs = is_last)
        row, last_pi_ev, last_cem_ev = ev if is_last else (ev, last_pi_ev, last_cem_ev)
        rows.append(row); log_row(f'cyc={cyc:2d}', row, time.perf_counter() - t0)

    plot_results(out_path / 'pilot_curves.png', 'Pilot: Mode-Conditioned SEA + WM + CEM + Binary-Sparse SAC (+ BC)', rows)
    if exists(last_pi_ev):
        render_showcase(fp_mod, last_pi_ev, 'Unified SAC pi(a|s, mode=TASK) [Zero CEM]', out_path, 'pilot_pi_task')
    if exists(last_cem_ev):
        render_showcase(fp_mod, last_cem_ev, 'Unified Policy+Q-Guided CEM (Binary Sparse >0.95)', out_path, 'pilot_guided_cem')
    (out_path / 'pilot.json').write_text(json.dumps(dict(rows = rows, elapsed_sec = round(time.perf_counter() - t0, 1)), indent = 2))
    print(f'\nSaved pilot curves & videos to {out_path} in {time.perf_counter() - t0:.1f}s', flush = True)

def compare(
    x_jepa_dir: str = '../x-jepa',
    out_dir: str = 'runs/sea_sparse_sac_compare',
    sea_phase_k: int = 80,
    total_k: int = 260,
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
    step_scale: float = 40.,
    seed: int = 42,
):
    t0 = time.perf_counter()
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed)
    out_path = Path(out_dir); out_path.mkdir(parents = True, exist_ok = True)
    tmp_dir, device, fp_mod = Path(tempfile.mkdtemp(prefix = 'sea_sparse_cmp_')), get_device(), load_faster_pusht(x_jepa_dir)

    test_push = build_heldout_push_tasks(fp_mod, num_tasks = 64, rollout_len = 16, step_scale = step_scale, seed = 999)
    dock_init = build_docking_tasks(num_tasks = 32, seed = 2025)
    milestones = [m * 1000 for m in range(sea_phase_k, total_k + 1, cycle_step_k)]

    print(f'\n=== 1. Baseline: Task-Only SAC (binary sparse >0.95, 0 -> {total_k}k) ===', flush = True)
    base_sac_rows = run_task_only_sac(fp_mod, dock_init, [40_000] + milestones, tmp_dir, start_mode = start_mode, max_steps = max_steps, step_scale = step_scale, seed = seed)

    print(f'\n=== 2. Unsupervised SEA Phase (0 -> {sea_phase_k}k, 0 task rewards) ===', flush = True)
    w_warm = RawPushTWrapper(fp_mod, 16, step_scale = step_scale, seed = seed)
    comp, dim_eff, warm_obs, warm_act, *_ = fit_state_compressor(
        w_warm, torch.device('cpu'), input_mode = 'state', img_size = 48, skill_len = 45, dim_latent = 8, horizon = 4,
        warmup_rounds = 16, steps = 1000, epx_eta = 20., reg_weight = 0.10, slow_weight = 20., energy_weight = 0.20, rank_rel_thresh = 0.25, target_std = 0.40,
    )
    wm_buf_u = WorldModelBuffer(tmp_dir / 'wm_unified', max_episodes = 12000, max_len = max(60, max_steps))
    wm_buf_u.add_batch(warm_obs, warm_act)
    agent = ModeConditionedSEA(fp_mod, comp, dim_eff, warm_obs, warm_act, tmp_dir, step_scale = step_scale, total_rounds = 90, dim_hidden = 128, seed = seed)
    agent.train_explore_until(wm_buf_u, sea_phase_k * 1000)

    wm_u = build_jepa_sigreg(dim_latent = 16, reg_weight = 0.10).to(device)
    pr_u, _ = update_world_model(wm_u, wm_buf_u, device, train_horizon = 10, steps = init_wm_steps, batch_size = 256, lr_start = 2e-3, lr_end = 1e-4, seed = seed)

    wm_buf_cem = wm_buf_u.clone_to(tmp_dir / 'wm_sea_cem_only')
    wm_cem = build_jepa_sigreg(dim_latent = 16, reg_weight = 0.10).to(device); wm_cem.load_state_dict(clone_sd(wm_u))
    pr_cem = build_probe(16).to(device); pr_cem.load_state_dict(clone_sd(pr_u)); pr_cem.eval()

    row0 = evaluate_unified(fp_mod, agent, wm_u, pr_u, test_push, dock_init, device, env_steps = wm_buf_u.total_env_steps, max_steps = max_steps, cem_horizon = cem_horizon, cem_gens = cem_gens, step_scale = step_scale, seed = seed, use_guide = False, q_cem_weight = 0.)
    unified_rows = [row0]
    sea_cem_rows = [dict(env_steps = wm_buf_cem.total_env_steps, h10_blk_px = row0['h10_blk_px'], win95_pct = row0['cem_win95_pct'], ge90_pct = row0['cem_ge90_pct'], ge80_pct = row0['cem_ge80_pct'], mean_cov = row0['cem_mean_cov'], win_steps = row0['cem_win_steps'])]

    print(f'\n=== 3. Unified Mode-Conditioned SEA + WM + CEM + Sparse-Task SAC + BC ({sea_phase_k}k -> {total_k}k) ===', flush = True)
    log_row('Unified', row0, time.perf_counter() - t0)

    total_u_steps, last_pi_ev, last_cem_ev = wm_buf_u.total_env_steps, None, None
    for cyc_i, target in enumerate(milestones[1:], start = 1):
        sub_i, up = 0, {}
        while total_u_steps < target:
            sub_i += 1
            total_u_steps += agent.store_explore(wm_buf_u, use_potential = True)

            use_guide = cyc_i >= 2 or sub_i >= 2
            cem_res = run_guided_cem(
                fp_mod, wm_u, pr_u, agent if use_guide else None,
                sample_starts(cem_eps_per_cycle, seed + 10000 + cyc_i * 100 + sub_i * 7, start_mode = start_mode),
                num_envs = cem_eps_per_cycle, device = device, max_steps = max_steps,
                cem_horizon = cem_horizon, cem_pop = cem_pop, cem_gens = cem_gens, action_noise = 0.04,
                use_policy_guide = use_guide, q_cem_weight = q_cem_weight if (use_guide and agent.win_buf.size >= 128) else 0.,
                step_scale = step_scale, seed = seed + 11000 + cyc_i * 100 + sub_i,
            )
            agent.ingest_rollouts(wm_buf_u, cem_res, from_cem = True)
            total_u_steps += cem_res['env_steps']

            tsk_res = run_policy_task(
                fp_mod, agent.sac, agent.encode_task_obs,
                sample_starts(task_eps_per_cycle, seed + 12000 + cyc_i * 100 + sub_i * 11, start_mode = start_mode),
                num_envs = task_eps_per_cycle, max_steps = max_steps, det_frac = 0.25,
                step_scale = step_scale, seed = seed + 13000 + cyc_i * 100 + sub_i,
            )
            agent.ingest_rollouts(wm_buf_u, tsk_res, from_cem = False)
            total_u_steps += tsk_res['env_steps']

            up = agent.update_mixed(sac_steps = sac_steps_per_cycle, bc_every = bc_every, q_filter_bc = True, use_bc = True)

        pr_u, _ = update_world_model(wm_u, wm_buf_u, device, train_horizon = 10, steps = cycle_wm_steps, batch_size = 256, lr_start = 1.2e-3, lr_end = 1e-4, seed = seed + cyc_i)
        is_last = cyc_i == len(milestones) - 1
        ev = evaluate_unified(fp_mod, agent, wm_u, pr_u, test_push, dock_init, device, env_steps = total_u_steps, max_steps = max_steps, cem_horizon = cem_horizon, cem_gens = cem_gens, step_scale = step_scale, seed = seed, use_guide = True, q_cem_weight = q_cem_weight, extra = up, return_trajs = is_last)
        row, last_pi_ev, last_cem_ev = ev if is_last else (ev, last_pi_ev, last_cem_ev)
        unified_rows.append(row); log_row('Unified', row, time.perf_counter() - t0)

    print(f'\n=== 4. Baseline: SEA + WM + CEM Only ({sea_phase_k}k -> {total_k}k) ===', flush = True)
    total_c_steps = wm_buf_cem.total_env_steps
    for cyc_i, target in enumerate(milestones[1:], start = 1):
        sub_i = 0
        while total_c_steps < target:
            sub_i += 1
            cem_res = run_guided_cem(
                fp_mod, wm_cem, pr_cem, None,
                sample_starts(64, seed + 20000 + cyc_i * 100 + sub_i * 7, start_mode = start_mode),
                num_envs = 64, device = device, max_steps = max_steps, cem_horizon = cem_horizon, cem_pop = cem_pop, cem_gens = cem_gens,
                action_noise = 0.08, step_scale = step_scale, seed = seed + 21000 + cyc_i * 100 + sub_i,
            )
            add_episodes_to_wm(wm_buf_cem, cem_res['obs_seq'], cem_res['act_seq'], cem_res['lengths'])
            total_c_steps += cem_res['env_steps']

        pr_cem, _ = update_world_model(wm_cem, wm_buf_cem, device, train_horizon = 10, steps = cycle_wm_steps, batch_size = 256, lr_start = 1.2e-3, lr_end = 1e-4, seed = seed + cyc_i)
        ev_c = run_guided_cem(fp_mod, wm_cem, pr_cem, None, dock_init, num_envs = len(dock_init), device = device, max_steps = max_steps, cem_horizon = cem_horizon, cem_pop = 384, cem_gens = cem_gens, step_scale = step_scale, seed = seed)
        row_c = dict(env_steps = total_c_steps, h10_blk_px = eval_h10_error(wm_cem, pr_cem, test_push, device), **summarize_docking(ev_c))
        sea_cem_rows.append(row_c)
        print(f'  [SEA+WM+CEM] steps={total_c_steps:7,d} | >0.95={row_c["win95_pct"]:5.1f}% >=.90={row_c["ge90_pct"]:5.1f}% cov={row_c["mean_cov"]:.3f} | H10={row_c["h10_blk_px"]:4.1f}px', flush = True)

    plot_results(out_path / 'sparse_sac_comparison.png', f'Mode-Conditioned SEA + WM + CEM + Binary-Sparse SAC + BC (0 -> {total_k}k)', unified_rows, baseline_sac_rows = base_sac_rows, sea_cem_rows = sea_cem_rows, sea_phase_k = float(sea_phase_k))
    if exists(last_pi_ev):
        render_showcase(fp_mod, last_pi_ev, f'Unified SAC pi(a|s, mode=TASK) at {total_k}k [Zero CEM]', out_path, 'pi_task')
    if exists(last_cem_ev):
        render_showcase(fp_mod, last_cem_ev, f'Unified Policy+Q-Guided CEM at {total_k}k (Binary Sparse >0.95)', out_path, 'guided_cem')

    torch.save(dict(comp = comp.state_dict(), sac = agent.sac.state_dict(), pot_sac = agent.pot_sac.state_dict(), classifier = agent.classifier.state_dict(), wm = wm_u.state_dict(), probe = pr_u.state_dict()), out_path / 'unified_agent.pt')
    (out_path / 'results.json').write_text(json.dumps(dict(task_only_sac = base_sac_rows, sea_wm_cem_only = sea_cem_rows, unified = unified_rows, elapsed_sec = round(time.perf_counter() - t0, 1)), indent = 2))
    print(f'\nSaved from-scratch comparison, curves & videos to {out_path} in {time.perf_counter() - t0:.1f}s', flush = True)

if __name__ == '__main__':
    Fire(dict(pilot = pilot, compare = compare))
