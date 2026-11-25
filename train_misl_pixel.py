from __future__ import annotations

import copy
import math
import time

from fire import Fire

import torch
from torch import nn, arange, cat, eye, linspace, rand, randint, randperm, stack, tensor, where, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.distributions import Normal

from torch_einops_utils import trainable_parameters

from SEA_pytorch import (
    StateCompressor,
    SkillClassifier,
    estimate_empowerment,
    sample_skills,
    sample_truncated_geometric
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def step_opt(opt, loss: Tensor):
    opt.zero_grad()
    loss.backward()
    opt.step()

@torch.no_grad()
def batched(fn, x: Tensor, chunk: int = 1024) -> Tensor:
    return cat([fn(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)])

# environment: 24x24 pixel Ball-in-Box with optional background distractor

class PixelBallInBoxEnv:
    def __init__(self, num_envs: int, *, episode_len: int = 50, distractor: str = 'stochastic', random_start: bool = True, device: torch.device = torch.device('cpu')):
        self.num_envs, self.episode_len, self.distractor, self.random_start, self.device, self.img_size = num_envs, episode_len, distractor, random_start, device, 24
        self.pos, self.dist_pos, self.dist_vel = (torch.zeros(num_envs, 2, device = device) for _ in range(3))
        coords = linspace(-1., 1., self.img_size, device = device)
        gy, gx = torch.meshgrid(coords, coords, indexing = 'ij')
        self.gx, self.gy = gx.unsqueeze(0), gy.unsqueeze(0)

    def _render(self) -> Tensor:
        ball = torch.exp(-0.5 * ((self.gx - self.pos[:, 0].view(-1, 1, 1)).square() + (self.gy - self.pos[:, 1].view(-1, 1, 1)).square()) / (0.13 ** 2))
        if self.distractor == 'none':
            return ball.unsqueeze(1)
        blob = 0.65 * torch.exp(-0.5 * ((self.gx - self.dist_pos[:, 0].view(-1, 1, 1)).square() + (self.gy - self.dist_pos[:, 1].view(-1, 1, 1)).square()) / (0.34 ** 2))
        return torch.maximum(ball, blob).unsqueeze(1)

    def reset(self, start_pos: Tensor | None = None) -> tuple[Tensor, Tensor]:
        self.pos = start_pos.to(self.device).clone() if exists(start_pos) else ((rand(self.num_envs, 2, device = self.device) * 2. - 1.) if self.random_start else torch.zeros(self.num_envs, 2, device = self.device))
        if self.distractor != 'none':
            self.dist_pos = rand(self.num_envs, 2, device = self.device) * 2. - 1.
            ang = rand(self.num_envs, device = self.device) * (2. * math.pi)
            self.dist_vel = stack((ang.cos(), ang.sin()), dim = -1) * 0.11
        return self._render(), self.pos.clone()

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        action = action.clamp(-1., 1.)
        self.pos = (self.pos + (action / action.norm(dim = -1, keepdim = True).clamp(min = 1.)) * 0.08).clamp(-1., 1.)
        if self.distractor == 'stochastic':
            self.dist_pos = rand(self.num_envs, 2, device = self.device) * 2. - 1.
        elif self.distractor == 'bouncing':
            nxt = self.dist_pos + self.dist_vel
            self.dist_vel = where((nxt > 1.) | (nxt < -1.), -self.dist_vel, self.dist_vel)
            self.dist_pos = nxt.clamp(-1., 1.)
        return self._render(), self.pos.clone()

# trajectory ring buffer

class PixelRingBuffer:
    def __init__(self, cap: int, episode_len: int, img_size: int, dim_skill: int, device: torch.device):
        self.cap, self.episode_len, self.device = cap, episode_len, device
        self.obs = torch.zeros(cap, episode_len + 1, 1, img_size, img_size, dtype = torch.uint8, device = device)
        self.actions = torch.zeros(cap, episode_len, 2, device = device)
        self.skills = torch.zeros(cap, dim_skill, device = device)
        self.ptr = self.size = 0

    def add(self, obs: Tensor, actions: Tensor, skills: Tensor):
        batch = obs.shape[0]
        idx = (arange(batch, device = self.device) + self.ptr) % self.cap
        self.obs[idx] = (obs.clamp(0., 1.) * 255.0).round().to(torch.uint8)
        self.actions[idx], self.skills[idx] = actions, skills
        self.ptr = (self.ptr + batch) % self.cap
        self.size = min(self.size + batch, self.cap)

    def sample_clips(self, batch: int, horizon: int) -> tuple[Tensor, Tensor]:
        ep = randint(0, self.size, (batch,), device = self.device)
        t0 = randint(0, self.episode_len - horizon + 1, (batch,), device = self.device)
        oo, oa = arange(horizon + 1, device = self.device), arange(horizon, device = self.device)
        return self.obs[ep[:, None], t0[:, None] + oo].float() / 255.0, self.actions[ep[:, None], t0[:, None] + oa]

    def sample_contrastive(self, batch: int, discount: float, recent: int = 512) -> tuple[Tensor, Tensor, Tensor]:
        cnt = min(self.size, recent)
        rec = (arange(cnt, device = self.device) + (self.ptr - cnt) % self.cap) % self.cap
        ep = rec[randperm(cnt, device = self.device)[:batch]] if cnt >= batch else rec[randint(0, cnt, (batch,), device = self.device)]
        t = randint(0, self.episode_len, (batch,), device = self.device)
        tf = t + sample_truncated_geometric(self.episode_len - t, discount, device = self.device)
        return self.obs[ep, t].float() / 255.0, self.obs[ep, tf].float() / 255.0, self.skills[ep]

    def sample_transitions(self, batch: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        ep = randint(0, self.size, (batch,), device = self.device)
        t = randint(0, self.episode_len, (batch,), device = self.device)
        return self.obs[ep, 0].float() / 255.0, self.obs[ep, t].float() / 255.0, self.obs[ep, t + 1].float() / 255.0, self.actions[ep, t], self.skills[ep]

# networks and swappable compressors

def MLP(dim_in: int, dim_hidden: int, dim_out: int, *, depth: int = 2, layernorm: bool = False) -> nn.Sequential:
    layers: list[Module] = []
    dim = dim_in
    for _ in range(depth):
        layers.extend([nn.Linear(dim, dim_hidden), *([nn.LayerNorm(dim_hidden)] if layernorm else []), nn.GELU()])
        dim = dim_hidden
    return nn.Sequential(*layers, nn.Linear(dim, dim_out))

class ConvEncoder(Module):
    def __init__(self, dim_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.GELU(), nn.Conv2d(16, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.GELU(), nn.Flatten(),
            nn.Linear(32 * 3 * 3, dim_out), nn.LayerNorm(dim_out), nn.GELU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.reshape(-1, *x.shape[-3:])).reshape(*x.shape[:-3], -1)

class AEStateCompressor(Module):
    def __init__(self, encoder: Module, dim_encoder_out: int, dim_latent: int):
        super().__init__()
        self.encoder = encoder
        self.to_latent = nn.Sequential(nn.Linear(dim_encoder_out, dim_latent), nn.LayerNorm(dim_latent, elementwise_affine = False))
        self.dec = nn.Sequential(
            nn.Linear(dim_latent, 32 * 3 * 3), nn.GELU(), nn.Unflatten(1, (32, 3, 3)),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid()
        )

    @torch.no_grad()
    def encode_detached(self, obs: Tensor) -> Tensor:
        return self.to_latent(self.encoder(obs))

    def forward(self, obs: Tensor, actions: Tensor | None = None) -> Tensor:
        if not exists(actions):
            return self.to_latent(self.encoder(obs))
        o = obs[:, 0]
        return F.mse_loss(self.dec(self.to_latent(self.encoder(o))), o)

class TanhGaussianActor(Module):
    def __init__(self, dim_state: int, dim_skill: int, dim_action: int = 2, dim_hidden: int = 256):
        super().__init__()
        self.net = MLP(dim_state * 2 + dim_skill, dim_hidden, dim_action * 2)

    def forward(self, s: Tensor, s0: Tensor, z: Tensor, *, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        mean, log_std = self.net(cat((s, s0, z), dim = -1)).chunk(2, dim = -1)
        if deterministic:
            return mean.tanh(), torch.zeros(s.shape[:-1], device = s.device, dtype = s.dtype)
        dist = Normal(mean, log_std.clamp(-5., 2.).exp())
        u = dist.rsample()
        return u.tanh(), (dist.log_prob(u) - 2. * (math.log(2.) - u - F.softplus(-2. * u))).sum(dim = -1)

class TwinCritic(Module):
    def __init__(self, dim_state: int, dim_skill: int, dim_action: int = 2, dim_hidden: int = 256):
        super().__init__()
        dim_in = dim_state * 2 + dim_skill + dim_action
        self.q1 = MLP(dim_in, dim_hidden, 1, layernorm = True)
        self.q2 = MLP(dim_in, dim_hidden, 1, layernorm = True)

    def forward(self, s: Tensor, s0: Tensor, z: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        x = cat((s, s0, z, a), dim = -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

# diagnostics

def ridge_r2(x: Tensor, y: Tensor) -> float:
    x, y = x.detach().cpu().double(), y.detach().cpu().double()
    xa = cat((x, torch.ones(x.shape[0], 1, dtype = torch.float64)), dim = -1)
    reg = 1e-3 * eye(xa.shape[1], dtype = torch.float64)
    reg[-1, -1] = 0.
    w = torch.linalg.solve(xa.T @ xa + reg, xa.T @ y)
    return float(1. - (y - xa @ w).square().sum().item() / ((y - y.mean(0, keepdim = True)).square().sum().item() + 1e-12))

def render_ascii(traj: Tensor, w: int = 33, h: int = 17) -> str:
    canvas = [[' ' for _ in range(w)] for _ in range(h)]
    for x in range(w): canvas[0][x] = canvas[h - 1][x] = '-'
    for y in range(h): canvas[y][0] = canvas[y][w - 1] = '|'
    canvas[0][0] = canvas[0][w - 1] = canvas[h - 1][0] = canvas[h - 1][w - 1] = '+'
    to_xy = lambda pt: (max(1, min(w - 2, int(round((pt[0].item() + 1.) * 0.5 * (w - 3))) + 1)), max(1, min(h - 2, int(round((1. - pt[1].item()) * 0.5 * (h - 3))) + 1)))
    for k in range(traj.shape[0]):
        for t in range(1, traj.shape[1] - 1, 3):
            px, py = to_xy(traj[k, t])
            if canvas[py][px] == ' ': canvas[py][px] = '.'
        px, py = to_xy(traj[k, -1])
        canvas[py][px] = '0123456789abcdef'[k % 16]
    cx, cy = to_xy(torch.zeros(2))
    canvas[cy][cx] = 'O'
    return '\n'.join(''.join(row) for row in canvas)

# training

def train(
    compressor: str = 'epiplexity',
    distractor: str = 'stochastic',
    dynamics_loss_weight: float = 1.,
    rollout_discount: float = 0.8,
    reg_weight: float | None = None,
    seed: int = 0,
    device: str = 'auto',
    rounds: int = 100,
    dim_skill: int = 2,
    dim_latent: int = 4,
    horizon: int = 4,
    compressor_warmup_steps: int = 800,
    noise_std: float | None = None,
    eval_every: int = 25
):
    assert compressor in ('epiplexity', 'pure_epiplexity', 'sigreg', 'ae', 'none')
    torch.manual_seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')) if device == 'auto' else torch.device(device)
    sac_dev = torch.device('cpu') if (dev.type == 'mps' and compressor != 'none') else dev

    num_envs, episode_len, batch_size, discount, temperature = 32, 50, 256, 0.95, 0.1
    env = PixelBallInBoxEnv(num_envs, episode_len = episode_len, distractor = distractor, device = dev)
    buf = PixelRingBuffer(2048, episode_len, env.img_size, dim_skill, dev)
    noise_std = default(noise_std, 0.03 if compressor == 'none' else 0.15)

    comp: Module | None = None
    sac_enc = target_sac_enc = None
    if compressor in ('epiplexity', 'pure_epiplexity', 'sigreg'):
        reg = 'sigreg' if compressor == 'sigreg' else 'epiplexity'
        dyn_w = 0. if compressor == 'pure_epiplexity' else dynamics_loss_weight
        comp = StateCompressor(ConvEncoder(128), dim_latent, dim_encoder_out = 128, dim_action = 2 if dyn_w > 0. else None, obs_ndim = 3, regularizer = reg, dynamics_loss_weight = dyn_w, rollout_discount = rollout_discount, reg_weight = reg_weight).to(dev)
    elif compressor == 'ae':
        comp = AEStateCompressor(ConvEncoder(128), 128, dim_latent).to(dev)

    if exists(comp):
        opt_comp = torch.optim.Adam(trainable_parameters(comp), lr = 1e-3)
        clf_enc, sac_dim = MLP(dim_latent, 256, dim_skill, depth = 2, layernorm = True), dim_latent
    else:
        clf_enc, sac_dim = nn.Sequential(ConvEncoder(128), MLP(128, 128, dim_skill, depth = 1, layernorm = True)), 32
        sac_enc = ConvEncoder(sac_dim).to(dev)
        target_sac_enc = copy.deepcopy(sac_enc).to(dev).requires_grad_(False)

    classifier = SkillClassifier(clf_enc, total_steps = rounds * num_envs * episode_len, temperature = temperature, noise_std = noise_std, num_samples = 512, state_compressor = comp).to(dev)
    actor, critic = TanhGaussianActor(sac_dim, dim_skill).to(sac_dev), TwinCritic(sac_dim, dim_skill).to(sac_dev)
    target_critic = copy.deepcopy(critic).to(sac_dev).requires_grad_(False)
    log_alpha = torch.zeros((), device = sac_dev, requires_grad = True)

    opt_clf = torch.optim.Adam(classifier.parameters(), lr = 3e-4)
    opt_actor = torch.optim.Adam(actor.parameters(), lr = 3e-4)
    opt_critic = torch.optim.Adam(list(critic.parameters()) + (list(sac_enc.parameters()) if exists(sac_enc) else []), lr = 3e-4)
    opt_alpha = torch.optim.Adam([log_alpha], lr = 3e-4)
    cp, tcp = list(critic.parameters()), list(target_critic.parameters())
    ep_s, tep_s = (list(sac_enc.parameters()), list(target_sac_enc.parameters())) if exists(sac_enc) else (None, None)

    @torch.no_grad()
    def encode_sac(obs: Tensor) -> Tensor:
        return comp.encode_detached(obs).to(sac_dev) if exists(comp) else sac_enc(obs)

    @torch.no_grad()
    def rollout(e: PixelBallInBoxEnv, *, random_actions = False, deterministic = False, start_pos = None, skills = None):
        o, p = e.reset(start_pos = start_pos)
        z = skills.to(dev) if exists(skills) else sample_skills(e.num_envs, dim_skill, device = dev)
        z_s, s0 = z.to(sac_dev), (None if random_actions else encode_sac(o))
        ao = torch.empty(e.num_envs, e.episode_len + 1, 1, e.img_size, e.img_size, device = dev)
        ap = torch.empty(e.num_envs, e.episode_len + 1, 2, device = dev)
        aa = torch.empty(e.num_envs, e.episode_len, 2, device = dev)
        ao[:, 0], ap[:, 0] = o, p
        for t in range(e.episode_len):
            a = (rand(e.num_envs, 2, device = dev) * 2. - 1.) if random_actions else actor(encode_sac(o), s0, z_s, deterministic = deterministic)[0].to(dev)
            o, p = e.step(a)
            aa[:, t], ao[:, t + 1], ap[:, t + 1] = a, o, p
        return ao, ap, aa, z

    # warmup rollouts and compressor prefit

    for _ in range(8):
        ao, _, aa, z = rollout(env, random_actions = True)
        buf.add(ao, aa, z)

    t0 = time.perf_counter()
    if exists(comp):
        for _ in range(compressor_warmup_steps):
            step_opt(opt_comp, comp(*buf.sample_clips(128, horizon)))
        comp.eval()

    pe = PixelBallInBoxEnv(1024, episode_len = 1, distractor = distractor, device = dev)
    po, pp = pe.reset()
    with torch.no_grad():
        lat0 = comp.encode_detached(po) if exists(comp) else classifier.state_encoder[0](po)
    r2_a, r2_d = ridge_r2(lat0, pp), (ridge_r2(lat0, pe.dist_pos) if distractor != 'none' else 0.)

    print(f'Pixel MISL | compressor={compressor} distractor={distractor} device={dev}')
    print(f'Post-warmup probe: lin R2_agent={r2_a:.2f}, lin R2_dist={r2_d:.2f}, std={lat0.std(0).mean().item():.2f}')
    print(f'{"rnd":>4} | {"nce":>5} | {"disp":>5} | {"align":>5} | {"gap":>5} | {"oct":>3} | {"cov":>5} | {"acc":>5} | {"MI":>5} | {"E_ctr":>5} | {"E_crn":>5}\n' + '-' * 82)

    @torch.no_grad()
    def evaluate():
        classifier.eval()
        ang = linspace(0., 2. * math.pi, 17, device = dev)[:-1]
        rs = stack((ang.cos(), ang.sin()), dim = -1)
        if dim_skill > 2:
            rs = cat((rs, torch.zeros(16, dim_skill - 2, device = dev)), dim = -1)
        _, tr, _, _ = rollout(PixelBallInBoxEnv(16, episode_len = episode_len, distractor = distractor, random_start = False, device = dev), deterministic = True, start_pos = torch.zeros(16, 2, device = dev), skills = rs)
        d = tr[:, -1] - tr[:, 0]
        dn = d.norm(dim = -1)
        ud = d / dn.clamp(min = 1e-6).unsqueeze(-1)
        u_s, _, vh_s = torch.linalg.svd(rs[:, :2].T @ ud)
        align = ((rs[:, :2] @ (u_s @ vh_s)) * ud).sum(-1).mean().item()
        ea = torch.remainder(torch.atan2(d[:, 1], d[:, 0]), 2. * math.pi)
        sa, _ = ea.sort()
        gap = math.degrees(cat((sa[1:] - sa[:-1], sa[:1] + 2. * math.pi - sa[-1:])).max().item())
        octs = int(torch.floor(ea / (math.pi / 4.)).long().clamp(0, 7).unique().numel())

        oe, pe2, _, ze = rollout(PixelBallInBoxEnv(64, episode_len = episode_len, distractor = distractor, random_start = False, device = dev), deterministic = True)
        pts = pe2.reshape(-1, 2)
        gx, gy = ((pts[:, 0] + 1.) * 5.).long().clamp(0, 9), ((pts[:, 1] + 1.) * 5.).long().clamp(0, 9)
        cov = (gx * 10 + gy).unique().numel() / 100.
        dl = sample_truncated_geometric(torch.full((64,), episode_len, dtype = torch.long, device = dev), discount, device = dev)
        ar = arange(64, device = dev)
        acc = ((((classifier.encode_state(oe[ar, dl]) - classifier.encode_state(oe[:, 0])) @ ze.T) / temperature).argmax(-1) == ar).float().mean().item()
        mi = classifier.reward(oe[:, 0], oe[ar, dl], ze, partition_weight = 1., increment_step = False).mean().item()

        probe_pts = tensor([[0., 0.], [-0.9, -0.9], [-0.9, 0.9], [0.9, -0.9], [0.9, 0.9]], device = dev)
        o_emp, _ = PixelBallInBoxEnv(5, episode_len = 1, distractor = distractor, random_start = False, device = dev).reset(start_pos = probe_pts)
        emp = estimate_empowerment(encode_sac(o_emp), critic, dim_skill, num_skills = 32, policy = actor, discount = discount)
        classifier.train()
        return dn.mean().item(), align, gap, octs, cov, acc, mi, emp[0].item(), emp[1:].mean().item(), tr.cpu()

    env_steps = 0
    for rnd in range(1, rounds + 1):
        ao, _, aa, z = rollout(env)
        buf.add(ao, aa, z)
        env_steps += num_envs * episode_len
        classifier.set_step_(env_steps)

        nce_v = 0.
        for _ in range(12):
            loss_nce = classifier(*buf.sample_contrastive(batch_size, discount))
            step_opt(opt_clf, loss_nce)
            nce_v += loss_nce.item()
        nce_v /= 12

        ao0, ao_, aon, aa_, az = buf.sample_transitions(20 * batch_size)
        with torch.no_grad():
            ar = cat([classifier.reward(ao0[i:i + 1024], aon[i:i + 1024], az[i:i + 1024], increment_step = False) for i in range(0, 20 * batch_size, 1024)]).to(sac_dev)
            if exists(comp):
                as0, as_, asn = (batched(comp.encode_detached, x).to(sac_dev) for x in (ao0, ao_, aon))
            aa_s, az_s = aa_.to(sac_dev), az.to(sac_dev)

        for i in range(20):
            sl = slice(i * batch_size, (i + 1) * batch_size)
            if exists(comp):
                s0_q, s_q, sn_t = as0[sl], as_[sl], asn[sl]
                s0_t = s0_pi = s0_q
                sn_pi = sn_t
            else:
                with torch.no_grad():
                    s0_t, sn_t = target_sac_enc(cat((ao0[sl], aon[sl]))).chunk(2)
                s0_q, s_q, sn_pi = sac_enc(cat((ao0[sl], ao_[sl], aon[sl]))).chunk(3)
                s0_pi, sn_pi = s0_q.detach(), sn_pi.detach()

            with torch.no_grad():
                an, lpn = actor(sn_pi, s0_pi, az_s[sl])
                tq1, tq2 = target_critic(sn_t, s0_t, az_s[sl], an)
                qt = ar[sl] + discount * (torch.minimum(tq1, tq2) - log_alpha.exp() * lpn)

            q1, q2 = critic(s_q, s0_q, az_s[sl], aa_s[sl])
            step_opt(opt_critic, F.mse_loss(q1, qt) + F.mse_loss(q2, qt))

            an2, lp2 = actor(s_q.detach(), s0_q.detach(), az_s[sl])
            q1n, q2n = critic(s_q.detach(), s0_q.detach(), az_s[sl], an2)
            step_opt(opt_actor, (log_alpha.exp().detach() * lp2 - torch.minimum(q1n, q2n)).mean())
            step_opt(opt_alpha, -(log_alpha * (lp2.detach() - 2.)).mean())

            with torch.no_grad():
                torch._foreach_lerp_(tcp, cp, 0.02)
                if exists(ep_s):
                    torch._foreach_lerp_(tep_s, ep_s, 0.02)

        if rnd == 1 or rnd % eval_every == 0 or rnd == rounds:
            disp, align, gap, octs, cov, acc, mi, e_ctr, e_crn, tr = evaluate()
            print(f'{rnd:4d} | {nce_v:5.2f} | {disp:5.2f} | {align:5.2f} | {gap:5.1f} | {octs:3d} | {cov:5.2f} | {acc:5.2f} | {mi:5.2f} | {e_ctr:5.2f} | {e_crn:5.2f}', flush = True)
    print(f'\nDone in {time.perf_counter() - t0:.1f}s\n{render_ascii(tr)}')

if __name__ == '__main__':
    Fire(train)
