import math
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

param = pytest.mark.parametrize

from SEA_pytorch import (
    SEA,
    causal_chunks,
    sample_truncated_geometric,
    bilinear_contrastive,
    sample_skills,
    effective_reward,
    estimate_empowerment,
    EffectiveReward,
    SkillClassifier
)

def test_sea():
    model = SEA()
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == x.shape

def test_sample_truncated_geometric():
    rem = torch.tensor([1, 5, 25, 100])

    for discount in (0.0, 0.5, 0.99, 1.0):
        delta = sample_truncated_geometric(rem, discount)
        assert (delta >= 1).all() and (delta <= rem).all()

        delta_zero = sample_truncated_geometric(rem, discount, allow_zero = True)
        assert (delta_zero >= 0).all() and (delta_zero <= rem).all()

    # boundary checks

    assert (sample_truncated_geometric(rem, 0.99, rand_uniform = torch.zeros(4)) == 1).all()
    assert (sample_truncated_geometric(rem, 0.99, rand_uniform = torch.ones(4)) == rem).all()
    assert sample_truncated_geometric(0, 0.99, allow_zero = True).item() == 0

    # pmf check

    samples = sample_truncated_geometric(8, 0.85, sample_shape = 100_000, generator = torch.Generator().manual_seed(0))
    empirical = torch.bincount(samples - 1, minlength = 8).float() / 100_000
    expected = (0.85 ** torch.arange(8, dtype = torch.float32))
    expected = expected / expected.sum()
    assert (empirical - expected).abs().max() < 0.01

def test_bilinear_contrastive():
    embed = torch.randn(8, 16, requires_grad = True)
    future_embed = torch.randn(8, 16, requires_grad = True)
    skill = F.normalize(torch.randn(8, 16), dim = -1)

    score = bilinear_contrastive(embed, future_embed, skill, return_loss = False)
    assert score.shape == (8,)

    loss = bilinear_contrastive(embed, future_embed, skill)
    assert loss.ndim == 0

    loss.backward()
    assert embed.grad is not None and future_embed.grad is not None

def test_effective_reward():
    anchor = torch.randn(8, 16)
    next_embed = torch.randn(8, 16)
    skill = sample_skills(8, 16)

    assert torch.allclose(skill.norm(dim = -1), torch.ones(8), atol = 1e-5)

    # schedule: 0 up to 30%, linear to 1 at 80%, 1 thereafter

    reward_fn = EffectiveReward(total_steps = 100, num_samples = 64)

    assert reward_fn.partition_weight == 0.
    r0 = reward_fn(anchor, next_embed, skill, increment_step = 30)
    assert r0.shape == (8,)
    assert torch.allclose(r0, bilinear_contrastive(anchor, next_embed, skill, return_loss = False))

    assert reward_fn.partition_weight == 0.
    reward_fn(anchor, next_embed, skill, increment_step = 25)
    assert math.isclose(reward_fn.partition_weight, 0.5)

    reward_fn(anchor, next_embed, skill, increment_step = 25)
    assert reward_fn.partition_weight == 1.

    # bounded above by log(K) when partition_weight == 1

    r1 = reward_fn(anchor, anchor + skill * 10., skill, increment_step = False)
    assert (r1 <= math.log(64) + 1e-5).all()
    assert reward_fn.step.item() == 80

    # eval mode does not increment step

    reward_fn.eval()
    reward_fn(anchor, next_embed, skill)
    assert reward_fn.step.item() == 80

    # set_step_

    reward_fn.set_step_(55)
    assert math.isclose(reward_fn.partition_weight, 0.5)

def test_skill_classifier():
    encoder = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 8))
    classifier = SkillClassifier(
        encoder,
        total_steps = 100,
        noise_std = 0.05,
        num_samples = 64,
        extract_obs = lambda s: s[..., :4]
    )

    state = torch.randn(6, 10)
    future_state = torch.randn(6, 10)
    skill = sample_skills(6, 8)

    # contrastive loss updates state_encoder

    loss = classifier(state, future_state, skill)
    assert loss.ndim == 0

    loss.backward()
    assert all(p.grad is not None for p in encoder.parameters())

    # eval mode is deterministic (no noise added)

    classifier.eval()
    logits_a = classifier(state, future_state, skill, return_loss = False)
    logits_b = classifier(state, future_state, skill, return_loss = False)
    assert logits_a.shape == (6,)
    assert torch.equal(logits_a, logits_b)

    # reward runs under no_grad, without noise, and delegates step schedule

    classifier.train()
    r0 = classifier.reward(state, future_state, skill, increment_step = 55)
    assert r0.shape == (6,)
    assert not r0.requires_grad
    assert math.isclose(classifier.partition_weight, 0.5)

    # unnoised reward with partition_weight = 0 matches clean logits

    assert torch.allclose(r0, logits_a)

    classifier.set_step_(80)
    assert classifier.partition_weight == 1.

def test_causal_chunks():
    seq = torch.arange(2 * 6 * 3, dtype = torch.float32).reshape(2, 6, 3)

    # strided replicate-padded state chunks

    chunks, mask = causal_chunks(seq, chunk_len = 3, indices = torch.tensor([0, 5]), stride = 2, return_mask = True)
    assert chunks.shape == (2, 3, 3)
    assert torch.equal(mask[0], torch.tensor([False, False, True]))
    assert torch.equal(chunks[0], seq[0, [0, 0, 0]])
    assert torch.equal(chunks[1], seq[1, [1, 3, 5]])

    # past action chunks (offset = -1, constant pad, discrete dtype preserved)

    act = torch.arange(2 * 5, dtype = torch.long).reshape(2, 5)
    a_chunks, a_mask = causal_chunks(act, chunk_len = 3, indices = torch.tensor([0, 5]), offset = -1, pad_value = 9, return_mask = True)
    assert a_chunks.dtype == torch.long and a_chunks.shape == (2, 3)
    assert not a_mask[0].any() and (a_chunks[0] == 9).all()
    assert a_mask[1].all() and torch.equal(a_chunks[1], act[1, 2:5])

    # full sequence & multi-dim image features

    assert causal_chunks(seq, chunk_len = 4).shape == (2, 6, 4, 3)
    assert causal_chunks(torch.randn(2, 6, 3, 4, 4), chunk_len = 3, indices = 4).shape == (2, 3, 3, 4, 4)

def test_skill_classifier_chunks_and_actions():
    B, T, C, D_s, D_a, D_z = 6, 10, 4, 8, 2, 8
    states = torch.randn(B, T + 1, D_s)
    actions = torch.randn(B, T, D_a)
    skill = sample_skills(B, D_z)

    t0 = torch.zeros(B, dtype = torch.long)
    tf = torch.randint(1, T + 1, (B,))

    s0 = causal_chunks(states, C, indices = t0, stride = 2)
    sf = causal_chunks(states, C, indices = tf, stride = 2)
    a0, m0 = causal_chunks(actions, C, indices = t0, stride = 2, offset = -1, pad_value = 0., return_mask = True)
    af, mf = causal_chunks(actions, C, indices = tf, stride = 2, offset = -1, pad_value = 0., return_mask = True)

    # dual-stream state chunk + past action chunk encoder

    s_enc = nn.Sequential(Rearrange('... c d -> ... (c d)'), nn.Linear(C * 4, D_z))
    a_enc = nn.Sequential(Rearrange('... c d -> ... (c d)'), nn.Linear(C * D_a, D_z))
    clf = SkillClassifier(
        s_enc,
        total_steps = 100,
        action_encoder = a_enc,
        noise_std = 0.05,
        num_samples = 32,
        extract_obs = lambda s: s[..., :4]
    )

    loss = clf(s0, sf, skill, actions = a0, future_actions = af, action_mask = m0, future_action_mask = mf)
    assert loss.ndim == 0
    loss.backward()
    assert all(p.grad is not None for p in s_enc.parameters())
    assert all(p.grad is not None for p in a_enc.parameters())

    # all-False action_mask at t = 0 matches actions = None exactly

    clf.eval()
    assert torch.allclose(clf.encode_state(s0, a0, action_mask = m0), clf.encode_state(s0))

    r = clf.reward(s0, sf, skill, next_actions = af, next_action_mask = mf, increment_step = False)
    assert r.shape == (B,)

    # custom joint encoder accepting (state, actions)

    class JointEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4 + D_a, D_z)
        def forward(self, s, a):
            return self.net(torch.cat((s, a), dim = -1)).mean(dim = -2)

    joint = JointEncoder()
    clf_joint = SkillClassifier(joint, total_steps = 100, num_samples = 32, extract_obs = lambda s: s[..., :4])
    clf_joint(s0, sf, skill, actions = a0, future_actions = af, action_mask = m0, future_action_mask = mf).backward()
    assert all(p.grad is not None for p in joint.parameters())

def test_estimate_empowerment():
    B, D_s, D_a, D_z = 4, 6, 2, 4
    state = torch.randn(B, D_s)
    skills = torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.]])

    # continuous SAC: twin critic min per skill, then mean over skills, then floor & (1 - gamma)

    def actor(s, s0, z):
        assert torch.equal(s, s0)
        return z[:, :D_a], torch.zeros(s.shape[0])

    def twin_critic(s, s0, z, a):
        assert torch.equal(s, s0)
        pos = z[:, 0] > 0.
        q1 = torch.where(pos, torch.full((s.shape[0],), 4.), torch.full((s.shape[0],), 2.))
        q2 = torch.where(pos, torch.full((s.shape[0],), 6.), torch.full((s.shape[0],), -2.))
        return q1, q2

    # per-skill min: [+4, -2] -> mean = +1.0 -> (1 - 0.9) * 1.0 = 0.1
    emp = estimate_empowerment(state, twin_critic, D_z, policy = actor, skills = skills, discount = 0.9, clamp_min = 0.)
    assert emp.shape == (B,)
    assert torch.allclose(emp, torch.full((B,), 0.1))

    # discrete PQN / DQN: max_a min(Q1, Q2)

    def discrete_q(s, s0, z):
        q1 = torch.stack((z[:, 0] + 1., z[:, 1] + 2., z[:, 2]), dim = -1)
        q2 = torch.stack((z[:, 0] + 3., z[:, 1] + 0.5, z[:, 2] + 1.), dim = -1)
        return q1, q2

    emp_pqn = estimate_empowerment(state, discrete_q, D_z, skills = skills)
    assert torch.allclose(emp_pqn, torch.full((B,), 1.25))

    # state chunk input + random sphere sampling

    chunks = torch.randn(B, 5, D_s)
    v_fn = lambda s, s0, z: (s.mean(dim = -2)[:, :D_z] * z).sum(dim = -1)
    assert estimate_empowerment(chunks, v_fn, D_z, num_skills = 8).shape == (B,)
