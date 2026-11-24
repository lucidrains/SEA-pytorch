import math
import pytest
import torch
from torch import nn
import torch.nn.functional as F

param = pytest.mark.parametrize

from SEA_pytorch import (
    SEA,
    sample_truncated_geometric,
    bilinear_contrastive,
    sample_skills,
    effective_reward,
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
