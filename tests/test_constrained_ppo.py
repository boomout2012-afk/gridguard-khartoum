from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from gridguard.constrained import eligible_candidates, evaluate_constrained_candidates
from gridguard.constrained_ppo import (
    ConstrainedActorCritic,
    ConstrainedPPOController,
    build_training_observation,
    clipped_ppo_loss,
    critical_action_mask,
    critical_action_tiers,
    generalized_advantage_estimate,
    load_model,
    masked_categorical,
    save_model,
)
from gridguard.controllers import DecisionContext
from gridguard.domain import LOCALITY_DEMAND_WEIGHTS, ControllerName
from gridguard.simulation import build_observation, dispatch_service_tiers


@pytest.fixture(autouse=True)
def _small_cpu_pool():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def _context(*, dtype=torch.float32, device="cpu") -> DecisionContext:
    generator = torch.Generator(device=device).manual_seed(99251)
    demand = 1 + torch.rand((8, 7, 5), dtype=dtype, device=device, generator=generator)
    supply = demand[..., :3].sum(dim=(-1, -2)) + demand[..., 3:].sum(dim=(-1, -2)) * 0.3
    past_demand = torch.full((8, 7), 100.0, dtype=dtype, device=device)
    past_served = past_demand * torch.rand((8, 7), generator=generator, dtype=dtype, device=device)
    previous = torch.ones((8, 7), dtype=torch.long, device=device)
    observation = build_observation(
        hour=8,
        demand=demand,
        supply=supply,
        cumulative_demand=past_demand,
        cumulative_served=past_served,
        previous_tier=previous,
    )
    return DecisionContext(
        hour=8,
        observation=observation,
        current_demand_mw=demand,
        current_supply_mw=supply,
        cumulative_demand_mwh=past_demand,
        cumulative_served_mwh=past_served,
        previous_service_tiers=previous,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fast_mask_matches_exact_constrained_feasibility_and_boundary(dtype):
    context = _context(dtype=dtype)
    candidates = evaluate_constrained_candidates(context)
    torch.testing.assert_close(critical_action_tiers("cpu"), candidates.tiers, atol=0, rtol=0)
    torch.testing.assert_close(
        critical_action_mask(context), eligible_candidates(candidates, None), atol=0, rtol=0
    )
    tiers = candidates.tiers[23].expand(8, -1)
    requested, _ = dispatch_service_tiers(
        demand=context.current_demand_mw, supply=context.current_supply_mw, tiers=tiers
    )
    boundary = requested.sum(dim=(-1, -2))
    for supply in (boundary, torch.nextafter(boundary, torch.zeros_like(boundary))):
        edge = replace(context, current_supply_mw=supply)
        expected = eligible_candidates(evaluate_constrained_candidates(edge), None)
        torch.testing.assert_close(critical_action_mask(edge), expected, atol=0, rtol=0)


def test_mask_shortfall_fallback_is_critical_only_and_zero_demand_allows_all():
    context = _context()
    context.current_supply_mw.zero_()
    mask = critical_action_mask(context)
    assert mask[:, 0].all()
    assert mask.sum() == 8
    context.current_demand_mw.zero_()
    assert critical_action_mask(context).all()


def test_features_preserve_observation_and_add_causal_history_magnitudes():
    context = _context()
    features = build_training_observation(context)
    peaks = torch.tensor(LOCALITY_DEMAND_WEIGHTS) * (700 / sum(LOCALITY_DEMAND_WEIGHTS))
    assert features.shape == (8, 46)
    torch.testing.assert_close(features[:, :32], context.observation, atol=0, rtol=0)
    torch.testing.assert_close(features[:, 32:39], context.cumulative_demand_mwh / (24 * peaks))
    torch.testing.assert_close(features[:, 39:], context.cumulative_served_mwh / (24 * peaks))
    different_history = replace(
        context,
        cumulative_demand_mwh=context.cumulative_demand_mwh * 2,
        cumulative_served_mwh=context.cumulative_served_mwh * 2,
    )
    assert not torch.equal(features, build_training_observation(different_history))
    assert not torch.is_inference(features)


def test_context_validation_rejects_missing_or_corrupted_history():
    incomplete = DecisionContext(hour=0, observation=torch.zeros(1, 32))
    for helper in (build_training_observation, critical_action_mask):
        with pytest.raises(ValueError, match="exact causal"):
            helper(incomplete)
    context = _context()
    context.current_demand_mw[0, 0, 0] = torch.nan
    for helper in (build_training_observation, critical_action_mask):
        with pytest.raises(ValueError, match="finite"):
            helper(context)


def test_masked_distribution_blocks_invalid_actions_and_gradients():
    logits = torch.tensor([[0.0, 1000.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    distribution = masked_categorical(logits, mask)
    torch.testing.assert_close(distribution.probs, torch.tensor([[0.5, 0.0, 0.5]]))
    torch.testing.assert_close(
        distribution.entropy(), torch.tensor([np.log(2)], dtype=torch.float32)
    )
    loss = -distribution.log_prob(torch.tensor([2])).mean()
    loss.backward()
    assert logits.grad[0, 1] == 0
    assert torch.isfinite(logits.grad).all()
    singleton = masked_categorical(logits.detach(), torch.tensor([[False, True, False]]))
    assert singleton.entropy().item() == 0
    assert singleton.log_prob(torch.tensor([1])).item() == 0


@pytest.mark.parametrize("case", ["empty", "nonfinite", "dtype", "shape"])
def test_distribution_rejects_invalid_masks_and_logits(case):
    logits = torch.zeros(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    if case == "empty":
        mask[0] = False
    elif case == "nonfinite":
        logits[0, 1] = torch.nan
    elif case == "dtype":
        mask = mask.float()
    else:
        mask = mask[:, :2]
    with pytest.raises((TypeError, ValueError)):
        masked_categorical(logits, mask)


def test_actor_sampling_log_probs_determinism_and_backpropagation():
    torch.manual_seed(99301)
    model = ConstrainedActorCritic(device="cpu")
    context = _context()
    features = build_training_observation(context)
    mask = critical_action_mask(context)
    generator = torch.Generator().manual_seed(31)
    indices, log_prob, entropy, value = model.get_action_and_value(
        features, mask, generator=generator
    )
    assert indices.shape == log_prob.shape == entropy.shape == value.shape == (8,)
    assert mask.gather(1, indices[:, None]).all()
    _, repeated_log_prob, _, _ = model.get_action_and_value(features, mask, action=indices)
    torch.testing.assert_close(repeated_log_prob, log_prob, atol=0, rtol=0)
    (-log_prob.mean() + value.square().mean() - entropy.mean() * 0.01).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    deterministic, *_ = model.get_action_and_value(features, mask, deterministic=True)
    assert deterministic.tolist() == [0] * 8
    controller = ConstrainedPPOController(model)
    assert controller.name is ControllerName.PPO
    assert controller.act(context).tolist() == [[1] * 7] * 8


def test_action_evaluation_rejects_an_action_outside_rollout_mask():
    model = ConstrainedActorCritic(device="cpu")
    mask = torch.zeros(2, 128, dtype=torch.bool)
    mask[:, 0] = True
    with pytest.raises(ValueError, match="eligible"):
        model.get_action_and_value(torch.zeros(2, 46), mask, action=torch.ones(2, dtype=torch.long))


@pytest.mark.parametrize("gamma,gae_lambda", [(1.0, 1.0), (1.0, 0.95), (0.9, 0.7)])
def test_gae_matches_terminal_scalar_recurrence(gamma, gae_lambda):
    rewards = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    values = torch.tensor([[0.4, 0.1], [0.3, 0.9], [0.2, 0.8]])
    advantages, returns = generalized_advantage_estimate(
        rewards, values, gamma=gamma, gae_lambda=gae_lambda
    )
    for row in range(2):
        future_advantage = future_value = 0.0
        for hour in reversed(range(3)):
            delta = rewards[hour, row].item() + gamma * future_value - values[hour, row].item()
            expected = delta + gamma * gae_lambda * future_advantage
            assert advantages[hour, row].item() == pytest.approx(expected, abs=2e-6)
            assert returns[hour, row].item() == pytest.approx(
                expected + values[hour, row], abs=2e-6
            )
            future_advantage = expected
            future_value = values[hour, row].item()


def test_ppo_clipping_matches_scalar_objective_and_finite_gradients():
    ratio = torch.tensor([1.5, 0.5, 1.1, 0.9])
    new_log_prob = ratio.log().requires_grad_()
    advantages = torch.tensor([1.0, -1.0, -1.0, 1.0])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    returns = torch.tensor([2.0, 2.0, 2.0, 2.0])
    entropy = torch.tensor([0.5, 1.0, 1.5, 2.0])
    loss = clipped_ppo_loss(
        new_log_prob,
        torch.zeros(4),
        advantages,
        values,
        returns,
        entropy,
        clip_ratio=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
    )
    expected_policy = -sum([1.2, -0.8, -1.1, 0.9]) / 4
    expected_value = 0.5 * sum([1, 0, 1, 4]) / 4
    assert loss.policy.item() == pytest.approx(expected_policy)
    assert loss.value.item() == pytest.approx(expected_value)
    assert loss.total.item() == pytest.approx(expected_policy + 0.5 * expected_value - 0.0125)
    assert loss.clip_fraction.item() == 0.5
    assert loss.approx_kl.item() >= 0
    loss.total.backward()
    assert torch.isfinite(new_log_prob.grad).all()
    assert torch.isfinite(values.grad).all()


def test_safe_checkpoint_roundtrip_freeze_hash_and_no_overwrite(tmp_path):
    model = ConstrainedActorCritic(device="cpu")
    path = tmp_path / "checkpoint.npz"
    save_model(model, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_model(path, device="cpu", expected_sha256=digest)
    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    mutable = load_model(path, device="cpu", freeze=False)
    assert all(p.requires_grad for p in mutable.parameters())
    with pytest.raises(ValueError, match="SHA-256"):
        load_model(path, device="cpu", expected_sha256="0" * 64)
    with pytest.raises(FileExistsError):
        save_model(model, path)


@pytest.mark.parametrize("corruption", ["extra", "missing", "nan", "shape", "dtype"])
def test_safe_checkpoint_rejects_malformed_arrays(tmp_path, corruption):
    model = ConstrainedActorCritic(device="cpu")
    path = tmp_path / "checkpoint.npz"
    save_model(model, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    key = "actor.0.weight"
    if corruption == "extra":
        arrays["surprise"] = np.ones(1)
    elif corruption == "missing":
        arrays.pop(key)
    elif corruption == "nan":
        arrays[key][0, 0] = np.nan
    elif corruption == "shape":
        arrays[key] = arrays[key][:1]
    else:
        arrays[key] = arrays[key].astype(np.float64)
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_model(path, device="cpu")


@pytest.mark.gpu
def test_cuda_mask_parity_and_training_tensor_device():
    assert torch.cuda.is_available(), "CUDA is required for the constrained PPO training check"
    context = _context(device="cuda:0")
    expected = eligible_candidates(evaluate_constrained_candidates(context), None)
    mask = critical_action_mask(context)
    torch.testing.assert_close(mask, expected, atol=0, rtol=0)
    model = ConstrainedActorCritic(device="cuda:0")
    features = build_training_observation(context)
    _, log_prob, entropy, value = model.get_action_and_value(features, mask)
    assert features.device == mask.device == log_prob.device == context.observation.device
    (-log_prob.mean() + value.square().mean() - 0.01 * entropy.mean()).backward()
    assert all(p.grad is not None and p.grad.device.type == "cuda" for p in model.parameters())
