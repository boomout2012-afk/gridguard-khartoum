from __future__ import annotations

import pytest
import torch

from gridguard.controllers import DecisionContext, FixedRotationController, PPOController
from gridguard.domain import ControllerName

EXPECTED_FIXED_ROTATION = torch.tensor(
    [
        [2, 2, 2, 0, 0, 0, 0],
        [2, 2, 2, 0, 0, 0, 0],
        [2, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 0],
        [0, 0, 0, 2, 2, 2, 0],
        [0, 0, 0, 2, 2, 2, 0],
        [2, 2, 0, 0, 0, 0, 2],
        [2, 2, 0, 0, 0, 0, 2],
        [2, 2, 0, 0, 0, 0, 2],
        [0, 0, 2, 2, 2, 0, 0],
        [0, 0, 2, 2, 2, 0, 0],
        [0, 0, 2, 2, 2, 0, 0],
        [2, 0, 0, 0, 0, 2, 2],
        [2, 0, 0, 0, 0, 2, 2],
        [2, 0, 0, 0, 0, 2, 2],
        [0, 2, 2, 2, 0, 0, 0],
        [0, 2, 2, 2, 0, 0, 0],
        [0, 2, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 2],
        [0, 0, 0, 0, 2, 2, 2],
        [0, 0, 0, 0, 2, 2, 2],
        [2, 2, 2, 0, 0, 0, 0],
        [2, 2, 2, 0, 0, 0, 0],
        [2, 2, 2, 0, 0, 0, 0],
    ],
    dtype=torch.long,
)


def test_fixed_rotation_reproduces_the_accepted_24_hour_schedule() -> None:
    controller = FixedRotationController()
    observations = torch.zeros((2, 32), dtype=torch.float32)

    actions = torch.stack(
        [
            controller.act(DecisionContext(hour=hour, observation=observations))
            for hour in range(24)
        ],
        dim=1,
    )

    assert controller.name is ControllerName.FIXED_ROTATION
    assert actions.dtype is torch.long
    assert actions.device == observations.device
    torch.testing.assert_close(actions[0], EXPECTED_FIXED_ROTATION, rtol=0, atol=0)
    torch.testing.assert_close(actions[1], EXPECTED_FIXED_ROTATION, rtol=0, atol=0)


def test_decision_context_rejects_a_non_32_wide_observation() -> None:
    observation = torch.zeros((2, 31), dtype=torch.float32)

    try:
        DecisionContext(hour=0, observation=observation)
    except ValueError as error:
        assert str(error) == "observation must have shape [batch, 32]"
    else:
        raise AssertionError("invalid observations must be rejected")


def test_decision_context_rejects_nonfinite_observations() -> None:
    observation = torch.zeros((1, 32), dtype=torch.float32)
    observation[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        DecisionContext(hour=0, observation=observation)


class _FakePPOPolicy:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action
        self.received_observation: torch.Tensor | None = None

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        self.received_observation = observation
        return self.action


class _PrevalidatedFakePPOPolicy(_FakePPOPolicy):
    def __init__(self, action: torch.Tensor) -> None:
        super().__init__(action)
        self.used_prevalidated_path = False

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        raise AssertionError("validated controller contexts must use the synchronization-free path")

    def act_prevalidated(self, observation: torch.Tensor) -> torch.Tensor:
        self.used_prevalidated_path = True
        self.received_observation = observation
        return self.action


def test_ppo_controller_is_an_explicit_adapter_without_loading_an_artifact() -> None:
    observation = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    expected_action = torch.tensor([[0, 1, 2, 0, 1, 2, 0], [2, 1, 0, 2, 1, 0, 2]])
    policy = _FakePPOPolicy(expected_action)
    controller = PPOController(policy=policy)

    action = controller.act(DecisionContext(hour=9, observation=observation))

    assert controller.name is ControllerName.PPO
    assert policy.received_observation is observation
    assert action is expected_action


def test_ppo_controller_uses_prevalidated_policy_path_when_available() -> None:
    observation = torch.zeros((2, 32), dtype=torch.float32)
    expected_action = torch.zeros((2, 7), dtype=torch.long)
    policy = _PrevalidatedFakePPOPolicy(expected_action)

    action = PPOController(policy=policy).act(DecisionContext(hour=4, observation=observation))

    assert action is expected_action
    assert policy.used_prevalidated_path is True
    assert policy.received_observation is observation
