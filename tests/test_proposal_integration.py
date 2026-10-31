"""Causal dispatch-state integration for the additional proposal comparator."""

from dataclasses import dataclass, field

import pytest
import torch

from gridguard.controllers import DecisionContext
from gridguard.domain import ControllerName, Scenario, ShortageLevel
from gridguard.simulation import simulate


@dataclass
class StateProbe:
    name = ControllerName.FIXED_ROTATION
    contexts: list[DecisionContext] = field(default_factory=list)

    def act(self, context):
        self.contexts.append(context)
        return torch.full((1, 7), 2, dtype=torch.long)


def test_simulation_supplies_only_current_and_past_dispatch_state():
    demand = torch.ones((1, 24, 7, 5))
    demand[:, 1:] *= 2
    scenario = Scenario(demand, demand.sum((-1, -2)) * 0.5, ShortageLevel.MODERATE, 91234)
    probe = StateProbe()
    result = simulate(scenario, probe)
    first, second = probe.contexts[:2]
    torch.testing.assert_close(first.current_demand_mw, demand[:, 0])
    torch.testing.assert_close(first.cumulative_demand_mwh, torch.zeros((1, 7)))
    torch.testing.assert_close(first.cumulative_served_mwh, torch.zeros((1, 7)))
    torch.testing.assert_close(second.current_demand_mw, demand[:, 1])
    torch.testing.assert_close(second.cumulative_demand_mwh, demand[:, 0].sum(-1))
    torch.testing.assert_close(second.cumulative_served_mwh, result.served[:, 0].sum(-1))
    assert second.previous_service_tiers.tolist() == [[2] * 7]
    assert second.current_demand_mw.shape == (1, 7, 5)


def test_partial_dispatch_context_is_rejected():
    with pytest.raises(ValueError, match="all dispatch-state"):
        DecisionContext(0, torch.zeros((1, 32)), current_supply_mw=torch.ones(1))


def test_optimization_is_a_distinct_controller_name():
    assert ControllerName("Optimization") is ControllerName.OPTIMIZATION
