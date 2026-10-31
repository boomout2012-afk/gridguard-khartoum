from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from gridguard.domain import canonical_step_reward
from gridguard.scenarios import _sector_profiles
from tools.training_provenance import (
    TrainingProvenanceError,
    build_training_summary,
    load_training_provenance,
    parse_training_provenance,
    render_training_provenance,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = REPOSITORY_ROOT / "data" / "model" / "training-provenance.yaml"
MODEL_MANIFEST_PATH = REPOSITORY_ROOT / "data" / "model" / "manifest.json"
SOURCE_REVISION = "52f6dd82605c993c8094448caecec86faa289c0c"
SOURCE_BINDINGS = (
    (
        "artifacts/training_manifest.json",
        "selected-run-training-manifest",
        "3b866b6f470e116ff009022032059a1f91b509fadc54dbff6c5962a8f2236a59",
    ),
    (
        "artifacts/training_metrics.jsonl",
        "selected-run-training-metrics",
        "f48d2f6da29682e1c47330aa2b33b6bb4941f4e0e248f2c846436f09a4b89bc4",
    ),
    (
        "experiments/s2-600-hidden512.json",
        "candidate-experiment-summary",
        "619eb67aa253a35d7a7dc0efdbc6c743d7b0af6db4687575e7fade9dbcfe940b",
    ),
    (
        "src/aipower_rl/config.py",
        "simulation-configuration-source",
        "5f1f9febaaf7cdae980ed67c83948362b378cd290118f5712f47da1a09d65d04",
    ),
    (
        "src/aipower_rl/scenarios.py",
        "training-scenario-generator-source",
        "689bb4319194baf2f593d98c4b3309d843167e039ed48fff20ee0a9b14aef186",
    ),
    (
        "src/aipower_rl/models.py",
        "actor-critic-architecture-source",
        "6fedcc55675a72aff048a3b1a9a98c86a87962427031c6a4450239fdccbbde03",
    ),
    (
        "src/aipower_rl/training.py",
        "training-orchestration-source",
        "886931d6c12b0ef45f4f17b89d134e27682c68a2dc4c2f90e2911e2dfec708fb",
    ),
    (
        "src/aipower_rl/learning.py",
        "learning-algorithm-source",
        "18cf22b0b58584b2abef164ec3ea956fa2f61197336a67a7b9c1f4faefc4fac2",
    ),
    (
        "src/aipower_rl/experiments.py",
        "experiment-selection-source",
        "605e32626caece41508f37cbfac0cb061f7510118ba463cf3ff5843f92a9d753",
    ),
)
SOURCE_HASHES = {path: digest for path, _, digest in SOURCE_BINDINGS}
EXPECTED_SECTOR_PROFILES = [
    {"sector": "Hospital", "baseline_amplitude": 0.95, "gaussian_terms": []},
    {
        "sector": "Water",
        "baseline_amplitude": 0.45,
        "gaussian_terms": [
            {"amplitude": 0.35, "center_hour": 6.0, "width_hours": 2.0},
            {"amplitude": 0.40, "center_hour": 18.0, "width_hours": 2.5},
        ],
    },
    {
        "sector": "Bakery",
        "baseline_amplitude": 0.35,
        "gaussian_terms": [
            {"amplitude": 0.60, "center_hour": 5.0, "width_hours": 1.5},
            {"amplitude": 0.45, "center_hour": 16.0, "width_hours": 2.0},
        ],
    },
    {
        "sector": "Residential",
        "baseline_amplitude": 0.40,
        "gaussian_terms": [
            {"amplitude": 0.30, "center_hour": 7.0, "width_hours": 2.0},
            {"amplitude": 0.65, "center_hour": 20.0, "width_hours": 3.0},
        ],
    },
    {
        "sector": "Commercial",
        "baseline_amplitude": 0.25,
        "gaussian_terms": [{"amplitude": 0.75, "center_hour": 13.0, "width_hours": 4.0}],
    },
]
SECTOR_PROFILE_NORMALIZATION = "divide-each-sector-by-its-24-hour-maximum"
EXPECTED_MODEL_INITIALIZATION = {
    "encoder_linear_layers": {
        "applies_to": ["actor_encoder", "critic_encoder"],
        "weight_initializer": "orthogonal_",
        "weight_gain_expression": "2 ** 0.5",
        "weight_gain": 2**0.5,
        "bias_initializer": "zeros_",
    },
    "actor_head": {
        "parameter_name": "actor",
        "weight_initializer": "normal_",
        "weight_standard_deviation": 0.01,
        "bias_initializer": "zeros_",
    },
    "critic_head": {
        "parameter_name": "critic",
        "weight_initializer": "orthogonal_",
        "weight_gain": 1.0,
        "bias_initializer": "zeros_",
    },
}
EXPECTED_PPO_LEARNING_SEMANTICS = {
    "advantage_normalization": {
        "mean_centering": "subtract-batch-mean",
        "standard_deviation": "population",
        "standard_deviation_unbiased": False,
        "denominator_clamp_min": 1e-8,
    },
    "policy_loss": {
        "probability_ratio": "exponentiated-new-minus-old-log-probability",
        "unclipped_surrogate": "probability-ratio-times-normalized-advantage",
        "clipped_ratio_lower_bound": "1.0-minus-clip-epsilon",
        "clipped_ratio_upper_bound": "1.0-plus-clip-epsilon",
        "clipped_surrogate": "clipped-probability-ratio-times-normalized-advantage",
        "surrogate_selection": "elementwise-minimum",
        "reduction": "negative-mean",
    },
    "value_loss": {
        "function": "mean-squared-error",
        "value_clipping": False,
    },
    "completed_episode_bootstrap": {
        "terminal_next_value": 0.0,
        "gae_nonterminal_mask": "logical-not-terminated",
    },
}
EXPECTED_BEHAVIOR_CLONING_OPTIMIZER = {
    "name": "torch.optim.AdamW",
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
    "decoupled_weight_decay": True,
}
EXPECTED_PPO_OPTIMIZER = {
    "name": "torch.optim.Adam",
    "learning_rate": 0.0001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
    "decoupled_weight_decay": False,
}
EXPECTED_GRADIENT_CLIP_DEFAULTS = {
    "norm_type": 2.0,
    "error_if_nonfinite": False,
    "foreach": None,
}
EXPECTED_OBJECTIVE = {
    "public_role": "training-objective-diagnostic",
    "formula": "(5*C + 2*U + 2*J - 0.25*S - 3*O) / 9",
    "episode_aggregation": "sum-of-24-hour-rewards",
    "component_definitions": {
        "C": {
            "name": "critical-service-ratio",
            "numerator": "critical-sector-served-power",
            "denominator": "max(critical-sector-demand-power, 1e-8)",
            "sectors": ["Hospital", "Water", "Bakery"],
        },
        "U": {
            "name": "supply-utilization-ratio",
            "numerator": "total-served-power",
            "denominator": "max(min(available-supply-power, total-demand-power), 1e-8)",
        },
        "J": {
            "name": "jain-fairness-of-cumulative-locality-service-ratios",
            "locality_service_ratio": (
                "cumulative-served-energy / max(cumulative-demand-energy, 1e-8)"
            ),
            "aggregation": "square(sum(x_i)) / (7*sum(square(x_i)) + 1e-8)",
            "history": "through-current-hour",
        },
        "S": {
            "name": "service-tier-switching-rate",
            "definition": "fraction-of-localities-whose-service-tier-changed",
            "first_hour_value": 0.0,
        },
        "O": {
            "name": "requested-power-oversubscription-ratio",
            "numerator": "max(total-requested-power - available-supply-power, 0)",
            "denominator": "max(total-demand-power, 1e-8)",
        },
    },
    "components": {
        "critical_service_weight": 5.0,
        "utilization_weight": 2.0,
        "jain_fairness_weight": 2.0,
        "switching_penalty_weight": 0.25,
        "oversubscription_penalty_weight": 3.0,
        "scale": 9.0,
    },
}

EXPECTED_PER_RUN_WORKLOAD = {
    "ppo_iterations": 600,
    "complete_episodes": 153_600,
    "transitions": 3_686_400,
}
EXPECTED_AGGREGATE_EXPERIMENT_WORKLOAD = {
    "ppo_iterations": 1_800,
    "complete_episodes": 460_800,
    "transitions": 11_059_200,
}
EXPECTED_PROMOTION_CHECKS = {
    "all_runs_safe": True,
    "all_performance_finite": True,
    "reward_improvement": 0.015751803362807237,
    "mean_reward_margin": True,
    "matched_seed_wins": 3,
    "two_of_three_seeds_improve": True,
    "worst_seed_regression": 0.007786220974391966,
    "no_large_seed_regression": True,
    "throughput_ratio": 1.2173495404586454,
    "throughput_within_25_percent": True,
}
EXPECTED_PROMOTION_THRESHOLDS = {
    "mean_reward_margin_minimum": 0.005,
    "matched_seed_wins_minimum": 2,
    "maximum_seed_regression": 0.02,
    "minimum_throughput_ratio": 0.75,
    "minimum_critical_service_ratio": 0.99,
    "maximum_oversubscription_ratio": 0.005,
    "required_power_balance_violation_rate": 0.0,
}
EXPECTED_TAXONOMY_CATEGORIES = [
    "directly-recorded-archive-fact",
    "bound-source-or-pytorch-default-reconstruction",
    "arithmetic-derived-value",
    "unavailable-historical-environment-fact",
]
EXPECTED_HISTORICAL_ENVIRONMENT_UNKNOWNS = [
    "python-interpreter-version",
    "operating-system-edition-and-build",
    "nvidia-driver-version",
    "cudnn-version-and-runtime-flags",
    "cpu-model-and-topology",
    "installed-system-memory",
    "complete-python-environment-lock-and-wheel-hashes",
    "training-process-environment-variables",
    "gpu-clock-power-and-thermal-state",
]


def _canonical_text() -> str:
    return PROVENANCE_PATH.read_text(encoding="utf-8")


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1
    return text.replace(old, new, 1)


def _mutated_text(mutator: Callable[[dict[str, Any]], None]) -> str:
    payload = yaml.safe_load(_canonical_text())
    assert isinstance(payload, dict)
    mutator(payload)
    return yaml.safe_dump(payload, sort_keys=False)


def _number_literal(node: ast.expr) -> float:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return _number_literal(node.left) ** _number_literal(node.right)
    value = ast.literal_eval(node)
    assert type(value) in (int, float)
    return float(value)


def _flatten_addition(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_flatten_addition(node.left), *_flatten_addition(node.right)]
    return [node]


def _sector_profile_specs_from_source(source: str) -> list[dict[str, Any]]:
    tree = ast.parse(textwrap.dedent(source))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_sector_profiles"
    )
    assignments = {
        node.targets[0].id: node.value
        for node in function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    stack_assignment = assignments["profiles"]
    assert isinstance(stack_assignment, ast.Call)
    assert isinstance(stack_assignment.func, ast.Attribute)
    assert stack_assignment.func.attr == "stack"
    stack_tuple = stack_assignment.args[0]
    assert isinstance(stack_tuple, (ast.Tuple, ast.List))
    profile_order = [item.id for item in stack_tuple.elts if isinstance(item, ast.Name)]
    assert len(profile_order) == len(stack_tuple.elts)

    specs: list[dict[str, Any]] = []
    for variable_name in profile_order:
        expression = assignments[variable_name]
        if isinstance(expression, ast.Call):
            assert isinstance(expression.func, ast.Attribute)
            assert expression.func.attr == "full_like"
            baseline = _number_literal(expression.args[1])
            gaussian_nodes: list[ast.expr] = []
        else:
            terms = _flatten_addition(expression)
            baseline = _number_literal(terms[0])
            gaussian_nodes = terms[1:]

        gaussian_terms: list[dict[str, float]] = []
        for gaussian_node in gaussian_nodes:
            assert isinstance(gaussian_node, ast.BinOp)
            assert isinstance(gaussian_node.op, ast.Mult)
            amplitude = _number_literal(gaussian_node.left)
            call = gaussian_node.right
            assert isinstance(call, ast.Call)
            assert isinstance(call.func, ast.Name)
            assert call.func.id == "_gaussian"
            gaussian_terms.append(
                {
                    "amplitude": amplitude,
                    "center_hour": _number_literal(call.args[1]),
                    "width_hours": _number_literal(call.args[2]),
                }
            )
        specs.append(
            {
                "sector": variable_name.replace("_", " ").title(),
                "baseline_amplitude": baseline,
                "gaussian_terms": gaussian_terms,
            }
        )
    return specs


def _profiles_from_specs(specs: list[dict[str, Any]]) -> torch.Tensor:
    hours = torch.arange(24, dtype=torch.float32)
    profiles = []
    for spec in specs:
        profile = torch.full_like(hours, float(spec["baseline_amplitude"]))
        for term in spec["gaussian_terms"]:
            center = float(term["center_hour"])
            width = float(term["width_hours"])
            gaussian = torch.exp(-0.5 * ((hours - center) / width).square())
            profile = profile + float(term["amplitude"]) * gaussian
        profiles.append(profile)
    stacked = torch.stack(profiles, dim=-1)
    return stacked / stacked.amax(dim=0, keepdim=True)


def _module_literal(source: str, name: str) -> Any:
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing module assignment {name}")


def _class_defaults(source: str, class_name: str) -> dict[str, Any]:
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.target.id: ast.literal_eval(node.value)
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
        and not isinstance(node.value, ast.Call)
    }


def _method_self_assignments(
    source: str,
    *,
    class_name: str,
    method_name: str,
) -> dict[str, ast.expr]:
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return {
        target.attr: node.value
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }


def _model_initialization_from_source(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ActorCriticPolicy"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_initialize"
    )
    encoder_loop = next(
        node
        for node in method.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "encoder"
    )
    assert isinstance(encoder_loop.iter, ast.Tuple)
    encoder_targets = []
    for node in encoder_loop.iter.elts:
        assert isinstance(node, ast.Attribute)
        assert isinstance(node.value, ast.Name) and node.value.id == "self"
        encoder_targets.append(node.attr)

    init_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"orthogonal_", "normal_", "zeros_"}
    ]

    def source_call(initializer: str, target: str) -> ast.Call:
        matches = [
            call
            for call in init_calls
            if call.func.attr == initializer and ast.unparse(call.args[0]) == target
        ]
        assert len(matches) == 1
        return matches[0]

    def keyword_number(call: ast.Call, name: str) -> float:
        keyword = next(keyword for keyword in call.keywords if keyword.arg == name)
        return _number_literal(keyword.value)

    encoder_weight_call = source_call("orthogonal_", "layer.weight")
    source_call("zeros_", "layer.bias")
    actor_weight_call = source_call("normal_", "self.actor.weight")
    source_call("zeros_", "self.actor.bias")
    critic_weight_call = source_call("orthogonal_", "self.critic.weight")
    source_call("zeros_", "self.critic.bias")
    encoder_gain = next(
        keyword.value for keyword in encoder_weight_call.keywords if keyword.arg == "gain"
    )
    return {
        "encoder_linear_layers": {
            "applies_to": encoder_targets,
            "weight_initializer": encoder_weight_call.func.attr,
            "weight_gain_expression": ast.unparse(encoder_gain),
            "weight_gain": _number_literal(encoder_gain),
            "bias_initializer": "zeros_",
        },
        "actor_head": {
            "parameter_name": "actor",
            "weight_initializer": actor_weight_call.func.attr,
            "weight_standard_deviation": keyword_number(actor_weight_call, "std"),
            "bias_initializer": "zeros_",
        },
        "critic_head": {
            "parameter_name": "critic",
            "weight_initializer": critic_weight_call.func.attr,
            "weight_gain": keyword_number(critic_weight_call, "gain"),
            "bias_initializer": "zeros_",
        },
    }


def _ppo_learning_semantics_from_source(source: str) -> dict[str, Any]:
    tree = ast.parse(source)

    def function(name: str) -> ast.FunctionDef:
        return next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def assignments(method: ast.FunctionDef, name: str) -> list[ast.expr]:
        return [
            node.value
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]

    ppo_update = function("ppo_update")
    advantage = assignments(ppo_update, "advantage")[-1]
    assert isinstance(advantage, ast.BinOp) and isinstance(advantage.op, ast.Div)
    centered = advantage.left
    assert isinstance(centered, ast.BinOp) and isinstance(centered.op, ast.Sub)
    assert ast.unparse(centered.left) == "advantage"
    assert ast.unparse(centered.right) == "advantage.mean()"
    denominator = advantage.right
    assert isinstance(denominator, ast.Call)
    assert isinstance(denominator.func, ast.Attribute)
    assert denominator.func.attr == "clamp_min"
    denominator_clamp_min = _number_literal(denominator.args[0])
    standard_deviation = denominator.func.value
    assert isinstance(standard_deviation, ast.Call)
    assert isinstance(standard_deviation.func, ast.Attribute)
    assert standard_deviation.func.attr == "std"
    assert ast.unparse(standard_deviation.func.value) == "advantage"
    unbiased_node = next(
        keyword.value for keyword in standard_deviation.keywords if keyword.arg == "unbiased"
    )
    standard_deviation_unbiased = ast.literal_eval(unbiased_node)
    assert type(standard_deviation_unbiased) is bool

    log_ratio = assignments(ppo_update, "log_ratio")[0]
    assert isinstance(log_ratio, ast.BinOp) and isinstance(log_ratio.op, ast.Sub)
    assert ast.unparse(log_ratio.left) == "new_log_prob"
    assert ast.unparse(log_ratio.right) == "old_log_prob[indices]"
    ratio = assignments(ppo_update, "ratio")[0]
    assert isinstance(ratio, ast.Call)
    assert isinstance(ratio.func, ast.Attribute) and ratio.func.attr == "exp"
    assert ast.unparse(ratio.func.value) == "log_ratio"

    unclipped = assignments(ppo_update, "unclipped")[0]
    assert isinstance(unclipped, ast.BinOp) and isinstance(unclipped.op, ast.Mult)
    assert ast.unparse(unclipped.left) == "ratio"
    assert ast.unparse(unclipped.right) == "advantage[indices]"
    clipped = assignments(ppo_update, "clipped")[0]
    assert isinstance(clipped, ast.BinOp) and isinstance(clipped.op, ast.Mult)
    assert ast.unparse(clipped.right) == "advantage[indices]"
    ratio_clamp = clipped.left
    assert isinstance(ratio_clamp, ast.Call)
    assert isinstance(ratio_clamp.func, ast.Attribute) and ratio_clamp.func.attr == "clamp"
    assert ast.unparse(ratio_clamp.func.value) == "ratio"
    lower_bound, upper_bound = ratio_clamp.args
    assert isinstance(lower_bound, ast.BinOp) and isinstance(lower_bound.op, ast.Sub)
    assert isinstance(upper_bound, ast.BinOp) and isinstance(upper_bound.op, ast.Add)
    assert ast.unparse(lower_bound.right) == "config.clip_epsilon"
    assert ast.unparse(upper_bound.right) == "config.clip_epsilon"
    lower_base = _number_literal(lower_bound.left)
    upper_base = _number_literal(upper_bound.left)

    policy_loss = assignments(ppo_update, "policy_loss")[0]
    assert isinstance(policy_loss, ast.UnaryOp) and isinstance(policy_loss.op, ast.USub)
    mean_call = policy_loss.operand
    assert isinstance(mean_call, ast.Call)
    assert isinstance(mean_call.func, ast.Attribute) and mean_call.func.attr == "mean"
    minimum_call = mean_call.func.value
    assert isinstance(minimum_call, ast.Call)
    assert ast.unparse(minimum_call.func) == "torch.minimum"
    assert [ast.unparse(argument) for argument in minimum_call.args] == ["unclipped", "clipped"]

    value_loss = assignments(ppo_update, "value_loss")[0]
    assert isinstance(value_loss, ast.Call)
    assert ast.unparse(value_loss.func) == "F.mse_loss"
    assert [ast.unparse(argument) for argument in value_loss.args] == [
        "new_value",
        "returns[indices]",
    ]
    new_value_sources = [
        node.value
        for node in ast.walk(ppo_update)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Tuple)
            and any(isinstance(item, ast.Name) and item.id == "new_value" for item in target.elts)
            for target in node.targets
        )
    ]
    assert len(new_value_sources) == 1
    assert ast.unparse(new_value_sources[0].func) == "policy.evaluate_actions"

    collect_rollout = function("collect_rollout")
    full_horizon_loop = next(node for node in collect_rollout.body if isinstance(node, ast.For))
    assert ast.unparse(full_horizon_loop.iter) == "range(simulator.config.horizon)"
    next_value = assignments(collect_rollout, "next_value")[0]
    assert isinstance(next_value, ast.Call)
    assert ast.unparse(next_value.func) == "torch.zeros_like"
    assert ast.unparse(next_value.args[0]) == "value_tensor[-1]"
    gae_call = next(
        node
        for node in ast.walk(collect_rollout)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "compute_gae"
    )
    assert ast.unparse(gae_call.args[3]) == "next_value"

    compute_gae = function("compute_gae")
    nonterminal = assignments(compute_gae, "nonterminal")[0]
    assert isinstance(nonterminal, ast.Call)
    assert isinstance(nonterminal.func, ast.Attribute) and nonterminal.func.attr == "to"
    terminated_mask = nonterminal.func.value
    assert isinstance(terminated_mask, ast.UnaryOp) and isinstance(terminated_mask.op, ast.Invert)
    assert ast.unparse(terminated_mask.operand) == "terminated[index]"

    return {
        "advantage_normalization": {
            "mean_centering": "subtract-batch-mean",
            "standard_deviation": ("sample" if standard_deviation_unbiased else "population"),
            "standard_deviation_unbiased": standard_deviation_unbiased,
            "denominator_clamp_min": denominator_clamp_min,
        },
        "policy_loss": {
            "probability_ratio": "exponentiated-new-minus-old-log-probability",
            "unclipped_surrogate": "probability-ratio-times-normalized-advantage",
            "clipped_ratio_lower_bound": f"{lower_base:.1f}-minus-clip-epsilon",
            "clipped_ratio_upper_bound": f"{upper_base:.1f}-plus-clip-epsilon",
            "clipped_surrogate": "clipped-probability-ratio-times-normalized-advantage",
            "surrogate_selection": "elementwise-minimum",
            "reduction": "negative-mean",
        },
        "value_loss": {
            "function": "mean-squared-error",
            "value_clipping": False,
        },
        "completed_episode_bootstrap": {
            "terminal_next_value": 0.0,
            "gae_nonterminal_mask": "logical-not-terminated",
        },
    }


def _function_from_source(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _assigned_call(source: str, *, function_name: str, target_name: str) -> ast.Call:
    function = _function_from_source(source, function_name)
    matches = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == target_name
        and isinstance(node.value, ast.Call)
    ]
    assert len(matches) == 1
    return matches[0]


def _call_argument_nodes(call: ast.Call, implementation: Callable[..., Any]) -> dict[str, ast.expr]:
    parameter_names = list(inspect.signature(implementation).parameters)
    assert len(call.args) <= len(parameter_names)
    arguments = dict(zip(parameter_names, call.args, strict=False))
    for keyword in call.keywords:
        assert keyword.arg is not None
        assert keyword.arg not in arguments
        arguments[keyword.arg] = keyword.value
    return arguments


def _optimizer_contract_from_source_call(
    call: ast.Call,
    *,
    expected_parameter_expression: str,
    expected_learning_rate_expression: str,
    learning_rate: float,
) -> dict[str, Any]:
    class_name = ast.unparse(call.func)
    optimizer_classes = {
        "torch.optim.Adam": torch.optim.Adam,
        "torch.optim.AdamW": torch.optim.AdamW,
    }
    optimizer_class = optimizer_classes[class_name]
    arguments = _call_argument_nodes(call, optimizer_class)
    assert ast.unparse(arguments["params"]) == expected_parameter_expression
    assert ast.unparse(arguments["lr"]) == expected_learning_rate_expression

    parameter = torch.nn.Parameter(torch.zeros(()))
    effective = dict(optimizer_class([parameter]).defaults)
    for name, node in arguments.items():
        if name in {"params", "lr"}:
            continue
        effective[name] = ast.literal_eval(node)
    beta1, beta2 = effective["betas"]
    return {
        "name": class_name,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": effective["eps"],
        "weight_decay": effective["weight_decay"],
        "amsgrad": effective["amsgrad"],
        "maximize": effective["maximize"],
        "foreach": effective["foreach"],
        "capturable": effective["capturable"],
        "differentiable": effective["differentiable"],
        "fused": effective["fused"],
        "decoupled_weight_decay": effective["decoupled_weight_decay"],
    }


def _gradient_clip_calls(source: str, *, function_name: str) -> list[ast.Call]:
    function = _function_from_source(source, function_name)
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.nn.utils.clip_grad_norm_"
    ]


def _gradient_clip_defaults_from_source_call(
    call: ast.Call,
    *,
    expected_parameter_expression: str,
    expected_max_norm_expression: str,
) -> dict[str, Any]:
    arguments = _call_argument_nodes(call, torch.nn.utils.clip_grad_norm_)
    assert ast.unparse(arguments["parameters"]) == expected_parameter_expression
    assert ast.unparse(arguments["max_norm"]) == expected_max_norm_expression
    signature = inspect.signature(torch.nn.utils.clip_grad_norm_)
    return {
        name: (
            ast.literal_eval(arguments[name])
            if name in arguments
            else signature.parameters[name].default
        )
        for name in ("norm_type", "error_if_nonfinite", "foreach")
    }


def _pinned_source_repository() -> Path:
    configured_repository = os.environ.get("GRIDGUARD_SOURCE_REPOSITORY", "").strip()
    if configured_repository:
        return Path(configured_repository).expanduser().resolve()
    return (REPOSITORY_ROOT.parent / "aipower_rl").resolve()


def test_pinned_source_repository_prefers_process_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit_repository = tmp_path / "trusted-source"
    monkeypatch.setenv("GRIDGUARD_SOURCE_REPOSITORY", str(explicit_repository))

    assert _pinned_source_repository() == explicit_repository.resolve()


def _pinned_source_blobs() -> dict[str, bytes]:
    source_repository = _pinned_source_repository()
    if not (source_repository / ".git").exists():
        if os.environ.get("GRIDGUARD_SOURCE_REPOSITORY", "").strip():
            pytest.fail(
                f"configured trusted source repository is not a Git repository: {source_repository}"
            )
        pytest.skip("trusted read-only source archive is unavailable")
    source_blobs: dict[str, bytes] = {}
    for source_path, _, expected_hash in SOURCE_BINDINGS:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(source_repository),
                "show",
                f"{SOURCE_REVISION}:{source_path}",
            ],
            check=True,
            capture_output=True,
        )
        assert hashlib.sha256(completed.stdout).hexdigest() == expected_hash
        source_blobs[source_path] = completed.stdout
    return source_blobs


def test_training_provenance_exposes_exact_selected_training_contract() -> None:
    payload = load_training_provenance(
        PROVENANCE_PATH,
        model_manifest_path=MODEL_MANIFEST_PATH,
    )

    assert payload["schema"] == "gridguard.ppo_training_provenance"
    assert payload["schema_version"] == 3
    assert payload["model"]["controller"] == "PPO"
    assert payload["model"]["checkpoint"]["sha256"] == (
        "c6e5712f966b46ba18f4c69fb457b9e69f78887c0c5c0279e5ce034f40f9c665"
    )
    assert payload["model"]["actor"]["sha256"] == (
        "3e4a7ce44267dd1cf3bbde7e77abafb8f31fdea77ece63acda7be17fa3198ad5"
    )
    assert payload["record"] == {
        "purpose": "Historical parameter and lineage record for the accepted PPO actor.",
        "executable_retraining_configuration": False,
        "bitwise_retraining_claimed": False,
        "optimizer_state_included": False,
        "raw_training_runs_included": False,
    }
    assert payload["objective"] == EXPECTED_OBJECTIVE
    assert payload["model"]["network"]["hidden_layers"] == [512, 512]
    assert payload["behavior_cloning_initialization"]["optimizer"]["weight_decay"] == 0.01
    assert payload["behavior_cloning_initialization"]["optimizer_updates"] == 1600
    assert payload["ppo"]["total_episodes"] == 153_600
    assert payload["ppo"]["total_transitions"] == 3_686_400
    assert payload["ppo"]["planned_optimizer_updates"] == 7200
    assert payload["ppo"]["selected_checkpoint_optimizer_updates"] == 6120
    assert payload["validation_and_selection"]["selected_seed"] == 43
    assert payload["validation_and_selection"]["selected_iteration"] == 510
    assert payload["attribution_limitations"] == {
        "behavior_cloning_only_ablation_run": False,
        "ppo_fine_tuning_incremental_effect_isolated": False,
        "interpretation": (
            "No behavior-cloning-only ablation was run, so this comparison cannot "
            "isolate the incremental effect of PPO fine-tuning."
        ),
    }
    assert len(payload["convergence"]["validation_series"]) == 120
    assert payload["convergence"]["validation_series"][-1] == {
        "iteration": 600,
        "training_objective_score": 23.50542449951172,
    }

    summary = build_training_summary(payload, provenance_path=PROVENANCE_PATH)
    assert summary["network"] == "32 -> 512 -> 512 -> 2187"
    assert summary["behavior_cloning_initialization"] == {
        "teacher_samples": 32768,
        "epochs": 100,
        "optimizer_updates": 1600,
        "max_gradient_norm": 1.0,
        "gradient_clip_defaults": EXPECTED_GRADIENT_CLIP_DEFAULTS,
        "optimizer": EXPECTED_BEHAVIOR_CLONING_OPTIMIZER,
    }
    assert summary["selection"]["validation_seeds"] == [70000, 70001, 70002]
    assert summary["experiment_scope"]["candidate_experiment"]["per_run_workload"] == (
        EXPECTED_PER_RUN_WORKLOAD
    )
    assert (
        summary["experiment_scope"]["candidate_experiment"]["aggregate_workload"]
        == EXPECTED_AGGREGATE_EXPERIMENT_WORKLOAD
    )
    assert summary["selection"]["promotion"]["checks"] == EXPECTED_PROMOTION_CHECKS
    assert summary["provenance_taxonomy"]["historical_environment_unknowns"] == (
        EXPECTED_HISTORICAL_ENVIRONMENT_UNKNOWNS
    )
    assert summary["attribution_limitations"] == payload["attribution_limitations"]
    assert summary["record"] == payload["record"]
    assert summary["training_provenance"]["path"] == "data/model/training-provenance.yaml"
    assert (
        summary["training_provenance"]["sha256"]
        == hashlib.sha256(PROVENANCE_PATH.read_bytes()).hexdigest()
    )


def test_schema_v3_distinguishes_experiment_run_and_selected_checkpoint_scope() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)
    scope = payload["experiment_scope"]

    assert scope == {
        "candidate_experiment": {
            "name": "s2-600-hidden512",
            "candidate_training_seeds": [42, 43, 44],
            "candidate_run_count": 3,
            "per_run_workload": EXPECTED_PER_RUN_WORKLOAD,
            "aggregate_workload": EXPECTED_AGGREGATE_EXPERIMENT_WORKLOAD,
        },
        "selected_run": {
            "training_seed": 43,
            "checkpoint_iteration": 510,
            "checkpoint_optimizer_updates": 6120,
            "scoped_sections": [
                "software_hardware",
                "convergence",
                "ppo.training_seed",
                "ppo.selected_checkpoint_optimizer_updates",
            ],
        },
    }
    assert payload["software_hardware"]["scope"] == "selected-run-seed-43"
    assert payload["convergence"]["scope"] == "selected-run-seed-43"
    assert payload["ppo"]["configuration_scope"] == "each-candidate-run"
    assert payload["ppo"]["training_seed_scope"] == "selected-run-seed-43"
    assert payload["ppo"]["selected_checkpoint_scope"] == "selected-run-seed-43"


def test_schema_v3_maps_fact_taxonomy_and_lists_retraining_unknowns() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)
    taxonomy = payload["provenance_taxonomy"]

    assert taxonomy["mapping_scope"] == (
        "Historical training-parameter and experiment fields; schema metadata, documentary "
        "boundary flags, current actor-export lineage, and source-binding rows are separately "
        "validated."
    )
    assert list(taxonomy["categories"]) == EXPECTED_TAXONOMY_CATEGORIES
    for name in EXPECTED_TAXONOMY_CATEGORIES:
        assert taxonomy["categories"][name]["definition"]
        assert taxonomy["categories"][name]["mapped_fields"]
    assert taxonomy["historical_environment_unknowns"] == [
        {
            "fact": fact,
            "status": "unavailable-in-bound-archive",
        }
        for fact in EXPECTED_HISTORICAL_ENVIRONMENT_UNKNOWNS
    ]
    assert taxonomy["reproducibility_boundary"] == {
        "executable_retraining_configuration": False,
        "bitwise_retraining_claimed": False,
        "statement": (
            "The bound record supports audit of the selected actor's historical lineage, "
            "not executable or bitwise-identical retraining."
        ),
    }


def test_schema_v3_records_actual_historical_promotion_and_selection_rule() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)
    selection = payload["validation_and_selection"]
    promotion = selection["promotion"]

    assert promotion == {
        "eligible": True,
        "incumbent": "s1-600-constant",
        "eligibility_rule": "all-boolean-promotion-checks-must-be-true",
        "reward_gate_evidence_role": (
            "historical-model-selection-fact-not-independent-current-headline-evidence"
        ),
        "thresholds": EXPECTED_PROMOTION_THRESHOLDS,
        "checks": EXPECTED_PROMOTION_CHECKS,
    }
    assert selection["selection_rule"] == {
        "safety_pool": (
            "passed-and-all-gates-true-and-finite-learned-metrics-and-critical-service-"
            "at-least-0.99-and-oversubscription-at-most-0.005-and-zero-power-balance-violations"
        ),
        "fallback": "use-all-runs-if-no-run-is-safe",
        "ranking": "maximum-best-validation-score",
    }
    assert selection["selected_seed"] == 43
    assert selection["selected_iteration"] == 510
    assert payload["objective"]["public_role"] == "training-objective-diagnostic"
    assert payload["attribution_limitations"]["behavior_cloning_only_ablation_run"] is False


def test_canonical_schema_v3_yaml_is_deterministically_generated() -> None:
    canonical = _canonical_text()
    payload = parse_training_provenance(canonical)

    assert render_training_provenance(payload) == canonical
    assert render_training_provenance(parse_training_provenance(canonical)) == canonical


def test_canonical_yaml_objective_matches_runtime_reward() -> None:
    objective = load_training_provenance(
        PROVENANCE_PATH,
        model_manifest_path=MODEL_MANIFEST_PATH,
    )["objective"]
    weights = objective["components"]
    demand = torch.ones((1, 7, 5), dtype=torch.float64)
    supply = torch.tensor([20.0], dtype=torch.float64)
    requested = demand.clone()
    locality_service_ratios = torch.linspace(0.2, 0.8, 7, dtype=torch.float64).reshape(1, 7, 1)
    served = demand * locality_service_ratios
    tiers = torch.tensor([[0, 1, 2, 0, 1, 2, 0]], dtype=torch.long)
    previous_tier = torch.zeros_like(tiers)
    next_cumulative_demand = demand.sum(dim=-1)
    next_cumulative_served = served.sum(dim=-1)

    runtime_reward = canonical_step_reward(
        hour=1,
        demand=demand,
        supply=supply,
        requested=requested,
        served=served,
        tiers=tiers,
        previous_tier=previous_tier,
        next_cumulative_demand=next_cumulative_demand,
        next_cumulative_served=next_cumulative_served,
    )
    critical_service = served[..., :3].sum(dim=(-1, -2)) / demand[..., :3].sum(dim=(-1, -2))
    utilization = served.sum(dim=(-1, -2)) / torch.minimum(
        supply,
        demand.sum(dim=(-1, -2)),
    )
    service_ratio = next_cumulative_served / next_cumulative_demand
    fairness = service_ratio.sum(dim=-1).square() / (7 * service_ratio.square().sum(dim=-1) + 1e-8)
    switching = (tiers != previous_tier).to(demand.dtype).mean(dim=-1)
    oversubscription = torch.relu(requested.sum(dim=(-1, -2)) - supply) / demand.sum(dim=(-1, -2))
    expected = (
        weights["critical_service_weight"] * critical_service
        + weights["utilization_weight"] * utilization
        + weights["jain_fairness_weight"] * fairness
        - weights["switching_penalty_weight"] * switching
        - weights["oversubscription_penalty_weight"] * oversubscription
    ) / weights["scale"]
    torch.testing.assert_close(runtime_reward, expected, rtol=0.0, atol=1e-12)

    first_hour_reward = canonical_step_reward(
        hour=0,
        demand=demand,
        supply=supply,
        requested=requested,
        served=served,
        tiers=tiers,
        previous_tier=previous_tier,
        next_cumulative_demand=next_cumulative_demand,
        next_cumulative_served=next_cumulative_served,
    )
    torch.testing.assert_close(
        first_hour_reward,
        expected + weights["switching_penalty_weight"] * switching / weights["scale"],
        rtol=0.0,
        atol=1e-12,
    )


def test_optimizer_and_gradient_clipping_defaults_are_complete_and_reviewer_visible() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)
    cloning = payload["behavior_cloning_initialization"]
    ppo = payload["ppo"]

    assert cloning["optimizer"] == EXPECTED_BEHAVIOR_CLONING_OPTIMIZER
    assert cloning["gradient_clip_defaults"] == EXPECTED_GRADIENT_CLIP_DEFAULTS
    assert ppo["optimizer"] == EXPECTED_PPO_OPTIMIZER
    assert ppo["gradient_clip_defaults"] == EXPECTED_GRADIENT_CLIP_DEFAULTS

    summary = build_training_summary(payload, provenance_path=PROVENANCE_PATH)
    assert summary["behavior_cloning_initialization"]["optimizer"] == (
        EXPECTED_BEHAVIOR_CLONING_OPTIMIZER
    )
    assert summary["behavior_cloning_initialization"]["gradient_clip_defaults"] == (
        EXPECTED_GRADIENT_CLIP_DEFAULTS
    )
    assert summary["ppo"]["optimizer"] == EXPECTED_PPO_OPTIMIZER
    assert summary["ppo"]["gradient_clip_defaults"] == EXPECTED_GRADIENT_CLIP_DEFAULTS


def test_training_generator_sector_profiles_match_clean_parity_implementation() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)
    generator = payload["training_generator"]
    declared_profiles = generator["sector_profiles"]

    assert declared_profiles == EXPECTED_SECTOR_PROFILES
    assert generator["sector_profile_normalization"] == SECTOR_PROFILE_NORMALIZATION
    assert declared_profiles == _sector_profile_specs_from_source(
        inspect.getsource(_sector_profiles)
    )
    assert [row["sector"] for row in declared_profiles] == [
        row["sector"] for row in generator["sector_shares"]
    ]
    torch.testing.assert_close(
        _profiles_from_specs(declared_profiles),
        _sector_profiles(torch.device("cpu")),
        rtol=0.0,
        atol=0.0,
    )


def test_compact_summary_exposes_complete_sector_profile_parameters() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)

    summary = build_training_summary(payload, provenance_path=PROVENANCE_PATH)

    assert summary["training_generator"] == {
        "horizon_hours": 24,
        "system_peak_mw": 700.0,
        "sector_profile_normalization": SECTOR_PROFILE_NORMALIZATION,
        "sector_profiles": EXPECTED_SECTOR_PROFILES,
    }
    assert summary["source_binding_count"] == len(SOURCE_BINDINGS)


def test_model_initialization_contract_is_complete_and_reviewer_visible() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)

    assert payload["model"]["network"]["initialization"] == EXPECTED_MODEL_INITIALIZATION
    summary = build_training_summary(payload, provenance_path=PROVENANCE_PATH)
    assert summary["model_initialization"] == EXPECTED_MODEL_INITIALIZATION


def test_ppo_learning_semantics_are_complete_and_reviewer_visible() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)

    assert payload["ppo"]["advantage_normalization"] is True
    assert payload["ppo"]["learning_semantics"] == EXPECTED_PPO_LEARNING_SEMANTICS
    summary = build_training_summary(payload, provenance_path=PROVENANCE_PATH)
    assert summary["ppo"]["advantage_normalization"] is True
    assert summary["ppo"]["learning_semantics"] == EXPECTED_PPO_LEARNING_SEMANTICS


def test_research_guides_link_the_historical_training_parameter_record() -> None:
    from tests.test_document_claims import _linked_local_paths

    linked_paths = set().union(
        *(
            _linked_local_paths(REPOSITORY_ROOT / relative)
            for relative in ("SUBMISSION_README.md", "docs/REVIEWER_GUIDE.md")
        )
    )
    assert PROVENANCE_PATH.resolve() in linked_paths


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            "schema: gridguard.ppo_training_provenance\n"
            "schema: gridguard.ppo_training_provenance\n",
            "duplicate",
        ),
        (
            "schema: &schema gridguard.ppo_training_provenance\ncopy: *schema\n",
            "anchor",
        ),
        ("schema: gridguard.ppo_training_provenance\n<<: {}\n", "merge"),
        (
            "schema: gridguard.ppo_training_provenance\n---\nschema_version: 1\n",
            "single document",
        ),
        ("schema: !!python/object:builtins.object {}\n", "tag"),
    ],
)
def test_strict_loader_rejects_ambiguous_or_unsafe_yaml(payload: str, message: str) -> None:
    with pytest.raises(TrainingProvenanceError, match=message):
        parse_training_provenance(payload)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("schema_version: 3", "schema_version: true", "schema_version"),
        (
            "  learning_rate: 0.0001\n  learning_rate_schedule",
            "  learning_rate: .nan\n  learning_rate_schedule",
            "finite",
        ),
        (
            "total_transitions: 3686400",
            "total_transitions: 3686399",
            "total_transitions",
        ),
        (
            "path: artifacts/training_manifest.json",
            "path: ../training_manifest.json",
            "relative path",
        ),
        (
            "behavior_cloning_only_ablation_run: false",
            "behavior_cloning_only_ablation_run: true",
            "behavior_cloning_only_ablation_run",
        ),
        (
            "  public_role: training-objective-diagnostic",
            "  public_role: primary-causal-outcome",
            "objective.public_role",
        ),
        (
            "  formula: (5*C + 2*U + 2*J - 0.25*S - 3*O) / 9",
            "  formula: arbitrary-reviewer-misstatement",
            "objective.formula",
        ),
        (
            "  episode_aggregation: sum-of-24-hour-rewards",
            "  episode_aggregation: last-hour-only",
            "objective.episode_aggregation",
        ),
        (
            "    critical_service_weight: 5.0",
            "    critical_service_weight: 500.0",
            "objective.components",
        ),
        (
            "      denominator: max(critical-sector-demand-power, 1e-8)",
            "      denominator: critical-sector-demand-power",
            "objective.component_definitions.C",
        ),
        (
            "      history: through-current-hour",
            "      history: final-hour-only",
            "objective.component_definitions.J",
        ),
        (
            "      first_hour_value: 0.0",
            "      first_hour_value: 1.0",
            "objective.component_definitions.S",
        ),
    ],
)
def test_semantic_validator_rejects_invalid_types_arithmetic_and_paths(
    old: str,
    new: str,
    message: str,
) -> None:
    tampered = _replace_once(_canonical_text(), old, new)

    with pytest.raises(TrainingProvenanceError, match=message):
        parse_training_provenance(tampered)


def test_semantic_validator_rejects_unknown_fields() -> None:
    tampered = _canonical_text() + "unknown_review_field: true\n"

    with pytest.raises(TrainingProvenanceError, match="unknown fields"):
        parse_training_provenance(tampered)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["experiment_scope"]["candidate_experiment"][
                "per_run_workload"
            ].__setitem__("complete_episodes", 153_599),
            "per_run_workload",
        ),
        (
            lambda payload: payload["experiment_scope"]["candidate_experiment"][
                "aggregate_workload"
            ].__setitem__("transitions", 11_059_199),
            "aggregate_workload",
        ),
        (
            lambda payload: payload["experiment_scope"]["candidate_experiment"].__setitem__(
                "candidate_run_count", 2
            ),
            "candidate_run_count",
        ),
        (
            lambda payload: payload["experiment_scope"]["selected_run"].__setitem__(
                "checkpoint_iteration", 505
            ),
            "selected_run",
        ),
        (
            lambda payload: payload["software_hardware"].__setitem__("scope", "all-runs"),
            "software_hardware.scope",
        ),
        (
            lambda payload: payload["convergence"].__setitem__("scope", "all-runs"),
            "convergence.scope",
        ),
        (
            lambda payload: payload["ppo"].__setitem__(
                "selected_checkpoint_scope", "candidate-experiment"
            ),
            "selected_checkpoint_scope",
        ),
        (
            lambda payload: payload["validation_and_selection"]["promotion"]["checks"].__setitem__(
                "matched_seed_wins", 2
            ),
            "promotion.checks",
        ),
        (
            lambda payload: payload["validation_and_selection"]["promotion"][
                "thresholds"
            ].__setitem__("minimum_throughput_ratio", 0.5),
            "promotion.thresholds",
        ),
        (
            lambda payload: payload["validation_and_selection"]["promotion"].__setitem__(
                "reward_gate_evidence_role", "current-independent-headline-evidence"
            ),
            "reward_gate_evidence_role",
        ),
        (
            lambda payload: payload["validation_and_selection"]["selection_rule"].__setitem__(
                "ranking", "first-run"
            ),
            "selection_rule",
        ),
        (
            lambda payload: payload["provenance_taxonomy"].__setitem__(
                "mapping_scope", "all fields are direct archive facts"
            ),
            "mapping_scope",
        ),
        (
            lambda payload: payload["provenance_taxonomy"]["categories"].pop(
                "arithmetic-derived-value"
            ),
            "provenance_taxonomy.categories",
        ),
        (
            lambda payload: payload["provenance_taxonomy"]["categories"][
                "directly-recorded-archive-fact"
            ]["mapped_fields"].append("ppo.optimizer"),
            "mapped_fields",
        ),
        (
            lambda payload: payload["provenance_taxonomy"]["historical_environment_unknowns"][
                0
            ].__setitem__("status", "known"),
            "historical_environment_unknowns",
        ),
        (
            lambda payload: payload["provenance_taxonomy"]["reproducibility_boundary"].__setitem__(
                "bitwise_retraining_claimed", True
            ),
            "bitwise_retraining_claimed",
        ),
    ],
)
def test_schema_v3_scope_taxonomy_and_promotion_tampering_fails_closed(
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    with pytest.raises(TrainingProvenanceError, match=message):
        parse_training_provenance(_mutated_text(mutator))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["training_generator"].pop("sector_profiles"),
            "missing fields",
        ),
        (
            lambda payload: payload["training_generator"]["sector_profiles"][0].update(
                {"unreviewed": True}
            ),
            "unknown fields",
        ),
        (
            lambda payload: payload["training_generator"]["sector_profiles"][0].__setitem__(
                "baseline_amplitude", True
            ),
            "baseline_amplitude",
        ),
        (
            lambda payload: payload["training_generator"]["sector_profiles"][1]["gaussian_terms"][
                0
            ].__setitem__("center_hour", float("nan")),
            "non-finite",
        ),
        (
            lambda payload: payload["training_generator"]["sector_profiles"][1]["gaussian_terms"][
                0
            ].__setitem__("center_hour", -1.0),
            "center_hour",
        ),
        (
            lambda payload: payload["training_generator"]["sector_profiles"][1]["gaussian_terms"][
                0
            ].__setitem__("width_hours", 0.0),
            "width_hours",
        ),
        (
            lambda payload: payload["training_generator"]["sector_profiles"].__setitem__(
                4,
                copy.deepcopy(payload["training_generator"]["sector_profiles"][0]),
            ),
            "duplicate sector",
        ),
        (
            lambda payload: payload["training_generator"]["sector_profiles"].pop(),
            "all five sectors",
        ),
        (
            lambda payload: payload["training_generator"]["sector_profiles"].reverse(),
            "sector order",
        ),
    ],
)
def test_sector_profile_schema_rejects_omissions_drift_and_invalid_values(
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    with pytest.raises(TrainingProvenanceError, match=message):
        parse_training_provenance(_mutated_text(mutator))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["model"]["network"].pop("initialization"),
            "missing fields",
        ),
        (
            lambda payload: payload["model"]["network"]["initialization"].update(
                {"unreviewed": True}
            ),
            "unknown fields",
        ),
        (
            lambda payload: payload["model"]["network"]["initialization"][
                "encoder_linear_layers"
            ].__setitem__("applies_to", ["actor_encoder"]),
            "accepted historical source",
        ),
        (
            lambda payload: payload["model"]["network"]["initialization"][
                "encoder_linear_layers"
            ].__setitem__("weight_gain", True),
            "weight_gain",
        ),
        (
            lambda payload: payload["model"]["network"]["initialization"][
                "encoder_linear_layers"
            ].__setitem__("weight_gain_expression", "sqrt(2)"),
            "accepted historical source",
        ),
        (
            lambda payload: payload["model"]["network"]["initialization"]["actor_head"].__setitem__(
                "weight_standard_deviation", 0.0
            ),
            "weight_standard_deviation",
        ),
        (
            lambda payload: payload["model"]["network"]["initialization"][
                "critic_head"
            ].__setitem__("weight_gain", float("inf")),
            "non-finite",
        ),
        (
            lambda payload: payload["model"]["network"]["initialization"].pop("actor_head"),
            "missing fields",
        ),
    ],
)
def test_model_initialization_schema_rejects_omissions_unknowns_types_and_drift(
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    with pytest.raises(TrainingProvenanceError, match=message):
        parse_training_provenance(_mutated_text(mutator))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload["ppo"].pop("learning_semantics"), "missing fields"),
        (
            lambda payload: payload["ppo"]["learning_semantics"]["advantage_normalization"].update(
                {"unreviewed": True}
            ),
            "unknown fields",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"][
                "advantage_normalization"
            ].__setitem__("mean_centering", True),
            "mean_centering",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"][
                "advantage_normalization"
            ].__setitem__("standard_deviation", "sample"),
            "accepted historical source",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"][
                "advantage_normalization"
            ].__setitem__("standard_deviation_unbiased", True),
            "accepted historical source",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"][
                "advantage_normalization"
            ].__setitem__("denominator_clamp_min", True),
            "denominator_clamp_min",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"][
                "advantage_normalization"
            ].__setitem__("denominator_clamp_min", 0.0),
            "denominator_clamp_min",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"]["policy_loss"].pop("reduction"),
            "missing fields",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"]["policy_loss"].__setitem__(
                "surrogate_selection", "elementwise-maximum"
            ),
            "accepted historical source",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"]["value_loss"].__setitem__(
                "value_clipping", True
            ),
            "accepted historical source",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"]["value_loss"].__setitem__(
                "function", "smooth-l1"
            ),
            "accepted historical source",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"][
                "completed_episode_bootstrap"
            ].__setitem__("terminal_next_value", float("nan")),
            "non-finite",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"][
                "completed_episode_bootstrap"
            ].__setitem__("terminal_next_value", 1.0),
            "accepted historical source",
        ),
        (
            lambda payload: payload["ppo"]["learning_semantics"][
                "completed_episode_bootstrap"
            ].__setitem__("gae_nonterminal_mask", False),
            "gae_nonterminal_mask",
        ),
    ],
)
def test_ppo_learning_semantics_schema_rejects_omissions_types_and_drift(
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    with pytest.raises(TrainingProvenanceError, match=message):
        parse_training_provenance(_mutated_text(mutator))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["behavior_cloning_initialization"]["optimizer"].pop("maximize"),
            "missing fields",
        ),
        (
            lambda payload: payload["behavior_cloning_initialization"].__setitem__(
                "max_gradient_norm", 2.0
            ),
            "historical source",
        ),
        (
            lambda payload: payload["behavior_cloning_initialization"]["optimizer"].__setitem__(
                "name", "AdamW"
            ),
            "historical source",
        ),
        (
            lambda payload: payload["behavior_cloning_initialization"]["optimizer"].__setitem__(
                "foreach", False
            ),
            "historical source",
        ),
        (
            lambda payload: payload["behavior_cloning_initialization"]["optimizer"].__setitem__(
                "decoupled_weight_decay", False
            ),
            "historical source",
        ),
        (
            lambda payload: payload["ppo"]["optimizer"].pop("decoupled_weight_decay"),
            "missing fields",
        ),
        (
            lambda payload: payload["ppo"].__setitem__("actor_max_gradient_norm", 0.75),
            "historical source",
        ),
        (
            lambda payload: payload["ppo"]["optimizer"].__setitem__("name", "torch.optim.AdamW"),
            "historical source",
        ),
        (
            lambda payload: payload["ppo"]["optimizer"].__setitem__("decoupled_weight_decay", True),
            "historical source",
        ),
        (
            lambda payload: payload["ppo"]["optimizer"].__setitem__("learning_rate", 0.001),
            "historical source",
        ),
        (
            lambda payload: payload["behavior_cloning_initialization"][
                "gradient_clip_defaults"
            ].pop("foreach"),
            "missing fields",
        ),
        (
            lambda payload: payload["ppo"]["gradient_clip_defaults"].__setitem__(
                "error_if_nonfinite", True
            ),
            "historical source",
        ),
        (
            lambda payload: payload["ppo"]["gradient_clip_defaults"].__setitem__("norm_type", True),
            "norm_type",
        ),
    ],
)
def test_optimizer_and_gradient_clip_schema_rejects_omissions_types_and_drift(
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    with pytest.raises(TrainingProvenanceError, match=message):
        parse_training_provenance(_mutated_text(mutator))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["source_bindings"][3].__setitem__("role", "unreviewed-role"),
            "source_bindings",
        ),
        (
            lambda payload: payload["source_bindings"][3].__setitem__("sha256", "0" * 64),
            "source_bindings",
        ),
        (lambda payload: payload["source_bindings"].pop(), "source_bindings"),
        (
            lambda payload: payload["source_bindings"][0].update({"unreviewed": True}),
            "unknown fields",
        ),
    ],
)
def test_source_binding_schema_rejects_role_hash_omission_and_unknown_fields(
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    with pytest.raises(TrainingProvenanceError, match=message):
        parse_training_provenance(_mutated_text(mutator))


def test_summary_cli_prints_only_compact_validated_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/training_provenance.py",
            "--path",
            str(PROVENANCE_PATH),
            "--model-manifest",
            str(MODEL_MANIFEST_PATH),
            "--summary-json",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["network"] == "32 -> 512 -> 512 -> 2187"
    assert "\n" not in completed.stdout.rstrip("\n")


def test_summary_cli_fails_closed_without_printing_json(tmp_path: Path) -> None:
    tampered = tmp_path / "training-provenance.yaml"
    tampered.write_text(
        _replace_once(_canonical_text(), "schema_version: 3", "schema_version: true"),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "tools/training_provenance.py",
            "--path",
            str(tampered),
            "--model-manifest",
            str(MODEL_MANIFEST_PATH),
            "--summary-json",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "schema_version" in completed.stderr


@pytest.mark.source_archive
def test_declared_training_sources_match_the_pinned_read_only_archive() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)
    declared = tuple(
        (binding["path"], binding["role"], binding["sha256"])
        for binding in payload["source_bindings"]
    )

    assert declared == SOURCE_BINDINGS
    assert set(_pinned_source_blobs()) == set(SOURCE_HASHES)


@pytest.mark.source_archive
def test_optimizer_and_gradient_clipping_contracts_match_pinned_source_ast() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)
    source_blobs = _pinned_source_blobs()
    learning_source = source_blobs["src/aipower_rl/learning.py"].decode("utf-8")
    training_source = source_blobs["src/aipower_rl/training.py"].decode("utf-8")
    cloning = payload["behavior_cloning_initialization"]
    ppo = payload["ppo"]

    cloning_optimizer = _assigned_call(
        learning_source,
        function_name="fit_behavior_cloning",
        target_name="optimizer",
    )
    assert (
        _optimizer_contract_from_source_call(
            cloning_optimizer,
            expected_parameter_expression="policy.actor_parameters()",
            expected_learning_rate_expression="learning_rate",
            learning_rate=cloning["optimizer"]["learning_rate"],
        )
        == cloning["optimizer"]
    )

    ppo_optimizer = _assigned_call(
        training_source,
        function_name="train",
        target_name="optimizer",
    )
    assert (
        _optimizer_contract_from_source_call(
            ppo_optimizer,
            expected_parameter_expression="policy.parameters()",
            expected_learning_rate_expression="config.ppo.learning_rate",
            learning_rate=ppo["learning_rate"],
        )
        == ppo["optimizer"]
    )

    cloning_clips = _gradient_clip_calls(
        learning_source,
        function_name="fit_behavior_cloning",
    )
    assert len(cloning_clips) == 1
    assert (
        _gradient_clip_defaults_from_source_call(
            cloning_clips[0],
            expected_parameter_expression="policy.parameters()",
            expected_max_norm_expression="1.0",
        )
        == cloning["gradient_clip_defaults"]
    )

    ppo_clips = _gradient_clip_calls(learning_source, function_name="ppo_update")
    assert len(ppo_clips) == 2
    ppo_defaults = _class_defaults(learning_source, "PPOConfig")
    assert ppo["actor_max_gradient_norm"] == ppo_defaults["max_grad_norm"]
    assert ppo["critic_max_gradient_norm"] == ppo_defaults["max_grad_norm"]
    expected_parameters = {
        "policy.actor_parameters()",
        "policy.critic_parameters()",
    }
    observed_parameters = {ast.unparse(call.args[0]) for call in ppo_clips}
    assert observed_parameters == expected_parameters
    for clip_call in ppo_clips:
        assert (
            _gradient_clip_defaults_from_source_call(
                clip_call,
                expected_parameter_expression=ast.unparse(clip_call.args[0]),
                expected_max_norm_expression="config.max_grad_norm",
            )
            == ppo["gradient_clip_defaults"]
        )


@pytest.mark.source_archive
def test_optimizer_ast_parity_rejects_class_and_default_rewrites_independent_of_hashes() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)
    learning_source = _pinned_source_blobs()["src/aipower_rl/learning.py"].decode("utf-8")
    source_call = "torch.optim.AdamW(policy.actor_parameters(), lr=learning_rate)"
    assert learning_source.count(source_call) == 1

    wrong_class_source = learning_source.replace(
        source_call,
        "torch.optim.Adam(policy.actor_parameters(), lr=learning_rate)",
        1,
    )
    wrong_class = _assigned_call(
        wrong_class_source,
        function_name="fit_behavior_cloning",
        target_name="optimizer",
    )
    assert (
        _optimizer_contract_from_source_call(
            wrong_class,
            expected_parameter_expression="policy.actor_parameters()",
            expected_learning_rate_expression="learning_rate",
            learning_rate=payload["behavior_cloning_initialization"]["optimizer"]["learning_rate"],
        )
        != payload["behavior_cloning_initialization"]["optimizer"]
    )

    wrong_default_source = learning_source.replace(
        source_call,
        "torch.optim.AdamW(policy.actor_parameters(), lr=learning_rate, foreach=False)",
        1,
    )
    wrong_default = _assigned_call(
        wrong_default_source,
        function_name="fit_behavior_cloning",
        target_name="optimizer",
    )
    assert (
        _optimizer_contract_from_source_call(
            wrong_default,
            expected_parameter_expression="policy.actor_parameters()",
            expected_learning_rate_expression="learning_rate",
            learning_rate=payload["behavior_cloning_initialization"]["optimizer"]["learning_rate"],
        )
        != payload["behavior_cloning_initialization"]["optimizer"]
    )


@pytest.mark.source_archive
def test_historical_yaml_families_match_the_pinned_archive_at_recorded_revision() -> None:
    payload = load_training_provenance(PROVENANCE_PATH)
    source_blobs = _pinned_source_blobs()

    training_manifest = json.loads(source_blobs["artifacts/training_manifest.json"])
    experiment_summary = json.loads(source_blobs["experiments/s2-600-hidden512.json"])
    metric_rows = [
        json.loads(line)
        for line in source_blobs["artifacts/training_metrics.jsonl"].decode("utf-8").splitlines()
    ]
    selected_validation_series = [
        {
            "iteration": row["iteration"],
            "training_objective_score": row["validation_episode_reward"],
        }
        for row in metric_rows
        if row["stage"] == "ppo" and "validation_episode_reward" in row
    ]
    training = training_manifest["training"]
    simulation = training_manifest["simulation"]
    performance = training_manifest["performance"]
    experiment_config = experiment_summary["config"]
    experiment_source = source_blobs["src/aipower_rl/experiments.py"].decode("utf-8")
    config_source = source_blobs["src/aipower_rl/config.py"].decode("utf-8")
    scenario_source = source_blobs["src/aipower_rl/scenarios.py"].decode("utf-8")
    model_source = source_blobs["src/aipower_rl/models.py"].decode("utf-8")
    learning_source = source_blobs["src/aipower_rl/learning.py"].decode("utf-8")
    config_defaults = _class_defaults(config_source, "SimulationConfig")
    model_assignments = _method_self_assignments(
        model_source,
        class_name="ActorCriticPolicy",
        method_name="__init__",
    )
    actor_encoder = model_assignments["actor_encoder"]
    critic_encoder = model_assignments["critic_encoder"]
    assert isinstance(actor_encoder, ast.Call)
    assert isinstance(critic_encoder, ast.Call)
    expected_encoder_layers = [
        "nn.Linear(observation_dim, hidden_size)",
        "nn.Tanh()",
        "nn.Linear(hidden_size, hidden_size)",
        "nn.Tanh()",
    ]
    assert [ast.unparse(node) for node in actor_encoder.args] == expected_encoder_layers
    assert [ast.unparse(node) for node in critic_encoder.args] == expected_encoder_layers
    assert ast.unparse(model_assignments["actor"]) == "nn.Linear(hidden_size, action_count)"
    assert ast.unparse(model_assignments["critic"]) == "nn.Linear(hidden_size, 1)"
    source_initialization = _model_initialization_from_source(model_source)
    assert payload["model"]["network"]["initialization"] == source_initialization

    assert payload["record"]["purpose"].startswith("Historical parameter and lineage record")
    assert payload["model"]["checkpoint"]["source_revision"] == SOURCE_REVISION
    assert payload["model"]["network"] == {
        "observation_dim": 4 + 4 * len(_module_literal(config_source, "LOCALITIES")),
        "hidden_layers": [training["hidden_size"], training["hidden_size"]],
        "separate_actor_critic_encoders": True,
        "activation": "tanh",
        "actor_output_logits": 3 ** len(_module_literal(config_source, "LOCALITIES")),
        "critic_output_values": 1,
        "locality_count": len(_module_literal(config_source, "LOCALITIES")),
        "service_tiers_per_locality": 3,
        "action_encoding": "little-endian-base-3",
        "initialization": source_initialization,
    }

    assert payload["software_hardware"]["gpu"] == training_manifest["device_name"]
    assert payload["software_hardware"]["scope"] == "selected-run-seed-43"
    assert payload["software_hardware"]["pytorch"] == training_manifest["torch_version"]
    assert payload["software_hardware"]["cuda"] == training_manifest["cuda_version"]
    assert payload["software_hardware"]["training_device"] == training["device"]
    assert payload["software_hardware"]["elapsed_seconds"] == training_manifest["elapsed_seconds"]
    assert payload["software_hardware"]["ppo_seconds"] == performance["ppo_seconds"]
    assert (
        payload["software_hardware"]["ppo_transitions_per_second"]
        == performance["ppo_transitions_per_second"]
    )
    assert payload["software_hardware"]["peak_cuda_memory_mb"] == performance["peak_cuda_memory_mb"]

    generator = payload["training_generator"]
    localities = _module_literal(config_source, "LOCALITIES")
    locality_weights = _module_literal(config_source, "LOCALITY_DEMAND_WEIGHTS")
    sectors = _module_literal(config_source, "SECTORS")
    sector_shares = _module_literal(config_source, "SECTOR_SHARES")
    assert generator["horizon_hours"] == simulation["horizon"] == config_defaults["horizon"]
    assert (
        generator["system_peak_mw"]
        == (simulation["system_peak_mw"])
        == config_defaults["system_peak_mw"]
    )
    assert generator["locality_demand_weights"] == [
        {"locality": locality, "relative_weight": weight}
        for locality, weight in zip(localities, locality_weights, strict=True)
    ]
    assert generator["sector_shares"] == [
        {"sector": sector, "share": share}
        for sector, share in zip(sectors, sector_shares, strict=True)
    ]
    assert generator["sector_profiles"] == _sector_profile_specs_from_source(scenario_source)
    assert generator["sector_profile_normalization"] == SECTOR_PROFILE_NORMALIZATION
    assert (
        generator["demand_noise"]["amplitude"]
        == (simulation["demand_noise"])
        == config_defaults["demand_noise"]
    )
    assert (
        generator["training_supply"]["noise_amplitude"]
        == (simulation["supply_noise"])
        == config_defaults["supply_noise"]
    )

    reward = simulation["reward"]
    assert payload["objective"]["components"] == {
        "critical_service_weight": reward["critical_service"],
        "utilization_weight": reward["utilization"],
        "jain_fairness_weight": reward["fairness"],
        "switching_penalty_weight": reward["switching"],
        "oversubscription_penalty_weight": reward["oversubscription"],
        "scale": reward["scale"],
    }

    cloning = payload["behavior_cloning_initialization"]
    assert cloning["teacher_samples"] == training["teacher_samples"]
    assert cloning["teacher_rollout_batch_size"] == training["teacher_rollout_batch_size"]
    assert cloning["epochs"] == training["behavior_cloning_epochs"]
    assert cloning["minibatch_size"] == training["behavior_cloning_batch_size"]
    assert cloning["optimizer"]["learning_rate"] == training["behavior_cloning_learning_rate"]

    ppo = payload["ppo"]
    archived_ppo = training["ppo"]
    assert ppo["training_seed"] == training["seed"]
    assert ppo["iterations"] == training["ppo_iterations"]
    assert ppo["complete_episodes_per_iteration"] == training["rollout_batch_size"]
    assert ppo["gamma"] == archived_ppo["gamma"]
    assert ppo["gae_lambda"] == archived_ppo["gae_lambda"]
    assert ppo["learning_rate"] == archived_ppo["learning_rate"]
    assert ppo["clip_epsilon"] == archived_ppo["clip_epsilon"]
    assert ppo["value_coefficient"] == archived_ppo["value_coefficient"]
    assert ppo["entropy_coefficient"] == archived_ppo["entropy_coefficient"]
    assert ppo["epochs_per_iteration"] == archived_ppo["epochs"]
    assert ppo["minibatch_size"] == archived_ppo["minibatch_size"]
    assert ppo["actor_max_gradient_norm"] == archived_ppo["max_grad_norm"]
    assert ppo["critic_max_gradient_norm"] == archived_ppo["max_grad_norm"]
    assert ppo["learning_semantics"] == _ppo_learning_semantics_from_source(learning_source)
    for name in (
        "hidden_size",
        "teacher_samples",
        "teacher_rollout_batch_size",
        "behavior_cloning_epochs",
        "behavior_cloning_batch_size",
        "behavior_cloning_learning_rate",
        "ppo_iterations",
        "rollout_batch_size",
        "validation_batch_size",
        "validate_every",
    ):
        assert experiment_config[name] == training[name]
    assert experiment_config["ppo"] == archived_ppo

    selection = payload["validation_and_selection"]
    assert selection["validation_batch_size_per_shortage"] == training["validation_batch_size"]
    assert selection["validate_every_iterations"] == training["validate_every"]
    assert selection["candidate_training_seeds"] == experiment_summary["seeds"]
    assert selection["selected_seed"] == experiment_summary["selected_seed"]
    assert selection["selected_score"] == training_manifest["best_validation_score"]
    assert selection["candidate_best_scores"] == [
        {"seed": run["seed"], "score": run["best_validation_score"]}
        for run in experiment_summary["runs"]
    ]
    assert selection["promotion"]["eligible"] == experiment_summary["promotion"]["eligible"]
    assert selection["promotion"]["incumbent"] == experiment_summary["promotion"]["incumbent"]
    assert selection["promotion"]["checks"] == experiment_summary["promotion"]["checks"]
    assert "eligible = all(value for value in checks.values() if isinstance(value, bool))" in (
        experiment_source
    )
    assert "safe_runs = [run for run in runs if _safe_run(run)]" in experiment_source
    assert "selection_pool = safe_runs or runs" in experiment_source
    assert (
        'selected = max(selection_pool, key=lambda run: float(run["best_validation_score"]))'
        in experiment_source
    )
    assert payload["experiment_scope"]["candidate_experiment"] == {
        "name": experiment_summary["name"],
        "candidate_training_seeds": experiment_summary["seeds"],
        "candidate_run_count": len(experiment_summary["runs"]),
        "per_run_workload": EXPECTED_PER_RUN_WORKLOAD,
        "aggregate_workload": EXPECTED_AGGREGATE_EXPERIMENT_WORKLOAD,
    }
    assert (
        payload["experiment_scope"]["selected_run"]["training_seed"]
        == (experiment_summary["selected_seed"])
    )
    assert payload["convergence"]["scope"] == "selected-run-seed-43"
    assert payload["convergence"]["validation_series"] == selected_validation_series
