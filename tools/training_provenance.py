"""Strict documentation-time validation for the historical PPO training record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

TRAINING_PROVENANCE_SCHEMA = "gridguard.ppo_training_provenance"
TRAINING_PROVENANCE_SCHEMA_VERSION = 3
TRAINING_PROVENANCE_FILENAME = "training-provenance.yaml"

_SELECTED_RUN_SCOPE = "selected-run-seed-43"
_CANDIDATE_EXPERIMENT_NAME = "s2-600-hidden512"
_CANDIDATE_TRAINING_SEEDS = (42, 43, 44)
_PER_RUN_WORKLOAD = {
    "ppo_iterations": 600,
    "complete_episodes": 153_600,
    "transitions": 3_686_400,
}
_AGGREGATE_EXPERIMENT_WORKLOAD = {
    "ppo_iterations": 1_800,
    "complete_episodes": 460_800,
    "transitions": 11_059_200,
}
_SELECTED_RUN = {
    "training_seed": 43,
    "checkpoint_iteration": 510,
    "checkpoint_optimizer_updates": 6_120,
    "scoped_sections": (
        "software_hardware",
        "convergence",
        "ppo.training_seed",
        "ppo.selected_checkpoint_optimizer_updates",
    ),
}

_PROMOTION_THRESHOLDS = {
    "mean_reward_margin_minimum": 0.005,
    "matched_seed_wins_minimum": 2,
    "maximum_seed_regression": 0.02,
    "minimum_throughput_ratio": 0.75,
    "minimum_critical_service_ratio": 0.99,
    "maximum_oversubscription_ratio": 0.005,
    "required_power_balance_violation_rate": 0.0,
}
_PROMOTION_CHECKS = {
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
_PROMOTION_ELIGIBILITY_RULE = "all-boolean-promotion-checks-must-be-true"
_REWARD_GATE_EVIDENCE_ROLE = (
    "historical-model-selection-fact-not-independent-current-headline-evidence"
)
_SELECTION_RULE = {
    "safety_pool": (
        "passed-and-all-gates-true-and-finite-learned-metrics-and-critical-service-"
        "at-least-0.99-and-oversubscription-at-most-0.005-and-zero-power-balance-violations"
    ),
    "fallback": "use-all-runs-if-no-run-is-safe",
    "ranking": "maximum-best-validation-score",
}

_PROVENANCE_MAPPING_SCOPE = (
    "Historical training-parameter and experiment fields; schema metadata, documentary boundary "
    "flags, current actor-export lineage, and source-binding rows are separately validated."
)

_PROVENANCE_TAXONOMY = {
    "directly-recorded-archive-fact": {
        "definition": (
            "Value copied from, or cryptographically measured from, a bound archive artifact."
        ),
        "mapped_fields": (
            "model.checkpoint",
            "software_hardware.training_device",
            "software_hardware.gpu",
            "software_hardware.pytorch",
            "software_hardware.cuda",
            "software_hardware.elapsed_seconds",
            "software_hardware.ppo_seconds",
            "software_hardware.ppo_transitions_per_second",
            "software_hardware.peak_cuda_memory_mb",
            "behavior_cloning_initialization.teacher_samples",
            "behavior_cloning_initialization.teacher_rollout_batch_size",
            "behavior_cloning_initialization.epochs",
            "behavior_cloning_initialization.minibatch_size",
            "behavior_cloning_initialization.optimizer.learning_rate",
            "ppo.iterations",
            "ppo.complete_episodes_per_iteration",
            "ppo.episode_horizon_hours",
            "ppo.gamma",
            "ppo.gae_lambda",
            "ppo.learning_rate",
            "ppo.clip_epsilon",
            "ppo.value_coefficient",
            "ppo.entropy_coefficient",
            "ppo.epochs_per_iteration",
            "ppo.minibatch_size",
            "ppo.actor_max_gradient_norm",
            "ppo.critic_max_gradient_norm",
            "ppo.optimizer.learning_rate",
            "training_generator.horizon_hours",
            "training_generator.system_peak_mw",
            "training_generator.demand_noise.amplitude",
            "training_generator.training_supply.noise_amplitude",
            "objective.components",
            "validation_and_selection.candidate_best_scores",
            "validation_and_selection.promotion.checks",
            "convergence.untrained_score",
            "convergence.after_behavior_cloning_score",
            "convergence.ppo_iteration_100_score",
            "convergence.selected_iteration_510_score",
            "convergence.final_iteration_600_score",
            "convergence.validation_series",
        ),
    },
    "bound-source-or-pytorch-default-reconstruction": {
        "definition": (
            "Value reconstructed from a pinned source blob or an explicitly named PyTorch "
            "2.8 default because the run artifacts did not serialize it directly."
        ),
        "mapped_fields": (
            "model.network.observation_dim",
            "model.network.hidden_layers",
            "model.network.separate_actor_critic_encoders",
            "model.network.activation",
            "model.network.critic_output_values",
            "model.network.locality_count",
            "model.network.service_tiers_per_locality",
            "model.network.action_encoding",
            "model.network.initialization",
            "training_generator.hour_resolution",
            "training_generator.locality_demand_weights",
            "training_generator.sector_shares",
            "training_generator.sector_profiles",
            "training_generator.sector_profile_normalization",
            "training_generator.demand_noise.distribution",
            "training_generator.demand_noise.multiplier_min",
            "training_generator.demand_noise.multiplier_max",
            "training_generator.training_supply.coupled_to_total_demand",
            "training_generator.training_supply.base_fraction_distribution",
            "training_generator.training_supply.base_fraction_min",
            "training_generator.training_supply.base_fraction_max",
            "training_generator.training_supply.shock_start_hour_min",
            "training_generator.training_supply.shock_start_hour_max_inclusive",
            "training_generator.training_supply.shock_duration_hours_min",
            "training_generator.training_supply.shock_duration_hours_max_inclusive",
            "training_generator.training_supply.shock_depth_distribution",
            "training_generator.training_supply.shock_depth_min",
            "training_generator.training_supply.shock_depth_max",
            "training_generator.training_supply.fraction_clamp_min",
            "training_generator.training_supply.fraction_clamp_max",
            "training_generator.training_supply.noise_distribution",
            "training_generator.training_supply.total_demand_cap_fraction",
            "objective.public_role",
            "objective.formula",
            "objective.episode_aggregation",
            "objective.component_definitions",
            "behavior_cloning_initialization.teacher_episode_seed_start",
            "behavior_cloning_initialization.teacher_episode_seed_end_inclusive",
            "behavior_cloning_initialization.exploration_probability",
            "behavior_cloning_initialization.exploration_prng_seed",
            "behavior_cloning_initialization.shuffle_prng_seed",
            "behavior_cloning_initialization.gradient_clip_defaults",
            "behavior_cloning_initialization.optimizer",
            "ppo.learning_semantics",
            "ppo.gradient_clip_defaults",
            "ppo.optimizer",
            "validation_and_selection.selection_rule",
        ),
    },
    "arithmetic-derived-value": {
        "definition": (
            "Value calculated exactly from validated directly recorded or reconstructed inputs."
        ),
        "mapped_fields": (
            "experiment_scope.candidate_experiment.per_run_workload",
            "experiment_scope.candidate_experiment.aggregate_workload",
            "model.network.actor_output_logits",
            "behavior_cloning_initialization.minibatches_per_epoch",
            "behavior_cloning_initialization.optimizer_updates",
            "ppo.transitions_per_iteration",
            "ppo.total_episodes",
            "ppo.total_transitions",
            "ppo.minibatches_per_epoch",
            "ppo.optimizer_updates_per_iteration",
            "ppo.planned_optimizer_updates",
            "ppo.selected_checkpoint_optimizer_updates",
            "validation_and_selection.validation_episodes_per_checkpoint",
        ),
    },
    "unavailable-historical-environment-fact": {
        "definition": (
            "Historical environment fact absent from every bound archive artifact and therefore "
            "left explicitly unknown."
        ),
        "mapped_fields": ("provenance_taxonomy.historical_environment_unknowns",),
    },
}
_HISTORICAL_ENVIRONMENT_UNKNOWNS = (
    "python-interpreter-version",
    "operating-system-edition-and-build",
    "nvidia-driver-version",
    "cudnn-version-and-runtime-flags",
    "cpu-model-and-topology",
    "installed-system-memory",
    "complete-python-environment-lock-and-wheel-hashes",
    "training-process-environment-variables",
    "gpu-clock-power-and-thermal-state",
)
_REPRODUCIBILITY_BOUNDARY_STATEMENT = (
    "The bound record supports audit of the selected actor's historical lineage, not executable "
    "or bitwise-identical retraining."
)

_PPO_ATTRIBUTION_LIMITATION = (
    "No behavior-cloning-only ablation was run, so this comparison cannot isolate the "
    "incremental effect of PPO fine-tuning."
)

_OBJECTIVE_TEXT = {
    "public_role": "training-objective-diagnostic",
    "formula": "(5*C + 2*U + 2*J - 0.25*S - 3*O) / 9",
    "episode_aggregation": "sum-of-24-hour-rewards",
}
_OBJECTIVE_COMPONENTS = {
    "critical_service_weight": 5.0,
    "utilization_weight": 2.0,
    "jain_fairness_weight": 2.0,
    "switching_penalty_weight": 0.25,
    "oversubscription_penalty_weight": 3.0,
    "scale": 9.0,
}
_OBJECTIVE_COMPONENT_DEFINITIONS = {
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
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_REVISION = "52f6dd82605c993c8094448caecec86faa289c0c"
_CHECKPOINT_PATH = "artifacts/checkpoints/best.pt"
_CHECKPOINT_SHA256 = "c6e5712f966b46ba18f4c69fb457b9e69f78887c0c5c0279e5ce034f40f9c665"
_ACTOR_SHA256 = "3e4a7ce44267dd1cf3bbde7e77abafb8f31fdea77ece63acda7be17fa3198ad5"
_SECTOR_PROFILE_NORMALIZATION = "divide-each-sector-by-its-24-hour-maximum"
_SECTOR_PROFILE_SPECS = (
    ("Hospital", 0.95, ()),
    ("Water", 0.45, ((0.35, 6.0, 2.0), (0.40, 18.0, 2.5))),
    ("Bakery", 0.35, ((0.60, 5.0, 1.5), (0.45, 16.0, 2.0))),
    ("Residential", 0.40, ((0.30, 7.0, 2.0), (0.65, 20.0, 3.0))),
    ("Commercial", 0.25, ((0.75, 13.0, 4.0),)),
)
_MODEL_INITIALIZATION = {
    "encoder_linear_layers": {
        "applies_to": ("actor_encoder", "critic_encoder"),
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
_PPO_LEARNING_SEMANTICS = {
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
_BEHAVIOR_CLONING_OPTIMIZER = {
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
_PPO_OPTIMIZER = {
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
_GRADIENT_CLIP_DEFAULTS = {
    "norm_type": 2.0,
    "error_if_nonfinite": False,
    "foreach": None,
}
_BEHAVIOR_CLONING_MAX_GRADIENT_NORM = 1.0
_PPO_MAX_GRADIENT_NORM = 0.5
_SOURCE_BINDINGS = (
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


class TrainingProvenanceError(RuntimeError):
    """Raised when the PPO training provenance cannot be trusted."""


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects YAML features which can obscure reviewed values."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise ComposerError(
                None,
                None,
                "YAML aliases are not allowed",
                event.start_mark,
            )
        event = self.peek_event()
        if event.anchor is not None:
            raise ComposerError(
                None,
                None,
                "YAML anchors are not allowed",
                event.start_mark,
            )
        if event.tag not in (None, "!"):
            raise ComposerError(
                None,
                None,
                "explicit YAML tags are not allowed",
                event.start_mark,
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, found {node.id}",
                node.start_mark,
            )
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise ConstructorError(
                    None,
                    None,
                    "YAML merge keys are not allowed",
                    key_node.start_mark,
                )
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as error:
                raise ConstructorError(
                    None,
                    None,
                    "mapping keys must be hashable",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    None,
                    None,
                    f"duplicate YAML key {key!r}",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


class _CanonicalSafeDumper(yaml.SafeDumper):
    """Stable block-style YAML dumper without aliases or platform newlines."""

    def ignore_aliases(self, data: Any) -> bool:
        return True

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, False)


def parse_training_provenance(text: str) -> dict[str, Any]:
    """Parse and semantically validate one strict YAML training record."""

    if not isinstance(text, str):
        raise TrainingProvenanceError("training provenance YAML must be text")
    try:
        payload = yaml.load(text, Loader=_StrictSafeLoader)
    except yaml.YAMLError as error:
        message = str(error)
        if "expected a single document" in message:
            message = f"training provenance must contain a single document: {message}"
        raise TrainingProvenanceError(message) from error
    document = _mapping(payload, field="training provenance")
    _reject_nonfinite_numbers(document, field="training provenance")
    _validate_document(document)
    return document


def render_training_provenance(payload: Mapping[str, Any]) -> str:
    """Validate and deterministically render the canonical block-style YAML record."""

    document = dict(payload)
    _reject_nonfinite_numbers(document, field="training provenance")
    _validate_document(document)
    rendered = yaml.dump(
        document,
        Dumper=_CanonicalSafeDumper,
        allow_unicode=False,
        default_flow_style=False,
        explicit_end=False,
        sort_keys=False,
        width=100,
    )
    return rendered.replace("\r\n", "\n")


def load_training_provenance(
    path: str | Path,
    *,
    model_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a strict YAML record and optionally bind it to the model manifest."""

    provenance_path = Path(path)
    try:
        text = provenance_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise TrainingProvenanceError(f"training provenance cannot be read: {error}") from error
    payload = parse_training_provenance(text)
    if model_manifest_path is not None:
        _validate_model_manifest_binding(
            payload,
            provenance_path=provenance_path,
            model_manifest_path=Path(model_manifest_path),
        )
    return payload


def build_training_summary(
    payload: Mapping[str, Any],
    *,
    provenance_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build compact, stable reviewer-facing fields from a validated record."""

    document = dict(payload)
    _reject_nonfinite_numbers(document, field="training provenance")
    _validate_document(document)
    path = (
        Path(provenance_path)
        if provenance_path is not None
        else Path(__file__).resolve().parents[1] / "data" / "model" / TRAINING_PROVENANCE_FILENAME
    )
    try:
        provenance_bytes = path.read_bytes()
    except OSError as error:
        raise TrainingProvenanceError(
            f"training provenance cannot be summarized: {error}"
        ) from error
    repository_root = Path(__file__).resolve().parents[1]
    try:
        display_path = path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        display_path = str(path.resolve())

    model = _mapping(document["model"], field="model")
    record = _mapping(document["record"], field="record")
    network = _mapping(model["network"], field="model.network")
    cloning = _mapping(
        document["behavior_cloning_initialization"],
        field="behavior_cloning_initialization",
    )
    ppo = _mapping(document["ppo"], field="ppo")
    learning_semantics = _mapping(ppo["learning_semantics"], field="ppo.learning_semantics")
    selection = _mapping(
        document["validation_and_selection"],
        field="validation_and_selection",
    )
    promotion = _mapping(
        selection["promotion"],
        field="validation_and_selection.promotion",
    )
    experiment_scope = _mapping(document["experiment_scope"], field="experiment_scope")
    candidate_experiment = _mapping(
        experiment_scope["candidate_experiment"],
        field="experiment_scope.candidate_experiment",
    )
    selected_run = _mapping(
        experiment_scope["selected_run"],
        field="experiment_scope.selected_run",
    )
    taxonomy = _mapping(document["provenance_taxonomy"], field="provenance_taxonomy")
    convergence = _mapping(document["convergence"], field="convergence")
    attribution = _mapping(
        document["attribution_limitations"],
        field="attribution_limitations",
    )
    generator = _mapping(document["training_generator"], field="training_generator")
    sector_profiles = []
    for index, value in enumerate(
        _sequence(generator["sector_profiles"], field="training_generator.sector_profiles")
    ):
        row = _mapping(value, field=f"training_generator.sector_profiles[{index}]")
        sector_profiles.append(
            {
                "sector": row["sector"],
                "baseline_amplitude": row["baseline_amplitude"],
                "gaussian_terms": [
                    dict(
                        _mapping(
                            term,
                            field=(f"training_generator.sector_profiles[{index}].gaussian_terms"),
                        )
                    )
                    for term in _sequence(
                        row["gaussian_terms"],
                        field=f"training_generator.sector_profiles[{index}].gaussian_terms",
                    )
                ],
            }
        )
    initialization = _mapping(network["initialization"], field="model.network.initialization")
    encoder_initialization = _mapping(
        initialization["encoder_linear_layers"],
        field="model.network.initialization.encoder_linear_layers",
    )
    actor_initialization = _mapping(
        initialization["actor_head"],
        field="model.network.initialization.actor_head",
    )
    critic_initialization = _mapping(
        initialization["critic_head"],
        field="model.network.initialization.critic_head",
    )
    hidden = _integer_list(network["hidden_layers"], field="model.network.hidden_layers")
    dimensions = [
        _integer(network["observation_dim"], field="model.network.observation_dim"),
        *hidden,
        _integer(network["actor_output_logits"], field="model.network.actor_output_logits"),
    ]
    return {
        "record": {
            "purpose": record["purpose"],
            "executable_retraining_configuration": record["executable_retraining_configuration"],
            "bitwise_retraining_claimed": record["bitwise_retraining_claimed"],
            "optimizer_state_included": record["optimizer_state_included"],
            "raw_training_runs_included": record["raw_training_runs_included"],
        },
        "experiment_scope": {
            "candidate_experiment": {
                "name": candidate_experiment["name"],
                "candidate_training_seeds": list(candidate_experiment["candidate_training_seeds"]),
                "candidate_run_count": candidate_experiment["candidate_run_count"],
                "per_run_workload": dict(
                    _mapping(
                        candidate_experiment["per_run_workload"],
                        field="experiment_scope.candidate_experiment.per_run_workload",
                    )
                ),
                "aggregate_workload": dict(
                    _mapping(
                        candidate_experiment["aggregate_workload"],
                        field="experiment_scope.candidate_experiment.aggregate_workload",
                    )
                ),
            },
            "selected_run": {
                **selected_run,
                "scoped_sections": list(selected_run["scoped_sections"]),
            },
        },
        "controller": model["controller"],
        "network": " -> ".join(str(value) for value in dimensions),
        "model_initialization": {
            "encoder_linear_layers": {
                **encoder_initialization,
                "applies_to": list(encoder_initialization["applies_to"]),
            },
            "actor_head": dict(actor_initialization),
            "critic_head": dict(critic_initialization),
        },
        "behavior_cloning_initialization": {
            "teacher_samples": cloning["teacher_samples"],
            "epochs": cloning["epochs"],
            "optimizer_updates": cloning["optimizer_updates"],
            "max_gradient_norm": cloning["max_gradient_norm"],
            "gradient_clip_defaults": dict(
                _mapping(
                    cloning["gradient_clip_defaults"],
                    field="behavior_cloning_initialization.gradient_clip_defaults",
                )
            ),
            "optimizer": dict(
                _mapping(
                    cloning["optimizer"],
                    field="behavior_cloning_initialization.optimizer",
                )
            ),
        },
        "ppo": {
            "iterations": ppo["iterations"],
            "total_episodes": ppo["total_episodes"],
            "total_transitions": ppo["total_transitions"],
            "learning_rate": ppo["learning_rate"],
            "gamma": ppo["gamma"],
            "gae_lambda": ppo["gae_lambda"],
            "advantage_normalization": ppo["advantage_normalization"],
            "clip_epsilon": ppo["clip_epsilon"],
            "entropy_coefficient": ppo["entropy_coefficient"],
            "actor_max_gradient_norm": ppo["actor_max_gradient_norm"],
            "critic_max_gradient_norm": ppo["critic_max_gradient_norm"],
            "gradient_clip_defaults": dict(
                _mapping(ppo["gradient_clip_defaults"], field="ppo.gradient_clip_defaults")
            ),
            "optimizer": dict(_mapping(ppo["optimizer"], field="ppo.optimizer")),
            "learning_semantics": {
                name: dict(_mapping(value, field=f"ppo.learning_semantics.{name}"))
                for name, value in learning_semantics.items()
            },
        },
        "training_generator": {
            "horizon_hours": generator["horizon_hours"],
            "system_peak_mw": generator["system_peak_mw"],
            "sector_profile_normalization": generator["sector_profile_normalization"],
            "sector_profiles": sector_profiles,
        },
        "selection": {
            "selected_seed": selection["selected_seed"],
            "selected_iteration": selection["selected_iteration"],
            "selected_checkpoint_optimizer_updates": ppo["selected_checkpoint_optimizer_updates"],
            "validation_seeds": list(selection["validation_seeds"]),
            "promotion": {
                "eligible": promotion["eligible"],
                "incumbent": promotion["incumbent"],
                "eligibility_rule": promotion["eligibility_rule"],
                "reward_gate_evidence_role": promotion["reward_gate_evidence_role"],
                "thresholds": dict(
                    _mapping(
                        promotion["thresholds"],
                        field="validation_and_selection.promotion.thresholds",
                    )
                ),
                "checks": dict(
                    _mapping(
                        promotion["checks"],
                        field="validation_and_selection.promotion.checks",
                    )
                ),
            },
        },
        "provenance_taxonomy": {
            "mapping_scope": taxonomy["mapping_scope"],
            "categories": list(
                _mapping(taxonomy["categories"], field="provenance_taxonomy.categories")
            ),
            "historical_environment_unknowns": [
                _mapping(
                    row,
                    field="provenance_taxonomy.historical_environment_unknowns[]",
                )["fact"]
                for row in _sequence(
                    taxonomy["historical_environment_unknowns"],
                    field="provenance_taxonomy.historical_environment_unknowns",
                )
            ],
            "reproducibility_boundary": dict(
                _mapping(
                    taxonomy["reproducibility_boundary"],
                    field="provenance_taxonomy.reproducibility_boundary",
                )
            ),
        },
        "attribution_limitations": dict(attribution),
        "convergence": {
            "untrained": convergence["untrained_score"],
            "after_behavior_cloning": convergence["after_behavior_cloning_score"],
            "ppo_iteration_100": convergence["ppo_iteration_100_score"],
            "selected_iteration_510": convergence["selected_iteration_510_score"],
            "final_iteration_600": convergence["final_iteration_600_score"],
        },
        "training_provenance": {
            "path": display_path,
            "sha256": hashlib.sha256(provenance_bytes).hexdigest(),
            "size_bytes": len(provenance_bytes),
        },
        "source_binding_count": len(
            _sequence(document["source_bindings"], field="source_bindings")
        ),
    }


def _validate_document(payload: dict[str, Any]) -> None:
    _exact_fields(
        payload,
        {
            "schema",
            "schema_version",
            "record",
            "experiment_scope",
            "provenance_taxonomy",
            "model",
            "software_hardware",
            "training_generator",
            "objective",
            "behavior_cloning_initialization",
            "ppo",
            "validation_and_selection",
            "attribution_limitations",
            "evaluation_seed_disclosure",
            "convergence",
            "source_bindings",
        },
        field="training provenance",
    )
    if _text(payload["schema"], field="schema") != TRAINING_PROVENANCE_SCHEMA:
        raise TrainingProvenanceError("unsupported training provenance schema")
    if _integer(payload["schema_version"], field="schema_version") != (
        TRAINING_PROVENANCE_SCHEMA_VERSION
    ):
        raise TrainingProvenanceError("unsupported training provenance schema_version")

    _validate_record(_mapping(payload["record"], field="record"))
    _validate_model(_mapping(payload["model"], field="model"))
    _validate_software_hardware(_mapping(payload["software_hardware"], field="software_hardware"))
    _validate_training_generator(
        _mapping(payload["training_generator"], field="training_generator")
    )
    _validate_objective(_mapping(payload["objective"], field="objective"))
    cloning = _mapping(
        payload["behavior_cloning_initialization"],
        field="behavior_cloning_initialization",
    )
    _validate_behavior_cloning(cloning)
    ppo = _mapping(payload["ppo"], field="ppo")
    _validate_ppo(ppo)
    selection = _mapping(
        payload["validation_and_selection"],
        field="validation_and_selection",
    )
    _validate_selection(selection)
    _validate_experiment_scope(
        _mapping(payload["experiment_scope"], field="experiment_scope"),
        ppo=ppo,
        selection=selection,
    )
    _validate_provenance_taxonomy(
        _mapping(payload["provenance_taxonomy"], field="provenance_taxonomy"),
        record=_mapping(payload["record"], field="record"),
    )
    _validate_attribution_limitations(
        _mapping(payload["attribution_limitations"], field="attribution_limitations")
    )
    disclosure = _mapping(
        payload["evaluation_seed_disclosure"],
        field="evaluation_seed_disclosure",
    )
    _validate_evaluation_seed_disclosure(disclosure, ppo=ppo, selection=selection)
    _validate_convergence(
        _mapping(payload["convergence"], field="convergence"),
        selection=selection,
    )
    _validate_source_bindings(payload["source_bindings"])

    if _integer(cloning["shuffle_prng_seed"], field="shuffle_prng_seed") != _integer(
        ppo["training_seed"], field="training_seed"
    ):
        raise TrainingProvenanceError(
            "behavior-cloning shuffle_prng_seed must equal the selected training_seed"
        )
    selected_iteration = _integer(selection["selected_iteration"], field="selected_iteration")
    selected_updates = _integer(
        ppo["selected_checkpoint_optimizer_updates"],
        field="selected_checkpoint_optimizer_updates",
    )
    updates_per_iteration = _integer(
        ppo["optimizer_updates_per_iteration"],
        field="optimizer_updates_per_iteration",
    )
    if selected_updates != selected_iteration * updates_per_iteration:
        raise TrainingProvenanceError(
            "selected_checkpoint_optimizer_updates does not match the selected iteration"
        )


def _validate_experiment_scope(
    scope: dict[str, Any],
    *,
    ppo: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    _exact_fields(
        scope,
        {"candidate_experiment", "selected_run"},
        field="experiment_scope",
    )
    candidate = _mapping(
        scope["candidate_experiment"],
        field="experiment_scope.candidate_experiment",
    )
    _exact_fields(
        candidate,
        {
            "name",
            "candidate_training_seeds",
            "candidate_run_count",
            "per_run_workload",
            "aggregate_workload",
        },
        field="experiment_scope.candidate_experiment",
    )
    if _text(candidate["name"], field="experiment_scope.candidate_experiment.name") != (
        _CANDIDATE_EXPERIMENT_NAME
    ):
        raise TrainingProvenanceError("experiment_scope candidate experiment name is not accepted")
    candidate_seeds = tuple(
        _integer_list(
            candidate["candidate_training_seeds"],
            field="experiment_scope.candidate_experiment.candidate_training_seeds",
        )
    )
    if candidate_seeds != _CANDIDATE_TRAINING_SEEDS:
        raise TrainingProvenanceError(
            "experiment_scope candidate_training_seeds do not match the bound experiment"
        )
    run_count = _positive_integer(
        candidate["candidate_run_count"],
        field="experiment_scope.candidate_experiment.candidate_run_count",
    )
    if run_count != len(candidate_seeds):
        raise TrainingProvenanceError(
            "experiment_scope.candidate_experiment.candidate_run_count is inconsistent"
        )
    per_run = _validate_workload(
        candidate["per_run_workload"],
        field="experiment_scope.candidate_experiment.per_run_workload",
        expected=_PER_RUN_WORKLOAD,
    )
    aggregate = _validate_workload(
        candidate["aggregate_workload"],
        field="experiment_scope.candidate_experiment.aggregate_workload",
        expected=_AGGREGATE_EXPERIMENT_WORKLOAD,
    )
    if any(aggregate[name] != per_run[name] * run_count for name in per_run):
        raise TrainingProvenanceError(
            "experiment_scope candidate aggregate_workload is inconsistent"
        )

    selected = _mapping(scope["selected_run"], field="experiment_scope.selected_run")
    _exact_fields(selected, set(_SELECTED_RUN), field="experiment_scope.selected_run")
    normalized_selected = {
        "training_seed": _positive_integer(
            selected["training_seed"], field="experiment_scope.selected_run.training_seed"
        ),
        "checkpoint_iteration": _positive_integer(
            selected["checkpoint_iteration"],
            field="experiment_scope.selected_run.checkpoint_iteration",
        ),
        "checkpoint_optimizer_updates": _positive_integer(
            selected["checkpoint_optimizer_updates"],
            field="experiment_scope.selected_run.checkpoint_optimizer_updates",
        ),
        "scoped_sections": tuple(
            _text_list(
                selected["scoped_sections"],
                field="experiment_scope.selected_run.scoped_sections",
            )
        ),
    }
    if normalized_selected != _SELECTED_RUN:
        raise TrainingProvenanceError(
            "experiment_scope.selected_run does not match the accepted selected checkpoint"
        )
    if (
        tuple(
            _integer_list(
                selection["candidate_training_seeds"],
                field="validation_and_selection.candidate_training_seeds",
            )
        )
        != candidate_seeds
    ):
        raise TrainingProvenanceError(
            "experiment_scope candidate seeds do not match validation_and_selection"
        )
    cross_checks = {
        "training_seed": _integer(ppo["training_seed"], field="ppo.training_seed"),
        "checkpoint_iteration": _integer(
            selection["selected_iteration"],
            field="validation_and_selection.selected_iteration",
        ),
        "checkpoint_optimizer_updates": _integer(
            ppo["selected_checkpoint_optimizer_updates"],
            field="ppo.selected_checkpoint_optimizer_updates",
        ),
    }
    for name, expected in cross_checks.items():
        if normalized_selected[name] != expected:
            raise TrainingProvenanceError(
                f"experiment_scope.selected_run.{name} does not match its scoped field"
            )
    if per_run != {
        "ppo_iterations": _integer(ppo["iterations"], field="ppo.iterations"),
        "complete_episodes": _integer(ppo["total_episodes"], field="ppo.total_episodes"),
        "transitions": _integer(ppo["total_transitions"], field="ppo.total_transitions"),
    }:
        raise TrainingProvenanceError(
            "experiment_scope per_run_workload does not match the PPO run"
        )


def _validate_workload(
    value: Any,
    *,
    field: str,
    expected: Mapping[str, int],
) -> dict[str, int]:
    workload = _mapping(value, field=field)
    _exact_fields(workload, set(expected), field=field)
    normalized = {
        name: _positive_integer(workload[name], field=f"{field}.{name}") for name in expected
    }
    if normalized != dict(expected):
        raise TrainingProvenanceError(f"{field} does not match the bound experiment workload")
    return normalized


def _validate_provenance_taxonomy(
    taxonomy: dict[str, Any],
    *,
    record: dict[str, Any],
) -> None:
    _exact_fields(
        taxonomy,
        {
            "mapping_scope",
            "categories",
            "historical_environment_unknowns",
            "reproducibility_boundary",
        },
        field="provenance_taxonomy",
    )
    if _text(taxonomy["mapping_scope"], field="provenance_taxonomy.mapping_scope") != (
        _PROVENANCE_MAPPING_SCOPE
    ):
        raise TrainingProvenanceError("provenance_taxonomy.mapping_scope is not accepted")
    categories = _mapping(taxonomy["categories"], field="provenance_taxonomy.categories")
    _exact_fields(categories, set(_PROVENANCE_TAXONOMY), field="provenance_taxonomy.categories")
    observed_paths: set[str] = set()
    for name, expected in _PROVENANCE_TAXONOMY.items():
        field = f"provenance_taxonomy.categories.{name}"
        category = _mapping(categories[name], field=field)
        _exact_fields(category, {"definition", "mapped_fields"}, field=field)
        definition = _text(category["definition"], field=f"{field}.definition")
        mapped_fields = tuple(_text_list(category["mapped_fields"], field=f"{field}.mapped_fields"))
        if definition != expected["definition"] or mapped_fields != expected["mapped_fields"]:
            raise TrainingProvenanceError(
                f"{field}.mapped_fields or definition does not match the accepted taxonomy"
            )
        if observed_paths.intersection(mapped_fields):
            raise TrainingProvenanceError("provenance_taxonomy mapped_fields overlap categories")
        observed_paths.update(mapped_fields)

    unknown_rows = _sequence(
        taxonomy["historical_environment_unknowns"],
        field="provenance_taxonomy.historical_environment_unknowns",
    )
    unknowns: list[str] = []
    for index, value in enumerate(unknown_rows):
        field = f"provenance_taxonomy.historical_environment_unknowns[{index}]"
        row = _mapping(value, field=field)
        _exact_fields(row, {"fact", "status"}, field=field)
        fact = _text(row["fact"], field=f"{field}.fact")
        status = _text(row["status"], field=f"{field}.status")
        if status != "unavailable-in-bound-archive":
            raise TrainingProvenanceError(
                "provenance_taxonomy historical_environment_unknowns must remain unavailable"
            )
        unknowns.append(fact)
    if tuple(unknowns) != _HISTORICAL_ENVIRONMENT_UNKNOWNS:
        raise TrainingProvenanceError(
            "provenance_taxonomy historical_environment_unknowns are incomplete or reordered"
        )

    boundary = _mapping(
        taxonomy["reproducibility_boundary"],
        field="provenance_taxonomy.reproducibility_boundary",
    )
    _exact_fields(
        boundary,
        {"executable_retraining_configuration", "bitwise_retraining_claimed", "statement"},
        field="provenance_taxonomy.reproducibility_boundary",
    )
    for name in ("executable_retraining_configuration", "bitwise_retraining_claimed"):
        if _boolean(boundary[name], field=f"provenance_taxonomy.{name}"):
            raise TrainingProvenanceError(f"provenance_taxonomy.{name} must be false")
        if boundary[name] is not record[name]:
            raise TrainingProvenanceError(
                f"provenance_taxonomy.{name} does not match record.{name}"
            )
    if _text(
        boundary["statement"], field="provenance_taxonomy.reproducibility_boundary.statement"
    ) != (_REPRODUCIBILITY_BOUNDARY_STATEMENT):
        raise TrainingProvenanceError(
            "provenance_taxonomy reproducibility boundary statement is not accepted"
        )


def _validate_attribution_limitations(limitations: dict[str, Any]) -> None:
    _exact_fields(
        limitations,
        {
            "behavior_cloning_only_ablation_run",
            "ppo_fine_tuning_incremental_effect_isolated",
            "interpretation",
        },
        field="attribution_limitations",
    )
    for name in (
        "behavior_cloning_only_ablation_run",
        "ppo_fine_tuning_incremental_effect_isolated",
    ):
        if _boolean(limitations[name], field=f"attribution_limitations.{name}"):
            raise TrainingProvenanceError(f"attribution_limitations.{name} must be false")
    if (
        _text(
            limitations["interpretation"],
            field="attribution_limitations.interpretation",
        )
        != _PPO_ATTRIBUTION_LIMITATION
    ):
        raise TrainingProvenanceError(
            "attribution_limitations.interpretation must state the bounded PPO attribution"
        )


def _validate_record(record: dict[str, Any]) -> None:
    _exact_fields(
        record,
        {
            "purpose",
            "executable_retraining_configuration",
            "bitwise_retraining_claimed",
            "optimizer_state_included",
            "raw_training_runs_included",
        },
        field="record",
    )
    _text(record["purpose"], field="record.purpose")
    for name in (
        "executable_retraining_configuration",
        "bitwise_retraining_claimed",
        "optimizer_state_included",
        "raw_training_runs_included",
    ):
        if _boolean(record[name], field=f"record.{name}"):
            raise TrainingProvenanceError(f"record.{name} must be false")


def _validate_model(model: dict[str, Any]) -> None:
    _exact_fields(model, {"controller", "checkpoint", "actor", "network"}, field="model")
    if _text(model["controller"], field="model.controller") != "PPO":
        raise TrainingProvenanceError("model.controller must be PPO")
    checkpoint = _mapping(model["checkpoint"], field="model.checkpoint")
    _exact_fields(
        checkpoint,
        {"path", "sha256", "size_bytes", "source_revision"},
        field="model.checkpoint",
    )
    if _relative_path(checkpoint["path"], field="model.checkpoint.path") != _CHECKPOINT_PATH:
        raise TrainingProvenanceError("model.checkpoint.path is not the accepted source checkpoint")
    if _sha256(checkpoint["sha256"], field="model.checkpoint.sha256") != _CHECKPOINT_SHA256:
        raise TrainingProvenanceError("model.checkpoint.sha256 is not the accepted source hash")
    _positive_integer(checkpoint["size_bytes"], field="model.checkpoint.size_bytes")
    revision = _text(checkpoint["source_revision"], field="model.checkpoint.source_revision")
    if not _REVISION_PATTERN.fullmatch(revision) or revision != _SOURCE_REVISION:
        raise TrainingProvenanceError("model.checkpoint.source_revision is not accepted")

    actor = _mapping(model["actor"], field="model.actor")
    _exact_fields(actor, {"filename", "sha256", "size_bytes"}, field="model.actor")
    if _relative_path(actor["filename"], field="model.actor.filename") != "ppo-policy.npz":
        raise TrainingProvenanceError("model.actor.filename must be ppo-policy.npz")
    if _sha256(actor["sha256"], field="model.actor.sha256") != _ACTOR_SHA256:
        raise TrainingProvenanceError("model.actor.sha256 is not the accepted actor hash")
    _positive_integer(actor["size_bytes"], field="model.actor.size_bytes")

    network = _mapping(model["network"], field="model.network")
    _exact_fields(
        network,
        {
            "observation_dim",
            "hidden_layers",
            "separate_actor_critic_encoders",
            "activation",
            "actor_output_logits",
            "critic_output_values",
            "locality_count",
            "service_tiers_per_locality",
            "action_encoding",
            "initialization",
        },
        field="model.network",
    )
    for name in (
        "observation_dim",
        "actor_output_logits",
        "critic_output_values",
        "locality_count",
        "service_tiers_per_locality",
    ):
        _positive_integer(network[name], field=f"model.network.{name}")
    hidden = _sequence(network["hidden_layers"], field="model.network.hidden_layers")
    if [_positive_integer(value, field="model.network.hidden_layers[]") for value in hidden] != [
        512,
        512,
    ]:
        raise TrainingProvenanceError("model.network.hidden_layers must be [512, 512]")
    _boolean(
        network["separate_actor_critic_encoders"],
        field="model.network.separate_actor_critic_encoders",
    )
    _text(network["activation"], field="model.network.activation")
    _text(network["action_encoding"], field="model.network.action_encoding")
    _validate_model_initialization(network["initialization"])
    locality_count = _integer(network["locality_count"], field="model.network.locality_count")
    tier_count = _integer(
        network["service_tiers_per_locality"],
        field="model.network.service_tiers_per_locality",
    )
    if _integer(network["actor_output_logits"], field="model.network.actor_output_logits") != (
        tier_count**locality_count
    ):
        raise TrainingProvenanceError("actor_output_logits must equal service_tiers^localities")


def _validate_model_initialization(value: Any) -> None:
    field = "model.network.initialization"
    initialization = _mapping(value, field=field)
    _exact_fields(
        initialization,
        {"encoder_linear_layers", "actor_head", "critic_head"},
        field=field,
    )

    encoder_field = f"{field}.encoder_linear_layers"
    encoder = _mapping(initialization["encoder_linear_layers"], field=encoder_field)
    _exact_fields(
        encoder,
        {
            "applies_to",
            "weight_initializer",
            "weight_gain_expression",
            "weight_gain",
            "bias_initializer",
        },
        field=encoder_field,
    )
    normalized_encoder = {
        "applies_to": tuple(_text_list(encoder["applies_to"], field=f"{encoder_field}.applies_to")),
        "weight_initializer": _text(
            encoder["weight_initializer"], field=f"{encoder_field}.weight_initializer"
        ),
        "weight_gain_expression": _text(
            encoder["weight_gain_expression"],
            field=f"{encoder_field}.weight_gain_expression",
        ),
        "weight_gain": _positive_number(
            encoder["weight_gain"], field=f"{encoder_field}.weight_gain"
        ),
        "bias_initializer": _text(
            encoder["bias_initializer"], field=f"{encoder_field}.bias_initializer"
        ),
    }

    actor_field = f"{field}.actor_head"
    actor = _mapping(initialization["actor_head"], field=actor_field)
    _exact_fields(
        actor,
        {
            "parameter_name",
            "weight_initializer",
            "weight_standard_deviation",
            "bias_initializer",
        },
        field=actor_field,
    )
    normalized_actor = {
        "parameter_name": _text(actor["parameter_name"], field=f"{actor_field}.parameter_name"),
        "weight_initializer": _text(
            actor["weight_initializer"], field=f"{actor_field}.weight_initializer"
        ),
        "weight_standard_deviation": _positive_number(
            actor["weight_standard_deviation"],
            field=f"{actor_field}.weight_standard_deviation",
        ),
        "bias_initializer": _text(
            actor["bias_initializer"], field=f"{actor_field}.bias_initializer"
        ),
    }

    critic_field = f"{field}.critic_head"
    critic = _mapping(initialization["critic_head"], field=critic_field)
    _exact_fields(
        critic,
        {"parameter_name", "weight_initializer", "weight_gain", "bias_initializer"},
        field=critic_field,
    )
    normalized_critic = {
        "parameter_name": _text(critic["parameter_name"], field=f"{critic_field}.parameter_name"),
        "weight_initializer": _text(
            critic["weight_initializer"], field=f"{critic_field}.weight_initializer"
        ),
        "weight_gain": _positive_number(critic["weight_gain"], field=f"{critic_field}.weight_gain"),
        "bias_initializer": _text(
            critic["bias_initializer"], field=f"{critic_field}.bias_initializer"
        ),
    }

    normalized = {
        "encoder_linear_layers": normalized_encoder,
        "actor_head": normalized_actor,
        "critic_head": normalized_critic,
    }
    if normalized != _MODEL_INITIALIZATION:
        raise TrainingProvenanceError(
            "model.network.initialization does not match the accepted historical source"
        )


def _validate_software_hardware(environment: dict[str, Any]) -> None:
    _exact_fields(
        environment,
        {
            "scope",
            "training_device",
            "gpu",
            "pytorch",
            "cuda",
            "elapsed_seconds",
            "ppo_seconds",
            "ppo_transitions_per_second",
            "peak_cuda_memory_mb",
        },
        field="software_hardware",
    )
    if _text(environment["scope"], field="software_hardware.scope") != _SELECTED_RUN_SCOPE:
        raise TrainingProvenanceError(
            "software_hardware.scope must identify the selected seed-43 run"
        )
    for name in ("training_device", "gpu", "pytorch", "cuda"):
        _text(environment[name], field=f"software_hardware.{name}")
    for name in (
        "elapsed_seconds",
        "ppo_seconds",
        "ppo_transitions_per_second",
        "peak_cuda_memory_mb",
    ):
        _positive_number(environment[name], field=f"software_hardware.{name}")


def _validate_training_generator(generator: dict[str, Any]) -> None:
    _exact_fields(
        generator,
        {
            "horizon_hours",
            "hour_resolution",
            "system_peak_mw",
            "locality_demand_weights",
            "sector_shares",
            "sector_profiles",
            "sector_profile_normalization",
            "demand_noise",
            "training_supply",
        },
        field="training_generator",
    )
    horizon_hours = _positive_integer(
        generator["horizon_hours"], field="training_generator.horizon_hours"
    )
    _positive_integer(generator["hour_resolution"], field="training_generator.hour_resolution")
    _positive_number(generator["system_peak_mw"], field="training_generator.system_peak_mw")

    locality_names = (
        "Khartoum",
        "Omdurman",
        "Bahri",
        "East Nile",
        "Jebel Awlia",
        "Karari",
        "Ombada",
    )
    locality_rows = _sequence(
        generator["locality_demand_weights"],
        field="training_generator.locality_demand_weights",
    )
    if len(locality_rows) != len(locality_names):
        raise TrainingProvenanceError("locality_demand_weights must contain seven localities")
    observed_names = []
    for index, value in enumerate(locality_rows):
        row = _mapping(value, field=f"locality_demand_weights[{index}]")
        _exact_fields(
            row,
            {"locality", "relative_weight"},
            field=f"locality_demand_weights[{index}]",
        )
        observed_names.append(_text(row["locality"], field=f"locality_demand_weights[{index}]"))
        _positive_number(
            row["relative_weight"],
            field=f"locality_demand_weights[{index}].relative_weight",
        )
    if tuple(observed_names) != locality_names:
        raise TrainingProvenanceError("locality_demand_weights has unexpected locality labels")

    sector_names = ("Hospital", "Water", "Bakery", "Residential", "Commercial")
    sector_rows = _sequence(generator["sector_shares"], field="training_generator.sector_shares")
    if len(sector_rows) != len(sector_names):
        raise TrainingProvenanceError("sector_shares must contain five sectors")
    observed_sectors = []
    shares = []
    for index, value in enumerate(sector_rows):
        row = _mapping(value, field=f"sector_shares[{index}]")
        _exact_fields(row, {"sector", "share"}, field=f"sector_shares[{index}]")
        observed_sectors.append(_text(row["sector"], field=f"sector_shares[{index}].sector"))
        shares.append(_positive_number(row["share"], field=f"sector_shares[{index}].share"))
    if tuple(observed_sectors) != sector_names:
        raise TrainingProvenanceError("sector_shares has unexpected sector labels")
    if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise TrainingProvenanceError("sector_shares must sum to one")
    _validate_sector_profiles(
        generator["sector_profiles"],
        horizon_hours=horizon_hours,
        sector_names=sector_names,
    )
    normalization = _text(
        generator["sector_profile_normalization"],
        field="training_generator.sector_profile_normalization",
    )
    if normalization != _SECTOR_PROFILE_NORMALIZATION:
        raise TrainingProvenanceError(
            "training_generator.sector_profile_normalization does not match the historical source"
        )

    demand_noise = _mapping(generator["demand_noise"], field="training_generator.demand_noise")
    _exact_fields(
        demand_noise,
        {"distribution", "amplitude", "multiplier_min", "multiplier_max"},
        field="training_generator.demand_noise",
    )
    _text(demand_noise["distribution"], field="training_generator.demand_noise.distribution")
    amplitude = _positive_number(
        demand_noise["amplitude"], field="training_generator.demand_noise.amplitude"
    )
    minimum = _positive_number(
        demand_noise["multiplier_min"],
        field="training_generator.demand_noise.multiplier_min",
    )
    maximum = _positive_number(
        demand_noise["multiplier_max"],
        field="training_generator.demand_noise.multiplier_max",
    )
    if not (math.isclose(minimum, 1.0 - amplitude) and math.isclose(maximum, 1.0 + amplitude)):
        raise TrainingProvenanceError("demand noise multiplier bounds do not match amplitude")

    supply = _mapping(generator["training_supply"], field="training_generator.training_supply")
    _exact_fields(
        supply,
        {
            "coupled_to_total_demand",
            "base_fraction_distribution",
            "base_fraction_min",
            "base_fraction_max",
            "shock_start_hour_min",
            "shock_start_hour_max_inclusive",
            "shock_duration_hours_min",
            "shock_duration_hours_max_inclusive",
            "shock_depth_distribution",
            "shock_depth_min",
            "shock_depth_max",
            "fraction_clamp_min",
            "fraction_clamp_max",
            "noise_distribution",
            "noise_amplitude",
            "total_demand_cap_fraction",
        },
        field="training_generator.training_supply",
    )
    _boolean(
        supply["coupled_to_total_demand"],
        field="training_generator.training_supply.coupled_to_total_demand",
    )
    for name in (
        "base_fraction_distribution",
        "shock_depth_distribution",
        "noise_distribution",
    ):
        _text(supply[name], field=f"training_generator.training_supply.{name}")
    for name in ("shock_start_hour_min", "shock_start_hour_max_inclusive"):
        _integer(supply[name], field=f"training_generator.training_supply.{name}")
    for name in ("shock_duration_hours_min", "shock_duration_hours_max_inclusive"):
        _positive_integer(supply[name], field=f"training_generator.training_supply.{name}")
    for name in (
        "base_fraction_min",
        "base_fraction_max",
        "shock_depth_min",
        "shock_depth_max",
        "fraction_clamp_min",
        "fraction_clamp_max",
        "noise_amplitude",
        "total_demand_cap_fraction",
    ):
        _positive_number(supply[name], field=f"training_generator.training_supply.{name}")
    for low_name, high_name in (
        ("base_fraction_min", "base_fraction_max"),
        ("shock_start_hour_min", "shock_start_hour_max_inclusive"),
        ("shock_duration_hours_min", "shock_duration_hours_max_inclusive"),
        ("shock_depth_min", "shock_depth_max"),
        ("fraction_clamp_min", "fraction_clamp_max"),
    ):
        if _number(supply[low_name], field=low_name) > _number(supply[high_name], field=high_name):
            raise TrainingProvenanceError(f"{low_name} must not exceed {high_name}")


def _validate_sector_profiles(
    value: Any,
    *,
    horizon_hours: int,
    sector_names: tuple[str, ...],
) -> None:
    rows = _sequence(value, field="training_generator.sector_profiles")
    if len(rows) != len(sector_names):
        raise TrainingProvenanceError(
            "training_generator.sector_profiles must contain all five sectors"
        )

    observed_names: list[str] = []
    observed_specs: list[tuple[str, float, tuple[tuple[float, float, float], ...]]] = []
    for index, value in enumerate(rows):
        field = f"training_generator.sector_profiles[{index}]"
        row = _mapping(value, field=field)
        _exact_fields(
            row,
            {"sector", "baseline_amplitude", "gaussian_terms"},
            field=field,
        )
        sector = _text(row["sector"], field=f"{field}.sector")
        baseline = _positive_number(
            row["baseline_amplitude"],
            field=f"{field}.baseline_amplitude",
        )
        gaussian_rows = _sequence(row["gaussian_terms"], field=f"{field}.gaussian_terms")
        gaussian_terms: list[tuple[float, float, float]] = []
        for term_index, term_value in enumerate(gaussian_rows):
            term_field = f"{field}.gaussian_terms[{term_index}]"
            term = _mapping(term_value, field=term_field)
            _exact_fields(
                term,
                {"amplitude", "center_hour", "width_hours"},
                field=term_field,
            )
            amplitude = _positive_number(term["amplitude"], field=f"{term_field}.amplitude")
            center_hour = _number(term["center_hour"], field=f"{term_field}.center_hour")
            width_hours = _positive_number(term["width_hours"], field=f"{term_field}.width_hours")
            if not 0.0 <= center_hour < float(horizon_hours):
                raise TrainingProvenanceError(
                    f"{term_field}.center_hour must be in [0, horizon_hours)"
                )
            if width_hours > float(horizon_hours):
                raise TrainingProvenanceError(
                    f"{term_field}.width_hours must not exceed horizon_hours"
                )
            gaussian_terms.append((amplitude, center_hour, width_hours))
        observed_names.append(sector)
        observed_specs.append((sector, baseline, tuple(gaussian_terms)))

    if len(set(observed_names)) != len(observed_names):
        raise TrainingProvenanceError(
            "training_generator.sector_profiles contains a duplicate sector"
        )
    if set(observed_names) != set(sector_names):
        raise TrainingProvenanceError(
            "training_generator.sector_profiles is missing or has an unexpected sector"
        )
    if tuple(observed_names) != sector_names:
        raise TrainingProvenanceError(
            "training_generator.sector_profiles has unexpected sector order"
        )
    if tuple(observed_specs) != _SECTOR_PROFILE_SPECS:
        raise TrainingProvenanceError(
            "training_generator.sector_profiles do not match the accepted historical source"
        )


def _validate_objective(objective: dict[str, Any]) -> None:
    _exact_fields(
        objective,
        {
            "public_role",
            "formula",
            "episode_aggregation",
            "component_definitions",
            "components",
        },
        field="objective",
    )
    for name, expected in _OBJECTIVE_TEXT.items():
        actual = _text(objective[name], field=f"objective.{name}")
        if actual != expected:
            raise TrainingProvenanceError(
                f"objective.{name} does not match the accepted training-objective contract"
            )
    definitions = _mapping(
        objective["component_definitions"],
        field="objective.component_definitions",
    )
    _exact_fields(
        definitions,
        set(_OBJECTIVE_COMPONENT_DEFINITIONS),
        field="objective.component_definitions",
    )
    for symbol, expected_definition in _OBJECTIVE_COMPONENT_DEFINITIONS.items():
        field = f"objective.component_definitions.{symbol}"
        definition = _mapping(definitions[symbol], field=field)
        _exact_fields(definition, set(expected_definition), field=field)
        normalized_definition: dict[str, Any] = {}
        for name, expected in expected_definition.items():
            value = definition[name]
            if isinstance(expected, str):
                normalized_definition[name] = _text(value, field=f"{field}.{name}")
            elif isinstance(expected, list):
                normalized_definition[name] = _text_list(value, field=f"{field}.{name}")
            else:
                normalized_definition[name] = _number(value, field=f"{field}.{name}")
        if normalized_definition != expected_definition:
            raise TrainingProvenanceError(
                f"{field} does not match the accepted training-objective contract"
            )
    components = _mapping(objective["components"], field="objective.components")
    _exact_fields(
        components,
        {
            "critical_service_weight",
            "utilization_weight",
            "jain_fairness_weight",
            "switching_penalty_weight",
            "oversubscription_penalty_weight",
            "scale",
        },
        field="objective.components",
    )
    validated_components = {
        name: _positive_number(value, field=f"objective.components.{name}")
        for name, value in components.items()
    }
    if validated_components != _OBJECTIVE_COMPONENTS:
        raise TrainingProvenanceError(
            "objective.components do not match the accepted training-objective contract"
        )


def _validate_behavior_cloning(cloning: dict[str, Any]) -> None:
    _exact_fields(
        cloning,
        {
            "role",
            "teacher",
            "teacher_samples",
            "teacher_rollout_batch_size",
            "teacher_episode_seed_start",
            "teacher_episode_seed_end_inclusive",
            "exploration_probability",
            "exploration_prng_seed",
            "shuffle_prng_seed",
            "epochs",
            "minibatch_size",
            "minibatches_per_epoch",
            "optimizer_updates",
            "optimized_parameters",
            "loss",
            "max_gradient_norm",
            "gradient_clip_defaults",
            "optimizer",
        },
        field="behavior_cloning_initialization",
    )
    for name in ("role", "teacher", "optimized_parameters", "loss"):
        _text(cloning[name], field=f"behavior_cloning_initialization.{name}")
    for name in (
        "teacher_samples",
        "teacher_rollout_batch_size",
        "teacher_episode_seed_start",
        "teacher_episode_seed_end_inclusive",
        "exploration_prng_seed",
        "shuffle_prng_seed",
        "epochs",
        "minibatch_size",
        "minibatches_per_epoch",
        "optimizer_updates",
    ):
        _positive_integer(cloning[name], field=f"behavior_cloning_initialization.{name}")
    exploration = _number(
        cloning["exploration_probability"],
        field="behavior_cloning_initialization.exploration_probability",
    )
    if not 0.0 <= exploration <= 1.0:
        raise TrainingProvenanceError("exploration_probability must be in [0, 1]")
    max_gradient_norm = _positive_number(
        cloning["max_gradient_norm"],
        field="behavior_cloning_initialization.max_gradient_norm",
    )
    if max_gradient_norm != _BEHAVIOR_CLONING_MAX_GRADIENT_NORM:
        raise TrainingProvenanceError(
            "behavior_cloning_initialization.max_gradient_norm does not match the accepted "
            "historical source"
        )
    _validate_gradient_clip_defaults(
        cloning["gradient_clip_defaults"],
        field="behavior_cloning_initialization.gradient_clip_defaults",
    )
    samples = _integer(cloning["teacher_samples"], field="teacher_samples")
    batch_size = _integer(cloning["minibatch_size"], field="minibatch_size")
    minibatches = math.ceil(samples / batch_size)
    if _integer(cloning["minibatches_per_epoch"], field="minibatches_per_epoch") != minibatches:
        raise TrainingProvenanceError("behavior-cloning minibatches_per_epoch is inconsistent")
    epochs = _integer(cloning["epochs"], field="epochs")
    if _integer(cloning["optimizer_updates"], field="optimizer_updates") != epochs * minibatches:
        raise TrainingProvenanceError("behavior-cloning optimizer_updates is inconsistent")
    episode_samples = (
        _integer(cloning["teacher_rollout_batch_size"], field="teacher_rollout_batch_size") * 24
    )
    episode_count = math.ceil(samples / episode_samples)
    seed_start = _integer(cloning["teacher_episode_seed_start"], field="teacher_episode_seed_start")
    seed_end = _integer(
        cloning["teacher_episode_seed_end_inclusive"],
        field="teacher_episode_seed_end_inclusive",
    )
    if seed_end != seed_start + episode_count - 1:
        raise TrainingProvenanceError("teacher episode seed range is inconsistent")
    if _integer(cloning["exploration_prng_seed"], field="exploration_prng_seed") != (
        seed_start + 10_000
    ):
        raise TrainingProvenanceError("exploration_prng_seed is inconsistent")
    _validate_optimizer(
        _mapping(cloning["optimizer"], field="behavior_cloning_initialization.optimizer"),
        field="behavior_cloning_initialization.optimizer",
        expected=_BEHAVIOR_CLONING_OPTIMIZER,
    )


def _validate_ppo(ppo: dict[str, Any]) -> None:
    _exact_fields(
        ppo,
        {
            "configuration_scope",
            "training_seed",
            "training_seed_scope",
            "iterations",
            "complete_episodes_per_iteration",
            "episode_horizon_hours",
            "transitions_per_iteration",
            "total_episodes",
            "total_transitions",
            "rollout_seed_formula",
            "gamma",
            "gae_lambda",
            "advantage_normalization",
            "learning_semantics",
            "learning_rate",
            "learning_rate_schedule",
            "clip_epsilon",
            "value_coefficient",
            "entropy_coefficient",
            "entropy_coefficient_schedule",
            "entropy_normalization",
            "epochs_per_iteration",
            "minibatch_size",
            "minibatches_per_epoch",
            "optimizer_updates_per_iteration",
            "planned_optimizer_updates",
            "selected_checkpoint_optimizer_updates",
            "selected_checkpoint_scope",
            "actor_max_gradient_norm",
            "critic_max_gradient_norm",
            "gradient_clip_defaults",
            "optimizer",
        },
        field="ppo",
    )
    if _text(ppo["configuration_scope"], field="ppo.configuration_scope") != ("each-candidate-run"):
        raise TrainingProvenanceError("ppo.configuration_scope must identify each candidate run")
    for name in ("training_seed_scope", "selected_checkpoint_scope"):
        if _text(ppo[name], field=f"ppo.{name}") != _SELECTED_RUN_SCOPE:
            raise TrainingProvenanceError(f"ppo.{name} must identify the selected seed-43 run")
    for name in (
        "training_seed",
        "iterations",
        "complete_episodes_per_iteration",
        "episode_horizon_hours",
        "transitions_per_iteration",
        "total_episodes",
        "total_transitions",
        "epochs_per_iteration",
        "minibatch_size",
        "minibatches_per_epoch",
        "optimizer_updates_per_iteration",
        "planned_optimizer_updates",
        "selected_checkpoint_optimizer_updates",
    ):
        _positive_integer(ppo[name], field=f"ppo.{name}")
    for name in (
        "rollout_seed_formula",
        "learning_rate_schedule",
        "entropy_coefficient_schedule",
        "entropy_normalization",
    ):
        _text(ppo[name], field=f"ppo.{name}")
    if not _boolean(ppo["advantage_normalization"], field="ppo.advantage_normalization"):
        raise TrainingProvenanceError(
            "ppo.advantage_normalization does not match the accepted historical source"
        )
    _validate_ppo_learning_semantics(ppo["learning_semantics"])
    for name in (
        "gamma",
        "gae_lambda",
        "learning_rate",
        "clip_epsilon",
        "value_coefficient",
        "entropy_coefficient",
        "actor_max_gradient_norm",
        "critic_max_gradient_norm",
    ):
        _positive_number(ppo[name], field=f"ppo.{name}")
    for name in ("actor_max_gradient_norm", "critic_max_gradient_norm"):
        if _number(ppo[name], field=f"ppo.{name}") != _PPO_MAX_GRADIENT_NORM:
            raise TrainingProvenanceError(
                f"ppo.{name} does not match the accepted historical source"
            )

    iterations = _integer(ppo["iterations"], field="ppo.iterations")
    episodes = _integer(
        ppo["complete_episodes_per_iteration"],
        field="ppo.complete_episodes_per_iteration",
    )
    horizon = _integer(ppo["episode_horizon_hours"], field="ppo.episode_horizon_hours")
    transitions = episodes * horizon
    if _integer(ppo["transitions_per_iteration"], field="ppo.transitions_per_iteration") != (
        transitions
    ):
        raise TrainingProvenanceError("transitions_per_iteration is inconsistent")
    if _integer(ppo["total_episodes"], field="ppo.total_episodes") != iterations * episodes:
        raise TrainingProvenanceError("total_episodes is inconsistent")
    if _integer(ppo["total_transitions"], field="ppo.total_transitions") != (
        iterations * transitions
    ):
        raise TrainingProvenanceError("total_transitions is inconsistent")
    minibatch_size = _integer(ppo["minibatch_size"], field="ppo.minibatch_size")
    minibatches = math.ceil(transitions / minibatch_size)
    if _integer(ppo["minibatches_per_epoch"], field="ppo.minibatches_per_epoch") != minibatches:
        raise TrainingProvenanceError("minibatches_per_epoch is inconsistent")
    epochs = _integer(ppo["epochs_per_iteration"], field="ppo.epochs_per_iteration")
    updates = epochs * minibatches
    if (
        _integer(
            ppo["optimizer_updates_per_iteration"],
            field="ppo.optimizer_updates_per_iteration",
        )
        != updates
    ):
        raise TrainingProvenanceError("optimizer_updates_per_iteration is inconsistent")
    if _integer(ppo["planned_optimizer_updates"], field="ppo.planned_optimizer_updates") != (
        iterations * updates
    ):
        raise TrainingProvenanceError("planned_optimizer_updates is inconsistent")
    _validate_optimizer(
        _mapping(ppo["optimizer"], field="ppo.optimizer"),
        field="ppo.optimizer",
        expected=_PPO_OPTIMIZER,
    )
    _validate_gradient_clip_defaults(
        ppo["gradient_clip_defaults"],
        field="ppo.gradient_clip_defaults",
    )


def _validate_ppo_learning_semantics(value: Any) -> None:
    field = "ppo.learning_semantics"
    semantics = _mapping(value, field=field)
    _exact_fields(
        semantics,
        {
            "advantage_normalization",
            "policy_loss",
            "value_loss",
            "completed_episode_bootstrap",
        },
        field=field,
    )

    advantage_field = f"{field}.advantage_normalization"
    advantage = _mapping(semantics["advantage_normalization"], field=advantage_field)
    _exact_fields(
        advantage,
        {
            "mean_centering",
            "standard_deviation",
            "standard_deviation_unbiased",
            "denominator_clamp_min",
        },
        field=advantage_field,
    )
    normalized_advantage = {
        "mean_centering": _text(
            advantage["mean_centering"], field=f"{advantage_field}.mean_centering"
        ),
        "standard_deviation": _text(
            advantage["standard_deviation"],
            field=f"{advantage_field}.standard_deviation",
        ),
        "standard_deviation_unbiased": _boolean(
            advantage["standard_deviation_unbiased"],
            field=f"{advantage_field}.standard_deviation_unbiased",
        ),
        "denominator_clamp_min": _positive_number(
            advantage["denominator_clamp_min"],
            field=f"{advantage_field}.denominator_clamp_min",
        ),
    }

    policy_field = f"{field}.policy_loss"
    policy = _mapping(semantics["policy_loss"], field=policy_field)
    policy_fields = {
        "probability_ratio",
        "unclipped_surrogate",
        "clipped_ratio_lower_bound",
        "clipped_ratio_upper_bound",
        "clipped_surrogate",
        "surrogate_selection",
        "reduction",
    }
    _exact_fields(policy, policy_fields, field=policy_field)
    normalized_policy = {
        name: _text(policy[name], field=f"{policy_field}.{name}") for name in policy_fields
    }

    value_field = f"{field}.value_loss"
    value_loss = _mapping(semantics["value_loss"], field=value_field)
    _exact_fields(value_loss, {"function", "value_clipping"}, field=value_field)
    normalized_value = {
        "function": _text(value_loss["function"], field=f"{value_field}.function"),
        "value_clipping": _boolean(
            value_loss["value_clipping"], field=f"{value_field}.value_clipping"
        ),
    }

    bootstrap_field = f"{field}.completed_episode_bootstrap"
    bootstrap = _mapping(semantics["completed_episode_bootstrap"], field=bootstrap_field)
    _exact_fields(
        bootstrap,
        {"terminal_next_value", "gae_nonterminal_mask"},
        field=bootstrap_field,
    )
    normalized_bootstrap = {
        "terminal_next_value": _number(
            bootstrap["terminal_next_value"],
            field=f"{bootstrap_field}.terminal_next_value",
        ),
        "gae_nonterminal_mask": _text(
            bootstrap["gae_nonterminal_mask"],
            field=f"{bootstrap_field}.gae_nonterminal_mask",
        ),
    }

    normalized = {
        "advantage_normalization": normalized_advantage,
        "policy_loss": normalized_policy,
        "value_loss": normalized_value,
        "completed_episode_bootstrap": normalized_bootstrap,
    }
    if normalized != _PPO_LEARNING_SEMANTICS:
        raise TrainingProvenanceError(
            "ppo.learning_semantics does not match the accepted historical source"
        )


def _validate_optimizer(
    optimizer: dict[str, Any],
    *,
    field: str,
    expected: Mapping[str, Any],
) -> None:
    _exact_fields(optimizer, set(expected), field=field)
    normalized: dict[str, Any] = {
        "name": _text(optimizer["name"], field=f"{field}.name"),
        "learning_rate": _positive_number(
            optimizer["learning_rate"], field=f"{field}.learning_rate"
        ),
    }
    for name in ("beta1", "beta2", "epsilon"):
        normalized[name] = _positive_number(optimizer[name], field=f"{field}.{name}")
    weight_decay = _number(optimizer["weight_decay"], field=f"{field}.weight_decay")
    if weight_decay < 0.0:
        raise TrainingProvenanceError(f"{field}.weight_decay must be non-negative")
    normalized["weight_decay"] = weight_decay
    for name in (
        "amsgrad",
        "maximize",
        "capturable",
        "differentiable",
        "decoupled_weight_decay",
    ):
        normalized[name] = _boolean(optimizer[name], field=f"{field}.{name}")
    for name in ("foreach", "fused"):
        value = optimizer[name]
        if value is not None:
            _boolean(value, field=f"{field}.{name}")
        normalized[name] = value
    if normalized != dict(expected):
        raise TrainingProvenanceError(
            f"{field} does not match the accepted historical source and PyTorch 2.8 defaults"
        )


def _validate_gradient_clip_defaults(value: Any, *, field: str) -> None:
    defaults = _mapping(value, field=field)
    _exact_fields(defaults, set(_GRADIENT_CLIP_DEFAULTS), field=field)
    foreach = defaults["foreach"]
    if foreach is not None:
        _boolean(foreach, field=f"{field}.foreach")
    normalized = {
        "norm_type": _positive_number(defaults["norm_type"], field=f"{field}.norm_type"),
        "error_if_nonfinite": _boolean(
            defaults["error_if_nonfinite"],
            field=f"{field}.error_if_nonfinite",
        ),
        "foreach": foreach,
    }
    if normalized != _GRADIENT_CLIP_DEFAULTS:
        raise TrainingProvenanceError(
            f"{field} does not match the accepted historical source and PyTorch 2.8 defaults"
        )


def _validate_selection(selection: dict[str, Any]) -> None:
    _exact_fields(
        selection,
        {
            "experiment_summary_path",
            "candidate_training_seeds",
            "validation_seeds",
            "validation_shortage_levels",
            "validation_batch_size_per_shortage",
            "validation_episodes_per_checkpoint",
            "validate_every_iterations",
            "selection_score_formula",
            "promotion",
            "selection_rule",
            "candidate_best_scores",
            "selected_seed",
            "selected_iteration",
            "selected_score",
        },
        field="validation_and_selection",
    )
    _validate_promotion(selection["promotion"])
    rule = _mapping(
        selection["selection_rule"],
        field="validation_and_selection.selection_rule",
    )
    _exact_fields(rule, set(_SELECTION_RULE), field="validation_and_selection.selection_rule")
    normalized_rule = {
        name: _text(rule[name], field=f"validation_and_selection.selection_rule.{name}")
        for name in _SELECTION_RULE
    }
    if normalized_rule != _SELECTION_RULE:
        raise TrainingProvenanceError(
            "validation_and_selection.selection_rule does not match the bound source"
        )
    _relative_path(
        selection["experiment_summary_path"],
        field="validation_and_selection.experiment_summary_path",
    )
    candidate_seeds = _integer_list(
        selection["candidate_training_seeds"],
        field="validation_and_selection.candidate_training_seeds",
    )
    validation_seeds = _integer_list(
        selection["validation_seeds"],
        field="validation_and_selection.validation_seeds",
    )
    if len(candidate_seeds) != len(set(candidate_seeds)) or len(validation_seeds) != len(
        set(validation_seeds)
    ):
        raise TrainingProvenanceError("training and validation seed lists must be unique")
    shortage_levels = _text_list(
        selection["validation_shortage_levels"],
        field="validation_and_selection.validation_shortage_levels",
    )
    if shortage_levels != ["Mild", "Moderate", "Severe"]:
        raise TrainingProvenanceError("validation_shortage_levels must use all three levels")
    batch_size = _positive_integer(
        selection["validation_batch_size_per_shortage"],
        field="validation_and_selection.validation_batch_size_per_shortage",
    )
    expected_episodes = batch_size * len(shortage_levels)
    if (
        _integer(
            selection["validation_episodes_per_checkpoint"],
            field="validation_and_selection.validation_episodes_per_checkpoint",
        )
        != expected_episodes
    ):
        raise TrainingProvenanceError("validation_episodes_per_checkpoint is inconsistent")
    _positive_integer(
        selection["validate_every_iterations"],
        field="validation_and_selection.validate_every_iterations",
    )
    _text(
        selection["selection_score_formula"],
        field="validation_and_selection.selection_score_formula",
    )
    score_rows = _sequence(
        selection["candidate_best_scores"],
        field="validation_and_selection.candidate_best_scores",
    )
    scores: dict[int, float] = {}
    for index, value in enumerate(score_rows):
        row = _mapping(value, field=f"candidate_best_scores[{index}]")
        _exact_fields(row, {"seed", "score"}, field=f"candidate_best_scores[{index}]")
        seed = _positive_integer(row["seed"], field=f"candidate_best_scores[{index}].seed")
        score = _number(row["score"], field=f"candidate_best_scores[{index}].score")
        if seed in scores:
            raise TrainingProvenanceError("candidate_best_scores contains a duplicate seed")
        scores[seed] = score
    if list(scores) != candidate_seeds:
        raise TrainingProvenanceError("candidate_best_scores does not match candidate seeds")
    selected_seed = _positive_integer(
        selection["selected_seed"], field="validation_and_selection.selected_seed"
    )
    selected_score = _number(
        selection["selected_score"], field="validation_and_selection.selected_score"
    )
    if selected_seed not in scores or scores[selected_seed] != selected_score:
        raise TrainingProvenanceError("selected score does not match the selected seed")
    if selected_score != max(scores.values()):
        raise TrainingProvenanceError("selected seed does not have the best validation score")
    _positive_integer(
        selection["selected_iteration"],
        field="validation_and_selection.selected_iteration",
    )


def _validate_promotion(value: Any) -> None:
    field = "validation_and_selection.promotion"
    promotion = _mapping(value, field=field)
    _exact_fields(
        promotion,
        {
            "eligible",
            "incumbent",
            "eligibility_rule",
            "reward_gate_evidence_role",
            "thresholds",
            "checks",
        },
        field=field,
    )
    if not _boolean(promotion["eligible"], field=f"{field}.eligible"):
        raise TrainingProvenanceError(f"{field}.eligible must match the promoted experiment")
    if _text(promotion["incumbent"], field=f"{field}.incumbent") != "s1-600-constant":
        raise TrainingProvenanceError(f"{field}.incumbent does not match the archive")
    if _text(promotion["eligibility_rule"], field=f"{field}.eligibility_rule") != (
        _PROMOTION_ELIGIBILITY_RULE
    ):
        raise TrainingProvenanceError(f"{field}.eligibility_rule does not match the bound source")
    if (
        _text(
            promotion["reward_gate_evidence_role"],
            field=f"{field}.reward_gate_evidence_role",
        )
        != _REWARD_GATE_EVIDENCE_ROLE
    ):
        raise TrainingProvenanceError(
            f"{field}.reward_gate_evidence_role must bound reward gates to historical selection"
        )

    thresholds = _mapping(promotion["thresholds"], field=f"{field}.thresholds")
    _exact_fields(thresholds, set(_PROMOTION_THRESHOLDS), field=f"{field}.thresholds")
    normalized_thresholds = {
        name: _number(thresholds[name], field=f"{field}.thresholds.{name}")
        for name in _PROMOTION_THRESHOLDS
    }
    if normalized_thresholds != _PROMOTION_THRESHOLDS:
        raise TrainingProvenanceError(f"{field}.thresholds do not match the bound source")

    checks = _mapping(promotion["checks"], field=f"{field}.checks")
    _exact_fields(checks, set(_PROMOTION_CHECKS), field=f"{field}.checks")
    normalized_checks: dict[str, bool | int | float] = {}
    for name, expected in _PROMOTION_CHECKS.items():
        value = checks[name]
        if type(expected) is bool:
            normalized_checks[name] = _boolean(value, field=f"{field}.checks.{name}")
        elif type(expected) is int:
            normalized_checks[name] = _integer(value, field=f"{field}.checks.{name}")
        else:
            normalized_checks[name] = _number(value, field=f"{field}.checks.{name}")
    if normalized_checks != _PROMOTION_CHECKS:
        raise TrainingProvenanceError(f"{field}.checks do not match the bound experiment summary")
    boolean_checks = [value for value in normalized_checks.values() if type(value) is bool]
    if _boolean(promotion["eligible"], field=f"{field}.eligible") != all(boolean_checks):
        raise TrainingProvenanceError(f"{field}.eligible is inconsistent with boolean checks")


def _validate_evaluation_seed_disclosure(
    disclosure: dict[str, Any],
    *,
    ppo: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    _exact_fields(
        disclosure,
        {
            "seeds",
            "used_for_gradient_updates",
            "used_for_periodic_validation",
            "used_for_promotion_and_model_selection",
            "untouched_final_test",
            "interpretation",
        },
        field="evaluation_seed_disclosure",
    )
    seeds = _integer_list(disclosure["seeds"], field="evaluation_seed_disclosure.seeds")
    if len(seeds) != len(set(seeds)):
        raise TrainingProvenanceError("evaluation seed list must be unique")
    expected_bools = {
        "used_for_gradient_updates": False,
        "used_for_periodic_validation": False,
        "used_for_promotion_and_model_selection": True,
        "untouched_final_test": False,
    }
    for name, expected in expected_bools.items():
        if _boolean(disclosure[name], field=f"evaluation_seed_disclosure.{name}") is not expected:
            raise TrainingProvenanceError(f"evaluation_seed_disclosure.{name} is inconsistent")
    _text(disclosure["interpretation"], field="evaluation_seed_disclosure.interpretation")
    validation_seeds = set(
        _integer_list(
            selection["validation_seeds"],
            field="validation_and_selection.validation_seeds",
        )
    )
    training_seed = _integer(ppo["training_seed"], field="ppo.training_seed")
    iterations = _integer(ppo["iterations"], field="ppo.iterations")
    rollout_seeds = set(range(training_seed + 1001, training_seed + 1001 + iterations))
    if set(seeds) & (validation_seeds | rollout_seeds):
        raise TrainingProvenanceError("evaluation seeds overlap gradient or validation seeds")


def _validate_convergence(convergence: dict[str, Any], *, selection: dict[str, Any]) -> None:
    _exact_fields(
        convergence,
        {
            "scope",
            "untrained_score",
            "after_behavior_cloning_score",
            "ppo_iteration_100_score",
            "selected_iteration_510_score",
            "final_iteration_600_score",
            "validation_series",
        },
        field="convergence",
    )
    if _text(convergence["scope"], field="convergence.scope") != _SELECTED_RUN_SCOPE:
        raise TrainingProvenanceError("convergence.scope must identify the selected seed-43 run")
    for name in (
        "untrained_score",
        "after_behavior_cloning_score",
        "ppo_iteration_100_score",
        "selected_iteration_510_score",
        "final_iteration_600_score",
    ):
        _number(convergence[name], field=f"convergence.{name}")
    series = _sequence(convergence["validation_series"], field="convergence.validation_series")
    expected_iterations = list(range(5, 601, 5))
    observed: dict[int, float] = {}
    for index, value in enumerate(series):
        row = _mapping(value, field=f"convergence.validation_series[{index}]")
        _exact_fields(
            row,
            {"iteration", "training_objective_score"},
            field=f"convergence.validation_series[{index}]",
        )
        iteration = _positive_integer(
            row["iteration"], field=f"convergence.validation_series[{index}].iteration"
        )
        observed[iteration] = _number(
            row["training_objective_score"],
            field=f"convergence.validation_series[{index}].training_objective_score",
        )
    if list(observed) != expected_iterations:
        raise TrainingProvenanceError(
            "convergence validation_series must contain every five-iteration checkpoint"
        )
    milestone_fields = {
        100: "ppo_iteration_100_score",
        510: "selected_iteration_510_score",
        600: "final_iteration_600_score",
    }
    for iteration, field in milestone_fields.items():
        if observed[iteration] != _number(convergence[field], field=f"convergence.{field}"):
            raise TrainingProvenanceError(f"convergence.{field} does not match validation_series")
    selected_iteration = _integer(
        selection["selected_iteration"], field="validation_and_selection.selected_iteration"
    )
    selected_score = _number(
        selection["selected_score"], field="validation_and_selection.selected_score"
    )
    if observed[selected_iteration] != selected_score or max(observed.values()) != selected_score:
        raise TrainingProvenanceError(
            "convergence series does not identify the selected checkpoint"
        )


def _validate_source_bindings(value: Any) -> None:
    rows = _sequence(value, field="source_bindings")
    declared: list[tuple[str, str, str]] = []
    observed_paths: set[str] = set()
    for index, value in enumerate(rows):
        row = _mapping(value, field=f"source_bindings[{index}]")
        _exact_fields(row, {"path", "role", "sha256"}, field=f"source_bindings[{index}]")
        path = _relative_path(row["path"], field=f"source_bindings[{index}].path")
        role = _text(row["role"], field=f"source_bindings[{index}].role")
        digest = _sha256(row["sha256"], field=f"source_bindings[{index}].sha256")
        if path in observed_paths:
            raise TrainingProvenanceError("source_bindings contains a duplicate path")
        observed_paths.add(path)
        declared.append((path, role, digest))
    if tuple(declared) != _SOURCE_BINDINGS:
        raise TrainingProvenanceError(
            "source_bindings do not match the accepted source path, role, order, and SHA-256 values"
        )


def _validate_model_manifest_binding(
    payload: dict[str, Any],
    *,
    provenance_path: Path,
    model_manifest_path: Path,
) -> None:
    try:
        manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingProvenanceError(f"model manifest cannot be read: {error}") from error
    manifest_mapping = _mapping(manifest, field="model manifest")
    if (
        manifest_mapping.get("schema") != "gridguard.ppo_actor"
        or manifest_mapping.get("schema_version") != 2
    ):
        raise TrainingProvenanceError("model manifest must use gridguard.ppo_actor schema v2")
    pointer = _mapping(
        manifest_mapping.get("training_provenance"),
        field="model manifest.training_provenance",
    )
    _exact_fields(
        pointer,
        {"filename", "format", "schema", "schema_version", "sha256", "size_bytes"},
        field="model manifest.training_provenance",
    )
    if (
        pointer["filename"] != TRAINING_PROVENANCE_FILENAME
        or pointer["format"] != "yaml"
        or pointer["schema"] != TRAINING_PROVENANCE_SCHEMA
        or pointer["schema_version"] != TRAINING_PROVENANCE_SCHEMA_VERSION
    ):
        raise TrainingProvenanceError("model manifest training_provenance pointer is invalid")
    expected_path = model_manifest_path.parent / pointer["filename"]
    if provenance_path.resolve() != expected_path.resolve():
        raise TrainingProvenanceError("model manifest points to a different training provenance")
    try:
        provenance_bytes = provenance_path.read_bytes()
    except OSError as error:
        raise TrainingProvenanceError(f"training provenance cannot be hashed: {error}") from error
    if len(provenance_bytes) != pointer["size_bytes"]:
        raise TrainingProvenanceError("model manifest training provenance size mismatch")
    if hashlib.sha256(provenance_bytes).hexdigest() != pointer["sha256"]:
        raise TrainingProvenanceError("model manifest training provenance SHA-256 mismatch")

    actor = _mapping(manifest_mapping.get("artifact"), field="model manifest.artifact")
    source = _mapping(manifest_mapping.get("source"), field="model manifest.source")
    yaml_actor = _mapping(payload["model"]["actor"], field="model.actor")
    yaml_checkpoint = _mapping(payload["model"]["checkpoint"], field="model.checkpoint")
    if any(
        actor.get(name) != yaml_actor.get(name) for name in ("filename", "sha256", "size_bytes")
    ):
        raise TrainingProvenanceError("model manifest actor does not match training provenance")
    checkpoint_pairs = {
        "checkpoint_path": "path",
        "checkpoint_sha256": "sha256",
        "source_revision": "source_revision",
    }
    if any(
        source.get(manifest_name) != yaml_checkpoint.get(yaml_name)
        for manifest_name, yaml_name in checkpoint_pairs.items()
    ):
        raise TrainingProvenanceError("model manifest source does not match training provenance")


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingProvenanceError(f"{field} must be a mapping")
    return dict(value)


def _sequence(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TrainingProvenanceError(f"{field} must be a sequence")
    return list(value)


def _exact_fields(mapping: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(mapping)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        display = sorted(repr(name) for name in unknown)
        raise TrainingProvenanceError(f"{field} has unknown fields: {display!r}")
    if missing:
        raise TrainingProvenanceError(f"{field} is missing fields: {sorted(missing)!r}")


def _text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise TrainingProvenanceError(f"{field} must be non-empty text")
    return value


def _boolean(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise TrainingProvenanceError(f"{field} must be a boolean")
    return value


def _integer(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise TrainingProvenanceError(f"{field} must be an integer, not a boolean")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    result = _integer(value, field=field)
    if result <= 0:
        raise TrainingProvenanceError(f"{field} must be positive")
    return result


def _number(value: Any, *, field: str) -> float:
    if type(value) not in (int, float):
        raise TrainingProvenanceError(f"{field} must be a number, not a boolean")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingProvenanceError(f"{field} must be finite")
    return result


def _positive_number(value: Any, *, field: str) -> float:
    result = _number(value, field=field)
    if result <= 0.0:
        raise TrainingProvenanceError(f"{field} must be positive")
    return result


def _integer_list(value: Any, *, field: str) -> list[int]:
    return [
        _integer(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    ]


def _text_list(value: Any, *, field: str) -> list[str]:
    return [
        _text(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    ]


def _relative_path(value: Any, *, field: str) -> str:
    path_text = _text(value, field=field)
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or path_text == "."
        or ".." in path.parts
        or "." in path.parts
        or ":" in path_text
        or "\\" in path_text
        or path_text.startswith("/")
    ):
        raise TrainingProvenanceError(f"{field} must be a safe relative path")
    return path_text


def _sha256(value: Any, *, field: str) -> str:
    digest = _text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise TrainingProvenanceError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _reject_nonfinite_numbers(value: Any, *, field: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise TrainingProvenanceError(f"{field} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, field=f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, field=f"{field}[{index}]")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--summary-json", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the bound record and emit one compact summary JSON object."""

    args = _parse_args(argv)
    try:
        payload = load_training_provenance(
            args.path,
            model_manifest_path=args.model_manifest,
        )
        summary = build_training_summary(payload, provenance_path=args.path)
    except TrainingProvenanceError as error:
        print(f"Training provenance validation: FAIL - {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
