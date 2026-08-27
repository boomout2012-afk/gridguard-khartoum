"""Validated, read-only evidence boundary for the presentation interface."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from gridguard import __version__
from gridguard.claim_contract import (
    RESULTS_SCOPE_LIMITATION as RESULTS_SCOPE_LIMITATION,
)
from gridguard.domain import HORIZON_HOURS
from gridguard.resources import runtime_root
from gridguard.robustness import (
    PRIMARY_METRICS,
    ROBUSTNESS_ARTIFACT_ID,
    ROBUSTNESS_EVIDENCE_ROLE,
    ROBUSTNESS_LEGACY_ARTIFACT_ID,
    ROBUSTNESS_LEGACY_EVIDENCE_ROLE,
    ROBUSTNESS_LEGACY_SCHEMA_VERSION,
    ROBUSTNESS_SCHEMA_NAME,
    ROBUSTNESS_SCHEMA_VERSION,
    ROBUSTNESS_SUPPLEMENT_SHA256,
    RobustnessArtifactError,
    verify_robustness_supplement,
)

CONTROLLER_NAMES = ("PPO", "Fixed Rotation")
RELEASE_ID = f"gridguard-khartoum-v{__version__}"

_BASE_EVALUATION_FIELDS = frozenset({"controller_means", "paired_inference", "power_balance"})
_CONTROLLER_MEAN_FIELDS = frozenset((*PRIMARY_METRICS, "power_balance_violation_rate"))
_INTERVAL_FIELDS = frozenset({"mean_delta", "ci95_lower", "ci95_upper", "result_classification"})
_METRIC_PRESENTATION = {
    "critical_outage_locality_hours": (
        "Critical-outage locality-hours",
        "locality-hours",
        1.0,
        "lower_is_better",
    ),
    "fairness": ("Jain fairness", "%", 100.0, "higher_is_better"),
    "utilization": ("Utilization", "%", 100.0, "higher_is_better"),
}
_METRIC_ORDER = tuple(_METRIC_PRESENTATION)
if _METRIC_ORDER != PRIMARY_METRICS:  # pragma: no cover - import-time contract guard
    raise RuntimeError("presentation primary metric order does not match robustness evidence")


class PresentationDataError(ValueError):
    """Raised when verified evidence cannot be presented without changing its meaning."""


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One documented-seed-disjoint metric and its paired PPO-minus-baseline inference."""

    key: str
    label: str
    unit: str
    display_scale: float
    direction: str
    role: str
    ppo: float
    fixed_rotation: float
    mean_delta: float
    ci95_lower: float
    ci95_upper: float
    result_classification: str


@dataclass(frozen=True, slots=True)
class TrainingRecordRow:
    """One preformatted reviewer row; Qt renders it without deriving training values."""

    label: str
    value: str

    @property
    def text(self) -> str:
        return f"{self.label} · {self.value}"


@dataclass(frozen=True, slots=True)
class HistoricalTrainingRecord:
    """Immutable presentation projection of the strictly validated PPO training record."""

    provenance_path: str
    observation_dim: int
    hidden_layers: tuple[int, ...]
    actor_output_logits: int
    activation: str
    action_encoding: str
    behavior_cloning_teacher_samples: int
    behavior_cloning_rollout_batch_size: int
    behavior_cloning_minibatch_size: int
    behavior_cloning_epochs: int
    behavior_cloning_optimizer_updates: int
    ppo_iterations: int
    ppo_episodes_per_iteration: int
    ppo_transitions_per_iteration: int
    ppo_total_episodes: int
    ppo_total_transitions: int
    candidate_experiment_iterations: int
    candidate_experiment_episodes: int
    candidate_experiment_transitions: int
    ppo_minibatch_size: int
    ppo_epochs_per_iteration: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    training_seed: int
    candidate_training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    selected_seed: int
    selected_iteration: int
    selected_checkpoint_optimizer_updates: int

    @classmethod
    def from_validated_payload(
        cls,
        payload: Mapping[str, object],
        *,
        model_manifest: Mapping[str, object],
        provenance_path: str,
    ) -> Self:
        """Project an already strict-validated YAML payload and verified model manifest."""

        canonical_path = _text(provenance_path, field="training provenance path")
        if canonical_path != "data/model/training-provenance.yaml":
            raise PresentationDataError(
                "training provenance path must be data/model/training-provenance.yaml"
            )

        model = _mapping(payload.get("model"), field="training model")
        network = _mapping(model.get("network"), field="training model.network")
        cloning = _mapping(
            payload.get("behavior_cloning_initialization"),
            field="behavior_cloning_initialization",
        )
        ppo = _mapping(payload.get("ppo"), field="ppo")
        selection = _mapping(
            payload.get("validation_and_selection"),
            field="validation_and_selection",
        )
        experiment_scope = _mapping(payload.get("experiment_scope"), field="experiment_scope")
        candidate_experiment = _mapping(
            experiment_scope.get("candidate_experiment"),
            field="experiment_scope.candidate_experiment",
        )
        aggregate_workload = _mapping(
            candidate_experiment.get("aggregate_workload"),
            field="experiment_scope.candidate_experiment.aggregate_workload",
        )
        architecture = _mapping(
            model_manifest.get("architecture"),
            field="model manifest architecture",
        )

        observation_dim = _positive_int(
            network.get("observation_dim"), field="model.network.observation_dim"
        )
        hidden_layers = _integer_sequence(
            network.get("hidden_layers"), field="model.network.hidden_layers"
        )
        actor_output_logits = _positive_int(
            network.get("actor_output_logits"), field="model.network.actor_output_logits"
        )
        manifest_observation_dim = _positive_int(
            architecture.get("observation_dim"),
            field="model manifest architecture.observation_dim",
        )
        manifest_hidden_size = _positive_int(
            architecture.get("hidden_size"), field="model manifest architecture.hidden_size"
        )
        manifest_action_count = _positive_int(
            architecture.get("action_count"), field="model manifest architecture.action_count"
        )
        activation = _text(network.get("activation"), field="model.network.activation")
        action_encoding = _text(
            network.get("action_encoding"), field="model.network.action_encoding"
        )
        if (
            observation_dim != manifest_observation_dim
            or not hidden_layers
            or any(size != manifest_hidden_size for size in hidden_layers)
            or actor_output_logits != manifest_action_count
            or activation
            != _text(architecture.get("activation"), field="model manifest activation")
            or action_encoding
            != _text(
                architecture.get("action_encoding"),
                field="model manifest action_encoding",
            )
        ):
            raise PresentationDataError(
                "training YAML actor topology does not match the verified model manifest"
            )

        return cls(
            provenance_path=canonical_path,
            observation_dim=observation_dim,
            hidden_layers=hidden_layers,
            actor_output_logits=actor_output_logits,
            activation=activation,
            action_encoding=action_encoding,
            behavior_cloning_teacher_samples=_positive_int(
                cloning.get("teacher_samples"),
                field="behavior_cloning_initialization.teacher_samples",
            ),
            behavior_cloning_rollout_batch_size=_positive_int(
                cloning.get("teacher_rollout_batch_size"),
                field="behavior_cloning_initialization.teacher_rollout_batch_size",
            ),
            behavior_cloning_minibatch_size=_positive_int(
                cloning.get("minibatch_size"),
                field="behavior_cloning_initialization.minibatch_size",
            ),
            behavior_cloning_epochs=_positive_int(
                cloning.get("epochs"), field="behavior_cloning_initialization.epochs"
            ),
            behavior_cloning_optimizer_updates=_positive_int(
                cloning.get("optimizer_updates"),
                field="behavior_cloning_initialization.optimizer_updates",
            ),
            ppo_iterations=_positive_int(ppo.get("iterations"), field="ppo.iterations"),
            ppo_episodes_per_iteration=_positive_int(
                ppo.get("complete_episodes_per_iteration"),
                field="ppo.complete_episodes_per_iteration",
            ),
            ppo_transitions_per_iteration=_positive_int(
                ppo.get("transitions_per_iteration"),
                field="ppo.transitions_per_iteration",
            ),
            ppo_total_episodes=_positive_int(ppo.get("total_episodes"), field="ppo.total_episodes"),
            ppo_total_transitions=_positive_int(
                ppo.get("total_transitions"), field="ppo.total_transitions"
            ),
            candidate_experiment_iterations=_positive_int(
                aggregate_workload.get("ppo_iterations"),
                field="experiment_scope.candidate_experiment.aggregate_workload.ppo_iterations",
            ),
            candidate_experiment_episodes=_positive_int(
                aggregate_workload.get("complete_episodes"),
                field="experiment_scope.candidate_experiment.aggregate_workload.complete_episodes",
            ),
            candidate_experiment_transitions=_positive_int(
                aggregate_workload.get("transitions"),
                field="experiment_scope.candidate_experiment.aggregate_workload.transitions",
            ),
            ppo_minibatch_size=_positive_int(ppo.get("minibatch_size"), field="ppo.minibatch_size"),
            ppo_epochs_per_iteration=_positive_int(
                ppo.get("epochs_per_iteration"), field="ppo.epochs_per_iteration"
            ),
            learning_rate=_finite_number(ppo.get("learning_rate"), field="ppo.learning_rate"),
            gamma=_finite_number(ppo.get("gamma"), field="ppo.gamma"),
            gae_lambda=_finite_number(ppo.get("gae_lambda"), field="ppo.gae_lambda"),
            clip_epsilon=_finite_number(ppo.get("clip_epsilon"), field="ppo.clip_epsilon"),
            training_seed=_positive_int(ppo.get("training_seed"), field="ppo.training_seed"),
            candidate_training_seeds=_integer_sequence(
                selection.get("candidate_training_seeds"),
                field="validation_and_selection.candidate_training_seeds",
            ),
            validation_seeds=_integer_sequence(
                selection.get("validation_seeds"),
                field="validation_and_selection.validation_seeds",
            ),
            selected_seed=_positive_int(
                selection.get("selected_seed"),
                field="validation_and_selection.selected_seed",
            ),
            selected_iteration=_positive_int(
                selection.get("selected_iteration"),
                field="validation_and_selection.selected_iteration",
            ),
            selected_checkpoint_optimizer_updates=_positive_int(
                ppo.get("selected_checkpoint_optimizer_updates"),
                field="ppo.selected_checkpoint_optimizer_updates",
            ),
        )

    @property
    def actor_topology(self) -> str:
        dimensions = (self.observation_dim, *self.hidden_layers)
        prefix = " -> ".join(f"{dimension:,}" for dimension in dimensions)
        return f"{prefix} -> {self.actor_output_logits:,} logits"

    @property
    def reviewer_rows(self) -> tuple[TrainingRecordRow, ...]:
        candidates = ", ".join(str(seed) for seed in self.candidate_training_seeds)
        validation = ", ".join(str(seed) for seed in self.validation_seeds)
        return (
            TrainingRecordRow(
                "Actor topology",
                f"{self.actor_topology} · {self.activation} · {self.action_encoding}",
            ),
            TrainingRecordRow(
                "Behavior cloning",
                f"{self.behavior_cloning_teacher_samples:,} teacher samples · "
                f"rollout batch {self.behavior_cloning_rollout_batch_size:,} · "
                f"minibatch {self.behavior_cloning_minibatch_size:,} · "
                f"{self.behavior_cloning_epochs:,} epochs · "
                f"{self.behavior_cloning_optimizer_updates:,} optimizer updates",
            ),
            TrainingRecordRow(
                f"Selected seed-{self.selected_seed} run",
                f"{self.ppo_iterations:,} iterations · "
                f"{self.ppo_total_episodes:,} complete episodes · "
                f"{self.ppo_total_transitions:,} transitions",
            ),
            TrainingRecordRow(
                "Three-seed candidate experiment",
                f"{self.candidate_experiment_iterations:,} iterations · "
                f"{self.candidate_experiment_episodes:,} complete episodes · "
                f"{self.candidate_experiment_transitions:,} transitions",
            ),
            TrainingRecordRow(
                "PPO batching",
                f"{self.ppo_episodes_per_iteration:,} episodes/iteration · "
                f"minibatch {self.ppo_minibatch_size:,} · "
                f"{self.ppo_epochs_per_iteration:,} epochs/iteration",
            ),
            TrainingRecordRow(
                "Learning",
                f"LR {self.learning_rate:.4f} · gamma {self.gamma:.2f} · "
                f"GAE lambda {self.gae_lambda:.2f} · clip epsilon {self.clip_epsilon:.2f}",
            ),
            TrainingRecordRow(
                "Seeds and selection",
                f"training {self.training_seed} · candidates {candidates} · "
                f"validation {validation} · selected seed {self.selected_seed} at iteration "
                f"{self.selected_iteration}",
            ),
        )

    @property
    def accessible_summary(self) -> str:
        rows = ". ".join(row.text for row in self.reviewer_rows)
        return (
            f"Verified presentation summary of canonical YAML {self.provenance_path}. {rows}. "
            "Historical record, not an executable retraining configuration."
        )


CANONICAL_TRAINING_RECORD = HistoricalTrainingRecord(
    provenance_path="data/model/training-provenance.yaml",
    observation_dim=32,
    hidden_layers=(512, 512),
    actor_output_logits=2_187,
    activation="tanh",
    action_encoding="little-endian-base-3",
    behavior_cloning_teacher_samples=32_768,
    behavior_cloning_rollout_batch_size=128,
    behavior_cloning_minibatch_size=2_048,
    behavior_cloning_epochs=100,
    behavior_cloning_optimizer_updates=1_600,
    ppo_iterations=600,
    ppo_episodes_per_iteration=256,
    ppo_transitions_per_iteration=6_144,
    ppo_total_episodes=153_600,
    ppo_total_transitions=3_686_400,
    candidate_experiment_iterations=1_800,
    candidate_experiment_episodes=460_800,
    candidate_experiment_transitions=11_059_200,
    ppo_minibatch_size=2_048,
    ppo_epochs_per_iteration=4,
    learning_rate=0.0001,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    training_seed=43,
    candidate_training_seeds=(42, 43, 44),
    validation_seeds=(70_000, 70_001, 70_002),
    selected_seed=43,
    selected_iteration=510,
    selected_checkpoint_optimizer_updates=6_120,
)


@dataclass(frozen=True, slots=True)
class PresentationSnapshot:
    """Documented-seed-disjoint values approved for display outside Qt widgets."""

    artifact_id: str
    evidence_role: str
    artifact_sha256: str
    release_id: str
    controller_names: tuple[str, str]
    count_statement: str
    independent_base_draw_count: int
    shortage_level_count: int
    matched_scenario_count: int
    controller_evaluation_count: int
    horizon_hours: int
    metrics: tuple[MetricComparison, ...]
    maximum_power_balance_violation_rate: float
    limitations: tuple[str, ...]
    training_record: HistoricalTrainingRecord

    @property
    def benchmark_id(self) -> str:
        """Retain the shell's evidence-identifier interface during the artifact migration."""

        return self.artifact_id

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> Self:
        """Validate the locked robustness fields consumed by the presentation."""

        schema = _mapping(payload.get("schema"), field="schema")
        _exact_fields(schema, frozenset({"name", "version"}), field="schema")
        schema_version = schema.get("version")
        if schema.get("name") != ROBUSTNESS_SCHEMA_NAME or schema_version not in {
            ROBUSTNESS_LEGACY_SCHEMA_VERSION,
            ROBUSTNESS_SCHEMA_VERSION,
        }:
            raise PresentationDataError(
                "evidence must identify gridguard robustness supplement version 1 or 2"
            )

        evidence_role = _text(payload.get("evidence_role"), field="evidence_role")
        expected_evidence_role = (
            ROBUSTNESS_LEGACY_EVIDENCE_ROLE
            if schema_version == ROBUSTNESS_LEGACY_SCHEMA_VERSION
            else ROBUSTNESS_EVIDENCE_ROLE
        )
        if evidence_role != expected_evidence_role:
            raise PresentationDataError(
                f"evidence_role must be {expected_evidence_role} for schema version "
                f"{schema_version}"
            )
        provenance = _mapping(payload.get("provenance"), field="provenance")
        artifact_id = _text(provenance.get("artifact_id"), field="provenance.artifact_id")
        expected_artifact_id = (
            ROBUSTNESS_LEGACY_ARTIFACT_ID
            if schema_version == ROBUSTNESS_LEGACY_SCHEMA_VERSION
            else ROBUSTNESS_ARTIFACT_ID
        )
        if artifact_id != expected_artifact_id:
            raise PresentationDataError(
                f"provenance.artifact_id must be {expected_artifact_id} for schema version "
                f"{schema_version}"
            )

        protocol = _mapping(payload.get("protocol"), field="protocol")
        _validate_controllers(protocol.get("controllers"))
        if protocol.get("delta_definition") != "PPO - Fixed Rotation":
            raise PresentationDataError("protocol.delta_definition must be PPO - Fixed Rotation")
        independent_draws = _exact_positive_int(
            protocol.get("independent_base_draw_count"),
            expected=320,
            field="protocol.independent_base_draw_count",
        )
        matched_conditions = _exact_positive_int(
            protocol.get("matched_scenario_count"),
            expected=960,
            field="protocol.matched_scenario_count",
        )
        controller_evaluations = _exact_positive_int(
            protocol.get("controller_evaluation_count"),
            expected=1_920,
            field="protocol.controller_evaluation_count",
        )
        shortage_levels = _text_sequence(
            protocol.get("shortage_levels"), field="protocol.shortage_levels"
        )
        if shortage_levels != ("mild", "moderate", "severe"):
            raise PresentationDataError(
                "protocol.shortage_levels must be mild, moderate, and severe"
            )

        definitions = _mapping(payload.get("metric_definitions"), field="metric_definitions")
        expected_definition_fields = frozenset((*PRIMARY_METRICS, "power_balance_violation_rate"))
        _exact_fields(definitions, expected_definition_fields, field="metric_definitions")
        for key, (_, unit, _, direction) in _METRIC_PRESENTATION.items():
            definition = _mapping(definitions.get(key), field=f"metric_definitions.{key}")
            _validate_metric_definition(
                definition,
                field=f"metric_definitions.{key}",
                unit="ratio" if unit == "%" else unit,
                direction=direction,
                role="primary",
            )
        safety_definition = _mapping(
            definitions.get("power_balance_violation_rate"),
            field="metric_definitions.power_balance_violation_rate",
        )
        _validate_metric_definition(
            safety_definition,
            field="metric_definitions.power_balance_violation_rate",
            unit="ratio",
            direction="lower_is_better",
            role="safety",
        )

        base = _mapping(payload.get("base_evaluation"), field="base_evaluation")
        _exact_fields(base, _BASE_EVALUATION_FIELDS, field="base_evaluation")
        means = _mapping(base.get("controller_means"), field="base_evaluation.controller_means")
        if set(means) != set(CONTROLLER_NAMES):
            raise PresentationDataError(
                "base_evaluation.controller_means must contain exactly PPO and Fixed Rotation"
            )
        controller_values: dict[str, Mapping[str, Any]] = {}
        for controller_name in CONTROLLER_NAMES:
            values = _mapping(
                means.get(controller_name),
                field=f"base_evaluation.controller_means.{controller_name}",
            )
            _exact_fields(
                values,
                _CONTROLLER_MEAN_FIELDS,
                field=f"{controller_name} headline metrics",
            )
            _nonnegative_number(
                values.get("critical_outage_locality_hours"),
                field=f"{controller_name}.critical_outage_locality_hours",
            )
            for ratio_key in ("fairness", "utilization", "power_balance_violation_rate"):
                _unit_interval_number(values.get(ratio_key), field=f"{controller_name}.{ratio_key}")
            controller_values[controller_name] = values

        inference = _mapping(base.get("paired_inference"), field="base_evaluation.paired_inference")
        _exact_fields(
            inference,
            frozenset(PRIMARY_METRICS),
            field="base_evaluation.paired_inference",
        )
        metrics: list[MetricComparison] = []
        for key in _METRIC_ORDER:
            item = _mapping(inference.get(key), field=f"base_evaluation.paired_inference.{key}")
            _exact_fields(item, _INTERVAL_FIELDS, field=f"paired inference {key}")
            mean = _finite_number(item.get("mean_delta"), field=f"{key}.mean_delta")
            lower = _finite_number(item.get("ci95_lower"), field=f"{key}.ci95_lower")
            upper = _finite_number(item.get("ci95_upper"), field=f"{key}.ci95_upper")
            if lower > upper:
                raise PresentationDataError(f"{key}.ci95_lower must not exceed ci95_upper")
            classification = _text(
                item.get("result_classification"), field=f"{key}.result_classification"
            )
            if classification not in {"favorable", "ambiguous", "adverse"}:
                raise PresentationDataError(f"{key}.result_classification is invalid")
            label, unit, display_scale, direction = _METRIC_PRESENTATION[key]
            metrics.append(
                MetricComparison(
                    key=key,
                    label=label,
                    unit=unit,
                    display_scale=display_scale,
                    direction=direction,
                    role="primary",
                    ppo=_finite_number(controller_values["PPO"].get(key), field=f"PPO.{key}"),
                    fixed_rotation=_finite_number(
                        controller_values["Fixed Rotation"].get(key),
                        field=f"Fixed Rotation.{key}",
                    ),
                    mean_delta=mean,
                    ci95_lower=lower,
                    ci95_upper=upper,
                    result_classification=classification,
                )
            )

        power_balance = _mapping(base.get("power_balance"), field="base_evaluation.power_balance")
        _exact_fields(
            power_balance,
            frozenset({"maximum_violation_rate"}),
            field="base_evaluation.power_balance",
        )
        maximum_violation_rate = _unit_interval_number(
            power_balance.get("maximum_violation_rate"),
            field="base_evaluation.power_balance.maximum_violation_rate",
        )
        controller_maximum = max(
            float(values["power_balance_violation_rate"]) for values in controller_values.values()
        )
        if maximum_violation_rate != controller_maximum:
            raise PresentationDataError(
                "base_evaluation.power_balance must equal the controller maximum"
            )

        limitations = _text_sequence(payload.get("limitations"), field="limitations")
        count_statement = (
            f"{independent_draws} documented-seed-disjoint independent base draws × "
            f"{len(shortage_levels)} shortage levels = {matched_conditions} matched scenario "
            f"conditions · {controller_evaluations:,} controller evaluations"
        )
        return cls(
            artifact_id=artifact_id,
            evidence_role=evidence_role,
            artifact_sha256=ROBUSTNESS_SUPPLEMENT_SHA256,
            release_id=RELEASE_ID,
            controller_names=CONTROLLER_NAMES,
            count_statement=count_statement,
            independent_base_draw_count=independent_draws,
            shortage_level_count=len(shortage_levels),
            matched_scenario_count=matched_conditions,
            controller_evaluation_count=controller_evaluations,
            horizon_hours=HORIZON_HOURS,
            metrics=tuple(metrics),
            maximum_power_balance_violation_rate=maximum_violation_rate,
            limitations=limitations,
            training_record=CANONICAL_TRAINING_RECORD,
        )

    def metric(self, key: str) -> MetricComparison:
        """Return one named headline metric."""

        for metric in self.metrics:
            if metric.key == key:
                return metric
        raise KeyError(key)


def load_presentation_snapshot(
    path: str | Path | None = None,
    *,
    verifier: Callable[..., dict[str, Any]] = verify_robustness_supplement,
) -> PresentationSnapshot:
    """Load only the pinned, semantically verified robustness supplement."""

    evidence_path = (
        runtime_root() / "data" / "evidence" / "robustness-supplement.json"
        if path is None
        else Path(path)
    )
    try:
        payload = verifier(evidence_path)
    except RobustnessArtifactError as error:
        raise PresentationDataError(f"robustness evidence failed verification: {error}") from error
    return PresentationSnapshot.from_mapping(_mapping(payload, field="evidence"))


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PresentationDataError(f"{field} must be an object")
    return value


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], *, field: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise PresentationDataError(
            f"{field} fields are not exact; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PresentationDataError(f"{field} must be a non-empty string")
    return value.strip()


def _text_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PresentationDataError(f"{field} must be a list of strings")
    return tuple(_text(item, field=f"{field} item") for item in value)


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise PresentationDataError(f"{field} must be a positive integer")
    return value


def _integer_sequence(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PresentationDataError(f"{field} must be a list of integers")
    values = tuple(_positive_int(item, field=f"{field} item") for item in value)
    if not values:
        raise PresentationDataError(f"{field} must not be empty")
    return values


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PresentationDataError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise PresentationDataError(f"{field} must be finite")
    return normalized


def _nonnegative_number(value: object, *, field: str) -> float:
    normalized = _finite_number(value, field=field)
    if normalized < 0.0:
        raise PresentationDataError(f"{field} must be non-negative")
    return normalized


def _unit_interval_number(value: object, *, field: str) -> float:
    normalized = _finite_number(value, field=field)
    if not 0.0 <= normalized <= 1.0:
        raise PresentationDataError(f"{field} must be between 0 and 1")
    return normalized


def _exact_positive_int(value: object, *, expected: int, field: str) -> int:
    if type(value) is not int or value != expected:
        raise PresentationDataError(f"{field} must be exactly {expected:,}")
    return value


def _validate_controllers(value: object) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != len(CONTROLLER_NAMES)
        or set(value) != set(CONTROLLER_NAMES)
    ):
        raise PresentationDataError("protocol must contain exactly PPO and Fixed Rotation")


def _validate_metric_definition(
    definition: Mapping[str, object],
    *,
    field: str,
    unit: str,
    direction: str,
    role: str,
) -> None:
    expected = {"unit": unit, "direction": direction, "role": role}
    _exact_fields(definition, frozenset(expected), field=field)
    if definition != expected:
        raise PresentationDataError(f"{field} does not match the locked metric definition")
