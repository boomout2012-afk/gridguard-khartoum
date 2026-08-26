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
TRAINING_PROVENANCE_SCHEMA_VERSION = 1
TRAINING_PROVENANCE_FILENAME = "training-provenance.yaml"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_REVISION = "52f6dd82605c993c8094448caecec86faa289c0c"
_CHECKPOINT_PATH = "artifacts/checkpoints/best.pt"
_CHECKPOINT_SHA256 = "c6e5712f966b46ba18f4c69fb457b9e69f78887c0c5c0279e5ce034f40f9c665"
_ACTOR_SHA256 = "3e4a7ce44267dd1cf3bbde7e77abafb8f31fdea77ece63acda7be17fa3198ad5"
_SOURCE_HASHES = {
    "artifacts/training_manifest.json": (
        "3b866b6f470e116ff009022032059a1f91b509fadc54dbff6c5962a8f2236a59"
    ),
    "artifacts/training_metrics.jsonl": (
        "f48d2f6da29682e1c47330aa2b33b6bb4941f4e0e248f2c846436f09a4b89bc4"
    ),
    "experiments/s2-600-hidden512.json": (
        "619eb67aa253a35d7a7dc0efdbc6c743d7b0af6db4687575e7fade9dbcfe940b"
    ),
    "src/aipower_rl/training.py": (
        "886931d6c12b0ef45f4f17b89d134e27682c68a2dc4c2f90e2911e2dfec708fb"
    ),
    "src/aipower_rl/learning.py": (
        "18cf22b0b58584b2abef164ec3ea956fa2f61197336a67a7b9c1f4faefc4fac2"
    ),
    "src/aipower_rl/experiments.py": (
        "605e32626caece41508f37cbfac0cb061f7510118ba463cf3ff5843f92a9d753"
    ),
}


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
    network = _mapping(model["network"], field="model.network")
    cloning = _mapping(
        document["behavior_cloning_initialization"],
        field="behavior_cloning_initialization",
    )
    ppo = _mapping(document["ppo"], field="ppo")
    selection = _mapping(
        document["validation_and_selection"],
        field="validation_and_selection",
    )
    convergence = _mapping(document["convergence"], field="convergence")
    hidden = _integer_list(network["hidden_layers"], field="model.network.hidden_layers")
    dimensions = [
        _integer(network["observation_dim"], field="model.network.observation_dim"),
        *hidden,
        _integer(network["actor_output_logits"], field="model.network.actor_output_logits"),
    ]
    return {
        "controller": model["controller"],
        "network": " -> ".join(str(value) for value in dimensions),
        "behavior_cloning_initialization": {
            "teacher_samples": cloning["teacher_samples"],
            "epochs": cloning["epochs"],
            "optimizer_updates": cloning["optimizer_updates"],
        },
        "ppo": {
            "iterations": ppo["iterations"],
            "total_episodes": ppo["total_episodes"],
            "total_transitions": ppo["total_transitions"],
            "learning_rate": ppo["learning_rate"],
            "gamma": ppo["gamma"],
            "gae_lambda": ppo["gae_lambda"],
            "clip_epsilon": ppo["clip_epsilon"],
            "entropy_coefficient": ppo["entropy_coefficient"],
        },
        "selection": {
            "selected_seed": selection["selected_seed"],
            "selected_iteration": selection["selected_iteration"],
            "selected_checkpoint_optimizer_updates": ppo["selected_checkpoint_optimizer_updates"],
            "validation_seeds": list(selection["validation_seeds"]),
        },
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
    }


def _validate_document(payload: dict[str, Any]) -> None:
    _exact_fields(
        payload,
        {
            "schema",
            "schema_version",
            "record",
            "model",
            "software_hardware",
            "training_generator",
            "objective",
            "behavior_cloning_initialization",
            "ppo",
            "validation_and_selection",
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
    locality_count = _integer(network["locality_count"], field="model.network.locality_count")
    tier_count = _integer(
        network["service_tiers_per_locality"],
        field="model.network.service_tiers_per_locality",
    )
    if _integer(network["actor_output_logits"], field="model.network.actor_output_logits") != (
        tier_count**locality_count
    ):
        raise TrainingProvenanceError("actor_output_logits must equal service_tiers^localities")


def _validate_software_hardware(environment: dict[str, Any]) -> None:
    _exact_fields(
        environment,
        {
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
            "demand_noise",
            "training_supply",
        },
        field="training_generator",
    )
    _positive_integer(generator["horizon_hours"], field="training_generator.horizon_hours")
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


def _validate_objective(objective: dict[str, Any]) -> None:
    _exact_fields(
        objective,
        {"public_role", "formula", "episode_aggregation", "components"},
        field="objective",
    )
    for name in ("public_role", "formula", "episode_aggregation"):
        _text(objective[name], field=f"objective.{name}")
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
    for name, value in components.items():
        _positive_number(value, field=f"objective.components.{name}")


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
    _positive_number(
        cloning["max_gradient_norm"],
        field="behavior_cloning_initialization.max_gradient_norm",
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
        learning_rate_required=True,
    )


def _validate_ppo(ppo: dict[str, Any]) -> None:
    _exact_fields(
        ppo,
        {
            "training_seed",
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
            "actor_max_gradient_norm",
            "critic_max_gradient_norm",
            "optimizer",
        },
        field="ppo",
    )
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
    _boolean(ppo["advantage_normalization"], field="ppo.advantage_normalization")
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
        learning_rate_required=False,
    )


def _validate_optimizer(
    optimizer: dict[str, Any],
    *,
    field: str,
    learning_rate_required: bool,
) -> None:
    fields = {"name", "beta1", "beta2", "epsilon", "weight_decay", "amsgrad"}
    if learning_rate_required:
        fields.add("learning_rate")
    _exact_fields(optimizer, fields, field=field)
    _text(optimizer["name"], field=f"{field}.name")
    for name in ("beta1", "beta2", "epsilon"):
        _positive_number(optimizer[name], field=f"{field}.{name}")
    weight_decay = _number(optimizer["weight_decay"], field=f"{field}.weight_decay")
    if weight_decay < 0.0:
        raise TrainingProvenanceError(f"{field}.weight_decay must be non-negative")
    _boolean(optimizer["amsgrad"], field=f"{field}.amsgrad")
    if learning_rate_required:
        _positive_number(optimizer["learning_rate"], field=f"{field}.learning_rate")


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
            "candidate_best_scores",
            "selected_seed",
            "selected_iteration",
            "selected_score",
        },
        field="validation_and_selection",
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
            "untrained_score",
            "after_behavior_cloning_score",
            "ppo_iteration_100_score",
            "selected_iteration_510_score",
            "final_iteration_600_score",
            "validation_series",
        },
        field="convergence",
    )
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
    declared: dict[str, str] = {}
    for index, value in enumerate(rows):
        row = _mapping(value, field=f"source_bindings[{index}]")
        _exact_fields(row, {"path", "role", "sha256"}, field=f"source_bindings[{index}]")
        path = _relative_path(row["path"], field=f"source_bindings[{index}].path")
        _text(row["role"], field=f"source_bindings[{index}].role")
        digest = _sha256(row["sha256"], field=f"source_bindings[{index}].sha256")
        if path in declared:
            raise TrainingProvenanceError("source_bindings contains a duplicate path")
        declared[path] = digest
    if declared != _SOURCE_HASHES:
        raise TrainingProvenanceError("source_bindings do not match the accepted source hashes")


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
