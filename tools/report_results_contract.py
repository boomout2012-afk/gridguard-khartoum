"""Fail-closed projections for numeric claims rendered into the final report."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from gridguard.controllers import FixedRotationController
from gridguard.domain import HORIZON_HOURS, LOCALITIES, SECTORS
from gridguard.evaluation import PAIRED_BENCHMARK_SHA256
from gridguard.metrics import (
    CRITICAL_OUTAGE_SERVICE_THRESHOLD,
    POWER_BALANCE_TOLERANCE_MW,
)
from gridguard.robustness import ROBUSTNESS_SUPPLEMENT_SHA256

_PRIMARY_METRICS = (
    "critical_outage_locality_hours",
    "fairness",
    "utilization",
)
_REPORT_TITLE = "GridGuard Khartoum final report"
_REPORT_SECTION_ORDER = (
    "Abstract",
    "Introduction",
    "Related Work",
    "Background",
    "Discussion",
    "Conclusion",
    "Assembly",
)
_DISPLAY_BOUNDARY_SHA256 = "df1afbb337047402feb18d2dd26273c671c36761970d0c48c591c62f3ae3d496"
_DISPLAY_RIVER_SHA256 = "0486a0f7318b5d7c576c13093b27f3f7aa6410329a7c54fb66eb37d3436cdafa"
# Frozen release prose after CRLF/CR-to-LF normalization only. Markdown whitespace is structural,
# so changing wrapping, indentation, or blank lines requires the same review as changing a claim.
_REPORT_NORMALIZED_CONTENT_SHA256 = {
    "preamble": "ddfa3ec89671b3c0aeb7b51d83a3583f29a5a8d9a384f6f28e3ae717f9732972",
    "Abstract": "307751003fdaa3700170affde3d92b9fbae0fe7d47ffc48d4fd58b8b3a62e3b5",
    "Introduction": "8b3808cb930f4d733f86526c587d3dbf7a6027c2237571253f63e2a51e04fffd",
    "Related Work": "e9cfc8be71a6c400582fbbc185b24d775c92a666f6cae82b480bdf8d7b361313",
    "Background": "56458cf3b6c4484f0ea1e457928c84ce4eafb1c8051a0f83bbcb56beacdab45f",
    "Discussion": "4e629d5c2c0ef1b6092058617e03cf00fdaac9f144a0da754de9d854067ee1ca",
    "Conclusion": "efac841c084a4c9334cdeaf2d94791c789e4b7f9f961799719f359d6060a2597",
    "Assembly": "f6f05c3a9b706a0f141f7260843c6f55631595e0d072cf4723d18a39d455797c",
}
_RENDERED_REPORT_SOURCE_SHA256 = {
    "docs/REPORT.md": "6ff23113ff4f679ae588cd5e69c9e189d4b9d8079a16136b923279bdd58e6fd2",
    "docs/RELATED_WORK.md": "e45390af42287e5c2315bbef3a93e326155895224c4a151621417df41c93c75a",
    "docs/METHODOLOGY.md": "ba7c169f1d896722cc44f3f7f71f57b5a7bb67de0bcfd43070f342f23b7e5038",
    "docs/IMPLEMENTATION.md": "0057a65a14417ba7a5f55968c231baeaae03411773c436258eb6f850b78fc7a1",
    "docs/RESULTS.md": "9b8456bf4588b973a4c7e6c4347621f9f964cbfda6092e5e2be47c4c92225030",
    "docs/REFERENCES.md": "4da4f4045e26d60a008fa59de4664c33dd994636c799b0fbf7fefa5b0e4fd2fb",
}
_REPORT_SOURCE_METADATA_PREFIX = "gridguard.report-sources.v1.sha256:"
_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}
_NUMBER_WORD_TOKEN = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion)[A-Za-z-]*\b",
    flags=re.IGNORECASE,
)
_DIGIT_BEARING_TOKEN = re.compile(r"\S*\d\S*")


class ReportResultsClaimError(ValueError):
    """Raised when report Markdown is not an exact projection of verified inputs."""


@dataclass(frozen=True, slots=True)
class _ClaimFragment:
    category: str
    text: str


@dataclass(frozen=True, slots=True)
class _TableContract:
    category: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class _SectionContract:
    title: str
    fragments: tuple[_ClaimFragment, ...] = ()
    tables: tuple[_TableContract, ...] = ()


_REPORT_PREAMBLE_FRAGMENTS = (
    _ClaimFragment("REPORT metadata", "(Index 18-317)"),
    _ClaimFragment("REPORT metadata", "(Index 18-310)"),
    _ClaimFragment("REPORT metadata", "(Index 15-306)"),
    _ClaimFragment("REPORT metadata", "**Date:** August 2026"),
)


def validate_results_numeric_claims(
    markdown: str,
    *,
    accepted: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> None:
    """Require every numeric Results claim to match the verified evidence projection."""

    if not isinstance(markdown, str):
        raise TypeError("Results Markdown must be text")
    sections = _level_two_sections(markdown)
    contracts = _results_contracts(accepted=accepted, robustness=robustness)
    _validate_results_document_schema(markdown, sections, contracts)
    for contract in contracts:
        _validate_section(sections, contract)


def validate_report_numeric_claims(
    markdown: str,
    *,
    accepted: Mapping[str, Any],
    robustness: Mapping[str, Any],
    training: Mapping[str, Any],
) -> None:
    """Bind every numeric claim in the report-owned sections to canonical inputs."""

    if not isinstance(markdown, str):
        raise TypeError("REPORT Markdown must be text")
    sections = _level_two_sections(markdown)
    _validate_report_document_schema(markdown, sections)
    for contract in _report_contracts(
        accepted=accepted,
        robustness=robustness,
        training=training,
    ):
        section = sections[contract.title]
        if _markdown_tables(section):
            raise ReportResultsClaimError(
                f"REPORT {contract.title}: numeric tables are outside the report projection"
            )
        _consume_claims_and_reject_unbound_numbers(
            section,
            contract.fragments,
            category=f"REPORT {contract.title}",
        )
        _validate_report_content_digest(contract.title, section)


def validate_rendered_report_sources(sources: Mapping[str, str]) -> None:
    """Require the complete rendered Markdown bundle to match the frozen release text."""

    for path, actual in _rendered_source_digests(sources).items():
        if actual != _RENDERED_REPORT_SOURCE_SHA256[path]:
            raise ReportResultsClaimError(
                f"rendered report source digest: {path} is not the locked release text"
            )


def _rendered_source_digests(sources: Mapping[str, str]) -> dict[str, str]:
    """Hash the current source bytes, independently of the approved digest values."""

    actual_paths = frozenset(sources)
    expected_paths = frozenset(_RENDERED_REPORT_SOURCE_SHA256)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ReportResultsClaimError(
            f"rendered report source schema is not closed; missing={missing}, extra={extra}"
        )
    digests = {}
    for path in sorted(expected_paths):
        source = sources[path]
        if not isinstance(source, str):
            raise TypeError(f"rendered report source {path} must be text")
        digests[path] = _canonical_markdown_sha256(source)
    return digests


def report_source_metadata(sources: Mapping[str, str]) -> str:
    """Bind a rendered PDF to its exact normalized Markdown source bundle.

    This detects stale output after approved source pins change. It is an internal
    freshness binding, not proof of authentic authorship or visual correctness.
    """

    source_digests = json.dumps(
        _rendered_source_digests(sources), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _REPORT_SOURCE_METADATA_PREFIX + hashlib.sha256(source_digests).hexdigest()


def validate_report_source_metadata(metadata: object, sources: Mapping[str, str]) -> None:
    """Reject missing or stale rendered-source metadata even when source pins pass."""

    expected = report_source_metadata(sources)
    if not isinstance(metadata, Mapping) or metadata.get("/Keywords") != expected:
        raise ReportResultsClaimError(
            "rendered report source binding does not match current Markdown; "
            "rebuild and review the PDF"
        )


def validate_methodology_training_claims(
    markdown: str,
    *,
    training: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    """Bind the report's displayed historical PPO settings to the strict YAML record."""

    if not isinstance(markdown, str):
        raise TypeError("Methodology Markdown must be text")
    section = _subsection(
        markdown,
        "Historical PPO training record",
        expected_next_level_two="Reward",
    )
    model = _mapping(training["model"], "model")
    network = _mapping(model["network"], "model.network")
    ppo = _mapping(training["ppo"], "ppo")
    cloning = _mapping(
        training["behavior_cloning_initialization"],
        "behavior_cloning_initialization",
    )
    summary_provenance = _mapping(summary["training_provenance"], "training_provenance")
    selection = _mapping(summary["selection"], "selection")
    experiment_scope = _mapping(training["experiment_scope"], "experiment_scope")
    candidate_experiment = _mapping(
        experiment_scope["candidate_experiment"],
        "experiment_scope.candidate_experiment",
    )
    selected_run = _mapping(
        experiment_scope["selected_run"],
        "experiment_scope.selected_run",
    )
    per_run_workload = _mapping(
        candidate_experiment["per_run_workload"],
        "experiment_scope.candidate_experiment.per_run_workload",
    )
    aggregate_workload = _mapping(
        candidate_experiment["aggregate_workload"],
        "experiment_scope.candidate_experiment.aggregate_workload",
    )
    taxonomy = _mapping(training["provenance_taxonomy"], "provenance_taxonomy")
    taxonomy_categories = tuple(_mapping(taxonomy["categories"], "provenance_taxonomy.categories"))
    promotion = _mapping(selection["promotion"], "selection.promotion")
    attribution = _mapping(training["attribution_limitations"], "attribution_limitations")
    software_hardware = _mapping(training["software_hardware"], "software_hardware")
    convergence = _mapping(training["convergence"], "convergence")
    hidden_layers = tuple(
        _integer(value, "model.network.hidden_layers") for value in network["hidden_layers"]
    )
    source_bindings = tuple(
        _mapping(value, "source_bindings") for value in training["source_bindings"]
    )
    code_source_binding_count = sum(
        str(binding["path"]).startswith("src/aipower_rl/") for binding in source_bindings
    )
    record_binding_count = len(source_bindings) - code_source_binding_count
    if not hidden_layers or len(set(hidden_layers)) != 1:
        raise ReportResultsClaimError(
            "training parameters: report wording requires equal-width hidden layers"
        )
    observation_dim = _integer(network["observation_dim"], "observation_dim")
    actor_output_logits = _integer(network["actor_output_logits"], "actor_output_logits")
    teacher_samples = _integer(cloning["teacher_samples"], "teacher_samples")
    iterations = _integer(ppo["iterations"], "ppo.iterations")
    episodes_per_iteration = _integer(
        ppo["complete_episodes_per_iteration"],
        "ppo.complete_episodes_per_iteration",
    )
    episode_horizon = _integer(ppo["episode_horizon_hours"], "ppo.episode_horizon_hours")
    total_episodes = _integer(ppo["total_episodes"], "ppo.total_episodes")
    total_transitions = _integer(ppo["total_transitions"], "ppo.total_transitions")
    epochs_per_iteration = _integer(ppo["epochs_per_iteration"], "ppo.epochs_per_iteration")
    minibatch_size = _integer(ppo["minibatch_size"], "ppo.minibatch_size")
    updates_per_iteration = _integer(
        ppo["optimizer_updates_per_iteration"],
        "ppo.optimizer_updates_per_iteration",
    )
    selected_seed = _integer(selection["selected_seed"], "selection.selected_seed")
    selected_iteration = _integer(
        selection["selected_iteration"],
        "selection.selected_iteration",
    )
    selected_updates = _integer(
        selection["selected_checkpoint_optimizer_updates"],
        "selection.selected_checkpoint_optimizer_updates",
    )
    candidate_run_count = _integer(
        candidate_experiment["candidate_run_count"],
        "experiment_scope.candidate_experiment.candidate_run_count",
    )
    candidate_seeds = tuple(
        _integer(value, "experiment_scope.candidate_experiment.candidate_training_seeds")
        for value in _sequence(
            candidate_experiment["candidate_training_seeds"],
            "experiment_scope.candidate_experiment.candidate_training_seeds",
        )
    )
    selected_run_seed = _integer(
        selected_run["training_seed"],
        "experiment_scope.selected_run.training_seed",
    )
    expected_per_run = {
        "ppo_iterations": iterations,
        "complete_episodes": total_episodes,
        "transitions": total_transitions,
    }
    actual_per_run = {
        name: _integer(per_run_workload[name], f"per_run_workload.{name}")
        for name in expected_per_run
    }
    actual_aggregate = {
        name: _integer(aggregate_workload[name], f"aggregate_workload.{name}")
        for name in expected_per_run
    }
    expected_aggregate = {
        name: value * candidate_run_count for name, value in expected_per_run.items()
    }
    expected_taxonomy_categories = (
        "directly-recorded-archive-fact",
        "bound-source-or-pytorch-default-reconstruction",
        "arithmetic-derived-value",
        "unavailable-historical-environment-fact",
    )
    selected_scope = f"selected-run-seed-{selected_seed}"
    if (
        candidate_run_count != len(candidate_seeds)
        or selected_seed != selected_run_seed
        or selected_seed not in candidate_seeds
        or actual_per_run != expected_per_run
        or actual_aggregate != expected_aggregate
    ):
        raise ReportResultsClaimError(
            "training scope: selected-run or candidate-experiment workload is inconsistent"
        )
    if taxonomy_categories != expected_taxonomy_categories:
        raise ReportResultsClaimError(
            "training provenance taxonomy: category schema is not canonical"
        )
    if (
        str(software_hardware["scope"]) != selected_scope
        or str(convergence["scope"]) != selected_scope
    ):
        raise ReportResultsClaimError(
            "training scope: hardware and convergence are not bound to the selected run"
        )
    if (
        promotion.get("reward_gate_evidence_role")
        != "historical-model-selection-fact-not-independent-current-headline-evidence"
    ):
        raise ReportResultsClaimError(
            "training promotion: reward gates are not bounded to historical selection"
        )
    if (
        attribution.get("behavior_cloning_only_ablation_run") is not False
        or attribution.get("ppo_fine_tuning_incremental_effect_isolated") is not False
    ):
        raise ReportResultsClaimError(
            "training attribution: behavior-cloning attribution boundary is inconsistent"
        )

    fragments = (
        _ClaimFragment(
            "training parameters",
            "The strict, model-manifest-bound record is "
            f"`{summary_provenance['path']}`, SHA-256 `{summary_provenance['sha256']}`. "
            f"It records the {observation_dim}-value "
            f"input, {_number_word(len(hidden_layers))} {hidden_layers[0]}-unit tanh layers, "
            "separate actor/critic encoders, "
            f"{_count(actor_output_logits)} action "
            "logits, optimizer settings, workload, seeds, validation, selection, selected-run "
            f"hardware and convergence, {_number_word(code_source_binding_count)} "
            "archive code-source bindings, and "
            f"{_number_word(record_binding_count)} run/metric/experiment record bindings.",
        ),
        _ClaimFragment(
            "training provenance taxonomy",
            "Its field map distinguishes directly recorded facts, source-reconstructed values "
            "(including named PyTorch defaults), arithmetic-derived values, and unavailable "
            "historical environment facts.",
        ),
        _ClaimFragment(
            "training parameters",
            "Behavior-cloning initialization supplied the starting "
            f"PPO actor from {_count(teacher_samples)} "
            "teacher samples",
        ),
        _ClaimFragment(
            "training attribution",
            "No behavior-cloning-only ablation was run, so this comparison cannot isolate the "
            "incremental effect of PPO fine-tuning; the initialization is checkpoint lineage, "
            "not an independently evaluated comparator.",
        ),
        _ClaimFragment(
            "training scope",
            f"For the selected seed-{selected_seed} run, PPO ran {_count(iterations)} iterations "
            f"with {_count(episodes_per_iteration)} "
            f"complete {episode_horizon}-hour episodes per iteration: {_count(total_episodes)} "
            f"episodes and {_count(total_transitions)} transitions.",
        ),
        _ClaimFragment(
            "training parameters",
            f"Each iteration used {_number_word(epochs_per_iteration)} epochs, "
            f"{_count(minibatch_size)}-sample minibatches, {_count(updates_per_iteration)} "
            f"optimizer updates, `gamma={_plain_number(ppo['gamma'])}`, "
            f"`GAE lambda={_plain_number(ppo['gae_lambda'])}`, learning rate "
            f"`{_plain_number(ppo['learning_rate'])}`, clip epsilon "
            f"`{_plain_number(ppo['clip_epsilon'])}`, value coefficient "
            f"`{_plain_number(ppo['value_coefficient'])}`, normalized entropy coefficient "
            f"`{_plain_number(ppo['entropy_coefficient'])}`, and separate actor/critic maximum "
            f"gradient norms of `{_plain_number(ppo['actor_max_gradient_norm'])}`.",
        ),
        _ClaimFragment(
            "training scope",
            f"The {_number_word(candidate_run_count)}-seed candidate experiment totaled "
            f"{_count(actual_aggregate['ppo_iterations'])} iterations, "
            f"{_count(actual_aggregate['complete_episodes'])} episodes, and "
            f"{_count(actual_aggregate['transitions'])} transitions.",
        ),
        _ClaimFragment(
            "training promotion",
            "Its historical promotion gates required all runs to be safe and finite, reward and "
            "seed-win margins, bounded worst-seed regression, and acceptable throughput. Those "
            "gates are historical selection facts, not independent headline evidence.",
        ),
        _ClaimFragment(
            "training parameters",
            "Adam optimized historical PPO parameters; the selected checkpoint was iteration "
            f"{_count(selected_iteration)} "
            f"after {_count(selected_updates)} "
            "updates.",
        ),
        _ClaimFragment(
            "training scope",
            "The recorded runtime, hardware, and convergence series apply only to selected seed "
            f"{selected_seed}.",
        ),
    )
    _consume_claims_and_reject_unbound_numbers(
        section,
        fragments,
        category="training parameters",
    )


def _report_contracts(
    *,
    accepted: Mapping[str, Any],
    robustness: Mapping[str, Any],
    training: Mapping[str, Any],
) -> tuple[_SectionContract, ...]:
    accepted_protocol = _mapping(accepted["protocol"], "accepted.protocol")
    accepted_aggregate = _mapping(accepted["aggregate"], "accepted.aggregate")
    robust_protocol = _mapping(robustness["protocol"], "robustness.protocol")
    sensitivity = _mapping(robustness["sensitivity"], "robustness.sensitivity")
    sensitivity_base = _mapping(
        sensitivity["base_parameters"],
        "robustness.sensitivity.base_parameters",
    )
    variants = tuple(
        _mapping(value, "robustness.sensitivity.variants")
        for value in _sequence(
            sensitivity["variants"],
            "robustness.sensitivity.variants",
        )
    )
    forecast = _mapping(
        robustness["forecast_evaluation"],
        "robustness.forecast_evaluation",
    )
    forecast_methods = _mapping(forecast["methods"], "forecast.methods")
    persistence = _mapping(
        forecast_methods["persistence_1h"],
        "forecast.methods.persistence_1h",
    )
    trailing = _mapping(
        forecast_methods["trailing_mean_3h"],
        "forecast.methods.trailing_mean_3h",
    )
    model = _mapping(training["model"], "training.model")
    network = _mapping(model["network"], "training.model.network")
    ppo = _mapping(training["ppo"], "training.ppo")
    experiment_scope = _mapping(training["experiment_scope"], "training.experiment_scope")
    candidate_experiment = _mapping(
        experiment_scope["candidate_experiment"],
        "training.experiment_scope.candidate_experiment",
    )
    selected_run = _mapping(
        experiment_scope["selected_run"],
        "training.experiment_scope.selected_run",
    )
    per_run_workload = _mapping(
        candidate_experiment["per_run_workload"],
        "training.experiment_scope.candidate_experiment.per_run_workload",
    )
    aggregate_workload = _mapping(
        candidate_experiment["aggregate_workload"],
        "training.experiment_scope.candidate_experiment.aggregate_workload",
    )
    taxonomy = _mapping(training["provenance_taxonomy"], "training.provenance_taxonomy")
    taxonomy_categories = tuple(
        _mapping(taxonomy["categories"], "training.provenance_taxonomy.categories")
    )
    selection = _mapping(
        training["validation_and_selection"],
        "training.validation_and_selection",
    )
    promotion = _mapping(selection["promotion"], "training.validation_and_selection.promotion")
    attribution = _mapping(
        training["attribution_limitations"],
        "training.attribution_limitations",
    )

    robust_seeds = tuple(
        _integer(value, "robustness.protocol.seeds")
        for value in _sequence(robust_protocol["seeds"], "robustness.protocol.seeds")
    )
    if not robust_seeds:
        raise ReportResultsClaimError("report holdout: verified seed sequence is empty")
    if robust_seeds != tuple(range(robust_seeds[0], robust_seeds[-1] + 1)):
        raise ReportResultsClaimError(
            "report holdout: report range requires consecutive verified seeds"
        )

    accepted_controllers = tuple(
        str(value)
        for value in _sequence(
            accepted_protocol["controllers"],
            "accepted.protocol.controllers",
        )
    )
    robust_controllers = tuple(
        str(value)
        for value in _sequence(
            robust_protocol["controllers"],
            "robustness.protocol.controllers",
        )
    )
    if accepted_controllers != robust_controllers or set(accepted_controllers) != {
        "Fixed Rotation",
        "PPO",
    }:
        raise ReportResultsClaimError(
            "report controller count: verified inputs do not describe the canonical pair"
        )

    locality_weights = _sequence(
        sensitivity_base["locality_weights"],
        "robustness.sensitivity.base_parameters.locality_weights",
    )
    sector_shares = _sequence(
        sensitivity_base["sector_shares"],
        "robustness.sensitivity.base_parameters.sector_shares",
    )
    locality_count = len(locality_weights)
    sector_count = len(sector_shares)
    network_locality_count = _integer(
        network["locality_count"],
        "training.model.network.locality_count",
    )
    service_tier_count = _integer(
        network["service_tiers_per_locality"],
        "training.model.network.service_tiers_per_locality",
    )
    action_count = _integer(
        network["actor_output_logits"],
        "training.model.network.actor_output_logits",
    )
    training_horizon = _integer(
        ppo["episode_horizon_hours"],
        "training.ppo.episode_horizon_hours",
    )
    accepted_horizon = _integer(
        accepted_protocol["horizon_hours"],
        "accepted.protocol.horizon_hours",
    )
    if locality_count != len(LOCALITIES) or network_locality_count != len(LOCALITIES):
        raise ReportResultsClaimError(
            "report model dimensions: locality counts disagree with the canonical domain"
        )
    if sector_count != len(SECTORS):
        raise ReportResultsClaimError(
            "report model dimensions: sector count disagrees with the canonical domain"
        )
    if training_horizon != HORIZON_HOURS or accepted_horizon != HORIZON_HOURS:
        raise ReportResultsClaimError(
            "report model dimensions: horizon disagrees with the canonical domain"
        )
    if service_tier_count**network_locality_count != action_count:
        raise ReportResultsClaimError(
            "report model dimensions: actor logits do not enumerate the stated joint actions"
        )

    sensitivity_component_count = locality_count + sector_count
    sensitivity_variant_count = len(variants)
    sensitivity_parameterization_count = _integer(
        robust_protocol["sensitivity_parameterization_count"],
        "robustness.protocol.sensitivity_parameterization_count",
    )
    if sensitivity_variant_count != sensitivity_component_count * 2:
        raise ReportResultsClaimError(
            "report sensitivity: verified variants do not perturb each component twice"
        )
    if sensitivity_parameterization_count != sensitivity_variant_count + 1:
        raise ReportResultsClaimError(
            "report sensitivity: parameterization count does not include exactly one base"
        )

    aggregate_unused: dict[str, float] = {}
    for controller in accepted_controllers:
        metrics = _mapping(
            accepted_aggregate[controller],
            f"accepted.aggregate.{controller}",
        )
        aggregate_unused[controller] = _number(metrics["available_mwh"]) - _number(
            metrics["served_mwh"]
        )

    robust_base_draws = _integer(
        robust_protocol["independent_base_draw_count"],
        "robustness.protocol.independent_base_draw_count",
    )
    robust_scenarios = _integer(
        robust_protocol["matched_scenario_count"],
        "robustness.protocol.matched_scenario_count",
    )
    robust_evaluations = _integer(
        robust_protocol["controller_evaluation_count"],
        "robustness.protocol.controller_evaluation_count",
    )
    accepted_base_draws = _integer(
        accepted_protocol["independent_base_draw_count"],
        "accepted.protocol.independent_base_draw_count",
    )
    ci_percent = _integer(
        _mapping(robust_protocol["bootstrap"], "robustness.protocol.bootstrap")["ci_percent"],
        "robustness.protocol.bootstrap.ci_percent",
    )
    persistence_window = _integer(
        persistence["window_hours"],
        "forecast.methods.persistence_1h.window_hours",
    )
    trailing_window = _integer(
        trailing["window_hours"],
        "forecast.methods.trailing_mean_3h.window_hours",
    )
    candidate_run_count = _integer(
        candidate_experiment["candidate_run_count"],
        "training.experiment_scope.candidate_experiment.candidate_run_count",
    )
    candidate_seeds = tuple(
        _integer(value, "training.experiment_scope.candidate_training_seeds")
        for value in _sequence(
            candidate_experiment["candidate_training_seeds"],
            "training.experiment_scope.candidate_training_seeds",
        )
    )
    selected_seed = _integer(
        selected_run["training_seed"],
        "training.experiment_scope.selected_run.training_seed",
    )
    per_run = {
        name: _integer(per_run_workload[name], f"training.per_run_workload.{name}")
        for name in ("ppo_iterations", "complete_episodes", "transitions")
    }
    aggregate_workload_values = {
        name: _integer(aggregate_workload[name], f"training.aggregate_workload.{name}")
        for name in per_run
    }
    ppo_per_run = {
        "ppo_iterations": _integer(ppo["iterations"], "training.ppo.iterations"),
        "complete_episodes": _integer(ppo["total_episodes"], "training.ppo.total_episodes"),
        "transitions": _integer(ppo["total_transitions"], "training.ppo.total_transitions"),
    }
    expected_aggregate = {name: value * candidate_run_count for name, value in per_run.items()}
    expected_taxonomy_categories = (
        "directly-recorded-archive-fact",
        "bound-source-or-pytorch-default-reconstruction",
        "arithmetic-derived-value",
        "unavailable-historical-environment-fact",
    )
    if (
        candidate_run_count != len(candidate_seeds)
        or selected_seed not in candidate_seeds
        or selected_seed
        != _integer(selection["selected_seed"], "training.validation_and_selection.selected_seed")
        or per_run != ppo_per_run
        or aggregate_workload_values != expected_aggregate
    ):
        raise ReportResultsClaimError(
            "report training scope: selected-run or candidate-experiment workload is inconsistent"
        )
    if taxonomy_categories != expected_taxonomy_categories:
        raise ReportResultsClaimError(
            "report training provenance taxonomy: category schema is not canonical"
        )
    if (
        promotion.get("reward_gate_evidence_role")
        != "historical-model-selection-fact-not-independent-current-headline-evidence"
    ):
        raise ReportResultsClaimError(
            "report training promotion: historical gates have an invalid evidence role"
        )
    if (
        attribution.get("behavior_cloning_only_ablation_run") is not False
        or attribution.get("ppo_fine_tuning_incremental_effect_isolated") is not False
    ):
        raise ReportResultsClaimError(
            "report training attribution: behavior-cloning attribution boundary is inconsistent"
        )

    return (
        _SectionContract(
            "Abstract",
            fragments=(
                _ClaimFragment(
                    "report localities",
                    f"across {_number_word(locality_count)} Khartoum State localities.",
                ),
                _ClaimFragment(
                    "report horizon",
                    f"identical synthetic {HORIZON_HOURS}-hour shortage scenarios.",
                ),
                _ClaimFragment(
                    "report holdout",
                    f"seeds {robust_seeds[0]}-{robust_seeds[-1]}: "
                    f"{_count(robust_base_draws)} independent base draws, "
                    f"{_count(robust_scenarios)} matched scenario conditions, and "
                    f"{_count(robust_evaluations)} controller evaluations.",
                ),
                _ClaimFragment(
                    "report primary outcomes",
                    f"are the {_number_word(len(_PRIMARY_METRICS))} primary outcomes",
                ),
                _ClaimFragment(
                    "report training scope",
                    f"The selected seed-{selected_seed} run used "
                    f"{_count(per_run['ppo_iterations'])} PPO iterations, "
                    f"{_count(per_run['complete_episodes'])} episodes, and "
                    f"{_count(per_run['transitions'])} transitions",
                ),
                _ClaimFragment(
                    "report training scope",
                    f"the {_number_word(candidate_run_count)}-seed candidate experiment totaled "
                    f"{_count(aggregate_workload_values['ppo_iterations'])} iterations, "
                    f"{_count(aggregate_workload_values['complete_episodes'])} episodes, and "
                    f"{_count(aggregate_workload_values['transitions'])} transitions.",
                ),
                _ClaimFragment(
                    "report training provenance taxonomy",
                    "source hashes, and evidence category of each recovered field.",
                ),
                _ClaimFragment(
                    "report training promotion",
                    "Hardware and convergence are explicitly selected-run facts, and historical "
                    "promotion gates are not independent headline evidence.",
                ),
                _ClaimFragment(
                    "report sensitivity",
                    f"covers {_count(sensitivity_parameterization_count)} parameterizations",
                ),
                _ClaimFragment(
                    "report forecast",
                    f"with {_number_word(persistence_window)}-hour persistence",
                ),
            ),
        ),
        _SectionContract(
            "Introduction",
            fragments=(
                _ClaimFragment(
                    "report controller count",
                    f"{_number_word(1)} frozen PPO policy is evaluated against "
                    f"{_number_word(1)} transparent but supply-blind Fixed Rotation schedule",
                ),
                _ClaimFragment(
                    "report deliverables",
                    f"The project has {_number_word(len(accepted_controllers))} deliverables",
                ),
            ),
        ),
        _SectionContract(
            "Related Work",
            fragments=tuple(
                _ClaimFragment("report citations", citation)
                for citation in ("[9]", "[10, 11]", "[12]", "[13]", "[14]", "[15]")
            ),
        ),
        _SectionContract(
            "Background",
            fragments=(
                _ClaimFragment(
                    "report model dimensions",
                    f"to {_number_word(1)} joint action across "
                    f"{_number_word(locality_count)} localities.",
                ),
                _ClaimFragment(
                    "report model dimensions",
                    f"{HORIZON_HOURS} {_number_word(1)}-hour steps, "
                    f"{_number_word(sector_count)} demand sectors per locality, and "
                    f"{_number_word(service_tier_count)} service tiers.",
                ),
                _ClaimFragment("report map sources", "revision v03 [7]"),
                _ClaimFragment("report map sources", "Natural Earth 1:10m data [8]."),
            ),
        ),
        _SectionContract(
            "Discussion",
            fragments=(
                _ClaimFragment(
                    "report holdout",
                    f"because its {_number_word(len(robust_seeds))} seeds are disjoint",
                ),
                _ClaimFragment(
                    "accepted reproduction",
                    f"The accepted {_count(accepted_base_draws)}-draw artifact",
                ),
                _ClaimFragment(
                    "report primary outcomes",
                    f"The {_number_word(len(_PRIMARY_METRICS))} primary outcomes",
                ),
                _ClaimFragment(
                    "accepted unused supply",
                    "Mean unused supply in the accepted reproduction is "
                    f"{aggregate_unused['Fixed Rotation']:,.1f} MWh for Fixed Rotation versus "
                    f"{aggregate_unused['PPO']:,.1f} MWh for PPO",
                ),
                _ClaimFragment(
                    "report sensitivity",
                    f"perturbs {_count(sensitivity_component_count)} allocation components "
                    f"in both directions, giving {_count(sensitivity_variant_count)} "
                    "one-factor-at-a-time variants and "
                    f"{_count(sensitivity_parameterization_count)} parameterizations including "
                    "the base.",
                ),
                _ClaimFragment(
                    "report sensitivity",
                    f"pointwise {_count(ci_percent)}%",
                ),
                _ClaimFragment(
                    "report forecast",
                    f"the {_number_word(trailing_window)}-hour trailing mean with "
                    f"{_number_word(persistence_window)}-hour persistence",
                ),
                _ClaimFragment(
                    "report model dimensions",
                    f"enumerates `{service_tier_count}^N` joint actions "
                    f"(`{service_tier_count}^{network_locality_count} = "
                    f"{_count(action_count)}` here), which limits scaling beyond "
                    f"{_number_word(network_locality_count)} localities.",
                ),
                _ClaimFragment(
                    "report controller count",
                    f"the {_number_word(len(accepted_controllers))} controllers on "
                    f"{_number_word(1)} selected run",
                ),
            ),
        ),
        _SectionContract(
            "Conclusion",
            fragments=(
                _ClaimFragment(
                    "report controller count",
                    f"for {_number_word(1)} frozen actor and {_number_word(1)} weak baseline",
                ),
                _ClaimFragment(
                    "report training provenance taxonomy",
                    "The strict training YAML makes each recovered parameter, provenance "
                    "category, scope, historical unknown, and selection caveat discoverable.",
                ),
                _ClaimFragment(
                    "report training attribution",
                    "No behavior-cloning-only ablation was run, so this comparison cannot isolate "
                    "the incremental effect of PPO fine-tuning.",
                ),
            ),
        ),
        _SectionContract("Assembly"),
    )


def _results_contracts(
    *,
    accepted: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> tuple[_SectionContract, ...]:
    accepted_protocol = _mapping(accepted["protocol"], "accepted.protocol")
    robust_protocol = _mapping(robustness["protocol"], "robustness.protocol")
    base = _mapping(robustness["base_evaluation"], "robustness.base_evaluation")
    sensitivity = _mapping(robustness["sensitivity"], "robustness.sensitivity")
    forecast = _mapping(robustness["forecast_evaluation"], "robustness.forecast_evaluation")
    base_parameters = _mapping(sensitivity["base_parameters"], "sensitivity.base_parameters")
    locality_count = len(_sequence(base_parameters["locality_weights"], "locality_weights"))
    sector_count = len(_sequence(base_parameters["sector_shares"], "sector_shares"))

    return (
        _result_statement_contract(base, robust_protocol),
        _provenance_contract(accepted, robustness, accepted_protocol, robust_protocol, sensitivity),
        _primary_contract(base, accepted_protocol, sensitivity, locality_count),
        _accepted_contract(accepted, accepted_protocol),
        _sensitivity_contract(base, sensitivity, robust_protocol),
        _forecast_contract(forecast, locality_count, sector_count),
        _baseline_contract(accepted, accepted_protocol),
        _interpretation_contract(forecast),
        _limitations_contract(sensitivity, locality_count, robust_protocol),
    )


def _result_statement_contract(
    base: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> _SectionContract:
    seeds = tuple(_integer(value, "robustness.protocol.seeds") for value in protocol["seeds"])
    levels = [str(value).title() for value in protocol["shortage_levels"]]
    inference = _mapping(base["paired_inference"], "base_evaluation.paired_inference")
    ci_percent = _integer(_mapping(protocol["bootstrap"], "bootstrap")["ci_percent"], "ci_percent")
    base_draw_count = _count(protocol["independent_base_draw_count"])
    scenario_count = _count(protocol["matched_scenario_count"])
    evaluation_count = _count(protocol["controller_evaluation_count"])
    maximum_violation = _number(
        _mapping(base["power_balance"], "base_evaluation.power_balance")["maximum_violation_rate"]
    )
    if maximum_violation != 0.0:
        raise ReportResultsClaimError(
            "result statement: zero-violation wording is not supported by evidence"
        )
    controllers = tuple(str(value) for value in protocol["controllers"])
    return _SectionContract(
        "Result statement",
        fragments=(
            _ClaimFragment("result statement", f"holdout on seeds {seeds[0]}-{seeds[-1]}"),
            _ClaimFragment(
                "result statement",
                f"contains {base_draw_count} independent base "
                f"draws crossed with {_human_join(levels)} shortage-only conditions: "
                f"{scenario_count} matched scenario conditions and {evaluation_count} "
                "controller-scenario evaluations.",
            ),
            _ClaimFragment(
                "result statement",
                f"All {_number_word(len(inference))} deterministic {ci_percent}% "
                "seed-stratified paired cluster intervals",
            ),
            _ClaimFragment(
                "result statement",
                f"Both controllers record {_number_word(int(maximum_violation))} "
                "power-balance violations.",
            ),
            _ClaimFragment(
                "result statement",
                f"The finding applies to {_number_word(controllers.count('PPO'))} frozen PPO "
                f"actor, {_number_word(controllers.count('Fixed Rotation'))} supply-blind Fixed "
                "Rotation baseline",
            ),
        ),
    )


def _provenance_contract(
    accepted: Mapping[str, Any],
    robustness: Mapping[str, Any],
    accepted_protocol: Mapping[str, Any],
    robust_protocol: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> _SectionContract:
    accepted_provenance = _mapping(accepted["provenance"], "accepted.provenance")
    frozen = _mapping(accepted_provenance["frozen_benchmark"], "frozen_benchmark")
    robust_provenance = _mapping(robustness["provenance"], "robustness.provenance")
    training = _mapping(robust_provenance["training_provenance"], "training_provenance")
    accepted_seeds = tuple(
        _integer(value, "accepted.protocol.seeds") for value in accepted_protocol["seeds"]
    )
    comparison_granularity = str(
        _mapping(accepted["frozen_parity"], "accepted.frozen_parity")["comparison_granularity"]
    )
    comparison_token = comparison_granularity.split(maxsplit=1)[0]
    if not comparison_token.isdecimal():
        raise ReportResultsClaimError(
            "evidence provenance: frozen comparison granularity has no count"
        )
    comparison_count = int(comparison_token)
    variant_count = len(_sequence(sensitivity["variants"], "sensitivity.variants"))
    accepted_draws = _count(accepted_protocol["independent_base_draw_count"])
    accepted_scenarios = _count(accepted_protocol["matched_scenario_count"])
    accepted_evaluations = _count(accepted_protocol["controller_evaluation_count"])
    robust_draws = _count(robust_protocol["independent_base_draw_count"])
    table = _TableContract(
        "evidence provenance",
        ("Evidence role", "File and SHA-256", "Permitted interpretation"),
        (
            (
                "Accepted aggregate source",
                f"`{frozen['path']}` - `{frozen['sha256']}`",
                f"Preserve the accepted {_count(comparison_count)} batch-cell summaries; it "
                "contains no historical "
                "episode rows or intervals",
            ),
            (
                "Accepted reproduction evidence",
                f"`data/evidence/paired-benchmark.json` - `{PAIRED_BENCHMARK_SHA256}`",
                f"Reproduce {accepted_draws} independent base draws, {accepted_scenarios} "
                f"matched scenario conditions, {accepted_evaluations} controller "
                f"evaluations, paired rows, and frozen parity; seeds `{accepted_seeds[0]}-"
                f"{accepted_seeds[-1]}` participated in model selection, so this is not an "
                "independently custodied final test",
            ),
            (
                "Documented-seed-disjoint holdout",
                f"`data/evidence/robustness-supplement.json` - `{ROBUSTNESS_SUPPLEMENT_SHA256}`",
                f"Report the {robust_draws}-draw primary "
                f"comparison, {_count(variant_count)} sensitivity variants, safety invariant, and "
                "synthetic forecast evaluation; disjointness covers documented activity only and "
                "does not imply preregistration or independent custody",
            ),
            (
                "Historical PPO training record",
                f"`{training['path']}` - `{training['sha256']}`",
                "Distinguish the selected seed-43 run from the three-seed candidate experiment; "
                "scope hardware/convergence to the selected run; expose promotion checks, "
                "provenance categories, unavailable historical facts, and source lineage; this "
                "is not a bitwise-retraining bundle and historical gates are not independent "
                "headline evidence",
            ),
        ),
    )
    return _SectionContract(
        "Results provenance and evidence roles",
        fragments=(
            _ClaimFragment(
                "reporting precision",
                "Human-facing prose, tables, UI, and slides use "
                f"{_number_word(1)} decimal place for MWh and locality-hours, "
                f"{_number_word(2)} decimals for percentages, and "
                f"{_number_word(3)} decimals for the Training-objective score (diagnostic).",
            ),
            _ClaimFragment(
                "display provenance",
                "OCHA/HDX administrative boundaries are pinned at SHA-256 "
                f"`{_DISPLAY_BOUNDARY_SHA256}`; Natural Earth Nile linework is pinned at "
                f"`{_DISPLAY_RIVER_SHA256}`.",
            ),
        ),
        tables=(table,),
    )


def _primary_contract(
    base: Mapping[str, Any],
    accepted_protocol: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    locality_count: int,
) -> _SectionContract:
    means = _mapping(base["controller_means"], "base_evaluation.controller_means")
    fixed = _mapping(means["Fixed Rotation"], "controller_means.Fixed Rotation")
    ppo = _mapping(means["PPO"], "controller_means.PPO")
    inference = _mapping(base["paired_inference"], "base_evaluation.paired_inference")
    outage = _mapping(inference["critical_outage_locality_hours"], "paired_inference.outage")
    fairness = _mapping(inference["fairness"], "paired_inference.fairness")
    utilization = _mapping(inference["utilization"], "paired_inference.utilization")
    table = _TableContract(
        "primary outcomes",
        (
            "Primary outcome",
            "Fixed Rotation",
            "PPO",
            "Mean paired difference",
            "Deterministic 95% interval",
            "Direction",
        ),
        (
            (
                "Critical-outage locality-hours",
                f"{_number(fixed['critical_outage_locality_hours']):.1f}",
                f"{_number(ppo['critical_outage_locality_hours']):.1f}",
                f"{_number(outage['mean_delta']):+.1f}",
                f"[{_number(outage['ci95_lower']):.1f}, {_number(outage['ci95_upper']):.1f}]",
                str(outage["result_classification"]).title(),
            ),
            (
                "Jain fairness",
                f"{_number(fixed['fairness']):.2%}",
                f"{_number(ppo['fairness']):.2%}",
                f"{_number(fairness['mean_delta']) * 100:+.2f} percentage points",
                f"[{_number(fairness['ci95_lower']) * 100:+.2f}, "
                f"{_number(fairness['ci95_upper']) * 100:+.2f}]",
                str(fairness["result_classification"]).title(),
            ),
            (
                "Utilization",
                f"{_number(fixed['utilization']):.2%}",
                f"{_number(ppo['utilization']):.2%}",
                f"{_number(utilization['mean_delta']) * 100:+.2f} percentage points",
                f"[{_number(utilization['ci95_lower']) * 100:+.2f}, "
                f"{_number(utilization['ci95_upper']) * 100:+.2f}]",
                str(utilization["result_classification"]).title(),
            ),
        ),
    )
    variants = _sequence(sensitivity["variants"], "sensitivity.variants")
    maximum_outage = locality_count * _integer(accepted_protocol["horizon_hours"], "horizon_hours")
    maximum_violation = _number(
        _mapping(base["power_balance"], "base_evaluation.power_balance")["maximum_violation_rate"]
    )
    return _SectionContract(
        "Documented-seed-disjoint primary outcomes",
        fragments=(
            _ClaimFragment(
                "primary outcomes",
                f"Critical-outage locality-hours can range from 0 to {maximum_outage} per episode.",
            ),
            _ClaimFragment(
                "primary outcomes",
                f"below the explicit {CRITICAL_OUTAGE_SERVICE_THRESHOLD:.0%} threshold.",
            ),
            _ClaimFragment(
                "power-balance safety",
                f"Maximum power-balance violation rate is {maximum_violation:.2%} across the base "
                f"comparison and all {_count(len(variants))} sensitivity variants for both "
                f"controllers under the {POWER_BALANCE_TOLERANCE_MW:.3f} MW simulator tolerance.",
            ),
        ),
        tables=(table,),
    )


def _accepted_contract(
    accepted: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> _SectionContract:
    aggregate = _mapping(accepted["aggregate"], "accepted.aggregate")
    fixed = _mapping(aggregate["Fixed Rotation"], "accepted.aggregate.Fixed Rotation")
    ppo = _mapping(aggregate["PPO"], "accepted.aggregate.PPO")
    inference = _mapping(accepted["paired_inference"], "accepted.paired_inference")
    outage = _mapping(inference["critical_outage_locality_hours"], "accepted inference outage")
    objective = _mapping(inference["episode_reward"], "accepted inference objective")
    table = _TableContract(
        "accepted reproduction",
        ("Outcome", "Fixed Rotation", "PPO", "Public role"),
        (
            (
                "Training-objective score (diagnostic)",
                f"{_number(fixed['episode_reward']):.3f}",
                f"{_number(ppo['episode_reward']):.3f}",
                "Diagnostic; shares components with reported outcomes",
            ),
            (
                "Critical-service ratio",
                f"{_number(fixed['critical_service_ratio']):.2%}",
                f"{_number(ppo['critical_service_ratio']):.2%}",
                "Supporting",
            ),
            (
                "Critical-outage locality-hours",
                f"{_number(fixed['critical_outage_locality_hours']):.1f}",
                f"{_number(ppo['critical_outage_locality_hours']):.1f}",
                "Primary",
            ),
            (
                "Jain fairness",
                f"{_number(fixed['fairness']):.2%}",
                f"{_number(ppo['fairness']):.2%}",
                "Primary",
            ),
            (
                "Utilization",
                f"{_number(fixed['utilization']):.2%}",
                f"{_number(ppo['utilization']):.2%}",
                "Primary",
            ),
            (
                "Served energy",
                f"{_number(fixed['served_mwh']):,.1f} MWh",
                f"{_number(ppo['served_mwh']):,.1f} MWh",
                "Supporting",
            ),
            (
                "Curtailed energy",
                f"{_number(fixed['curtailed_mwh']):,.1f} MWh",
                f"{_number(ppo['curtailed_mwh']):,.1f} MWh",
                "Complementary energy identity",
            ),
            (
                "Switching rate",
                f"{_number(fixed['switching_rate']):.2%}",
                f"{_number(ppo['switching_rate']):.2%}",
                "Secondary",
            ),
            (
                "Power-balance violation rate",
                f"{_number(fixed['power_balance_violation_rate']):.2%}",
                f"{_number(ppo['power_balance_violation_rate']):.2%}",
                "Safety",
            ),
        ),
    )
    ci_percent = (
        _number(_mapping(protocol["bootstrap"], "accepted.protocol.bootstrap")["confidence_level"])
        * 100
    )
    base_draw_count = _count(protocol["independent_base_draw_count"])
    scenario_count = _count(protocol["matched_scenario_count"])
    evaluation_count = _count(protocol["controller_evaluation_count"])
    outage_delta = _number(outage["mean_delta"])
    outage_lower = _number(outage["ci95_lower"])
    outage_upper = _number(outage["ci95_upper"])
    objective_delta = _number(objective["mean_delta"])
    return _SectionContract(
        "Accepted reproduction context",
        fragments=(
            _ClaimFragment(
                "accepted reproduction",
                f"It has {base_draw_count} independent base draws, {scenario_count} matched "
                f"scenario conditions, and {evaluation_count} controller-scenario evaluations.",
            ),
            _ClaimFragment(
                "accepted reproduction",
                f"The accepted critical-outage paired difference is {outage_delta:+.1f} "
                f"locality-hours with a deterministic {ci_percent:.0f}% interval of "
                f"[{outage_lower:.1f}, {outage_upper:.1f}]. The Training-objective score "
                f"difference is {objective_delta:+.3f}",
            ),
            _ClaimFragment(
                "accepted reproduction",
                f"not {_number_word(2)} independent endpoints.",
            ),
            _ClaimFragment(
                "accepted reproduction",
                f"does not make the {_number_word(len(protocol['seeds']))} reused seeds "
                "independently custodied.",
            ),
        ),
        tables=(table,),
    )


def _sensitivity_contract(
    base: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> _SectionContract:
    variants = tuple(_mapping(value, "sensitivity variant") for value in sensitivity["variants"])
    factors = sorted({_number(variant["factor"]) for variant in variants})
    if len(factors) != 2:
        raise ReportResultsClaimError("sensitivity: expected exactly two perturbation factors")
    minus_percent = (factors[0] - 1.0) * 100
    plus_percent = (factors[1] - 1.0) * 100
    inference = _mapping(base["paired_inference"], "base_evaluation.paired_inference")
    interval_count = len(variants) * len(inference) * 2
    did_counts: dict[str, Counter[str]] = {}
    paired_counts: dict[str, Counter[str]] = {}
    for metric in _PRIMARY_METRICS:
        did_counts[metric] = Counter(
            str(
                _mapping(
                    _mapping(variant["difference_in_differences"], "difference_in_differences")[
                        metric
                    ],
                    metric,
                )["result_classification"]
            )
            for variant in variants
        )
        paired_counts[metric] = Counter(
            str(
                _mapping(_mapping(variant["paired_inference"], "paired_inference")[metric], metric)[
                    "result_classification"
                ]
            )
            for variant in variants
        )
    if any(counts != Counter({"favorable": len(variants)}) for counts in paired_counts.values()):
        raise ReportResultsClaimError(
            "sensitivity: report wording requires every variant interval to be favorable"
        )
    table = _TableContract(
        "sensitivity",
        ("Difference-in-differences outcome", "Favorable", "Ambiguous", "Adverse"),
        tuple(
            (
                label,
                str(did_counts[metric]["favorable"]),
                str(did_counts[metric]["ambiguous"]),
                str(did_counts[metric]["adverse"]),
            )
            for metric, label in (
                ("critical_outage_locality_hours", "Critical-outage locality-hours"),
                ("fairness", "Jain fairness"),
                ("utilization", "Utilization"),
            )
        ),
    )
    ci_percent = _integer(
        _mapping(protocol["bootstrap"], "robustness.protocol.bootstrap")["ci_percent"], "ci_percent"
    )
    variant_count = _count(len(variants))
    parameterization_count = _count(protocol["sensitivity_parameterization_count"])
    base_draw_count = _count(protocol["independent_base_draw_count"])
    shortage_level_count = _number_word(len(protocol["shortage_levels"]))
    sensitivity_scenarios = _count(protocol["sensitivity_matched_scenario_count"])
    sensitivity_evaluations = _count(protocol["sensitivity_controller_evaluation_count"])
    primary_metric_count = _number_word(len(_PRIMARY_METRICS))
    base_parameters = _mapping(sensitivity["base_parameters"], "sensitivity.base_parameters")
    locality_count = len(_sequence(base_parameters["locality_weights"], "locality_weights"))
    sector_count = len(_sequence(base_parameters["sector_shares"], "sector_shares"))
    return _SectionContract(
        "Sensitivity across illustrative demand allocation",
        fragments=(
            _ClaimFragment(
                "sensitivity",
                f"The {_number_word(1)}-factor-at-a-time analysis changes each of "
                f"{_number_word(locality_count)} locality weights and "
                f"{_number_word(sector_count)} sector shares",
            ),
            _ClaimFragment(
                "sensitivity",
                f"by {minus_percent:+.0f}% and {plus_percent:+.0f}%, renormalizing the "
                "affected vector.",
            ),
            _ClaimFragment(
                "sensitivity",
                f"The {variant_count} variants plus the unchanged base give "
                f"{parameterization_count} parameterizations.",
            ),
            _ClaimFragment(
                "sensitivity",
                f"same {base_draw_count} base draws across all {shortage_level_count} shortage "
                "levels and both controllers",
            ),
            _ClaimFragment(
                "sensitivity",
                f"producing {sensitivity_scenarios} matched sensitivity scenarios and "
                f"{sensitivity_evaluations} "
                "controller-scenario evaluations when the base is included.",
            ),
            _ClaimFragment(
                "sensitivity",
                f"All {_count(interval_count)} sensitivity intervals are descriptive, pointwise "
                f"{ci_percent}% intervals",
            ),
            _ClaimFragment(
                "sensitivity",
                f"For every one of the {variant_count} variants, the paired interval remains "
                f"favorable for PPO on all {primary_metric_count} primary outcomes.",
            ),
            _ClaimFragment(
                "sensitivity",
                f"local +/-{abs(plus_percent):.0f}% neighborhood",
            ),
        ),
        tables=(table,),
    )


def _forecast_contract(
    forecast: Mapping[str, Any],
    locality_count: int,
    sector_count: int,
) -> _SectionContract:
    methods = _mapping(forecast["methods"], "forecast.methods")
    trailing = _mapping(methods["trailing_mean_3h"], "forecast.methods.trailing_mean_3h")
    persistence = _mapping(methods["persistence_1h"], "forecast.methods.persistence_1h")
    trailing_pooled = _mapping(trailing["pooled"], "trailing_mean_3h.pooled")
    persistence_pooled = _mapping(persistence["pooled"], "persistence_1h.pooled")
    trailing_locality = _mapping(trailing_pooled["locality_sector"], "trailing locality")
    trailing_system = _mapping(trailing_pooled["system_total"], "trailing system")
    persistence_locality = _mapping(
        persistence_pooled["locality_sector"],
        "persistence locality",
    )
    persistence_system = _mapping(
        persistence_pooled["system_total"],
        "persistence system",
    )
    comparison = _mapping(forecast["comparison"], "forecast.comparison")
    lower_methods = _mapping(comparison["lower_error_method"], "forecast.lower_error_method")
    if set(lower_methods.values()) != {"persistence_1h"}:
        raise ReportResultsClaimError(
            "forecast: report wording requires persistence to have every lower-error measure"
        )
    differences = _mapping(
        comparison["trailing_minus_persistence"],
        "forecast.trailing_minus_persistence",
    )
    target_hours = tuple(
        _integer(value, "forecast.target_hours") for value in forecast["target_hours"]
    )
    table = _TableContract(
        "forecast",
        ("Aggregation", "Method", "MAE", "RMSE", "Lower-error method"),
        (
            (
                "Locality-sector MW",
                "Three-hour trailing mean",
                f"{_number(trailing_locality['mae']):.3f}",
                f"{_number(trailing_locality['rmse']):.3f}",
                "Persistence",
            ),
            (
                "Locality-sector MW",
                "One-hour persistence",
                f"{_number(persistence_locality['mae']):.3f}",
                f"{_number(persistence_locality['rmse']):.3f}",
                "Persistence",
            ),
            (
                "System-total MW",
                "Three-hour trailing mean",
                f"{_number(trailing_system['mae']):.3f}",
                f"{_number(trailing_system['rmse']):.3f}",
                "Persistence",
            ),
            (
                "System-total MW",
                "One-hour persistence",
                f"{_number(persistence_system['mae']):.3f}",
                f"{_number(persistence_system['rmse']):.3f}",
                "Persistence",
            ),
        ),
    )
    trailing_window = _integer(trailing["window_hours"], "trailing_mean_3h.window_hours")
    persistence_window = _integer(persistence["window_hours"], "persistence_1h.window_hours")
    locality_prediction_count = _count(forecast["locality_sector_prediction_count"])
    system_prediction_count = _count(forecast["system_total_prediction_count"])
    return _SectionContract(
        "Synthetic forecast comparison",
        fragments=(
            _ClaimFragment(
                "forecast",
                f"same {_count(forecast['trace_count'])} documented-seed-disjoint synthetic "
                f"demand traces and common target hours {target_hours[0]}-{target_hours[-1]}.",
            ),
            _ClaimFragment(
                "forecast",
                f"The {_number_word(trailing_window)}-hour trailing mean uses only hours "
                f"`t-{trailing_window}` through `t-1`; the {_number_word(persistence_window)}-hour "
                f"persistence baseline uses `t-{persistence_window}`.",
            ),
            _ClaimFragment(
                "forecast",
                f"Each method has {locality_prediction_count} locality-sector predictions and "
                f"{system_prediction_count} "
                "system-total predictions.",
            ),
            _ClaimFragment(
                "forecast",
                f"episode x target x {locality_count} x {sector_count} absolute cell errors",
            ),
            _ClaimFragment(
                "forecast",
                f"all {locality_count * sector_count} locality-sector residuals at a target.",
            ),
            _ClaimFragment(
                "forecast",
                f"Errors are in MW at each {_number_word(persistence_window)}-hour target",
            ),
            _ClaimFragment(
                "forecast",
                f"Persistence has lower error on all {_number_word(len(lower_methods))} stored "
                "measures.",
            ),
            _ClaimFragment(
                "forecast",
                f"are {_number(differences['locality_sector_mae']):+.3f} MW locality-sector MAE, "
                f"{_number(differences['locality_sector_rmse']):+.3f} MW locality-sector RMSE, "
                f"{_number(differences['system_total_mae']):+.3f} MW system MAE, and "
                f"{_number(differences['system_total_rmse']):+.3f} MW system RMSE.",
            ),
        ),
        tables=(table,),
    )


def _baseline_contract(
    accepted: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> _SectionContract:
    cells = tuple(_mapping(value, "accepted cell") for value in accepted["cells"])
    shortage_levels = tuple(str(value) for value in protocol["shortage_levels"])
    critical = _cell_metric_means(cells, shortage_levels, "critical_service_ratio")
    utilization = _cell_metric_means(cells, shortage_levels, "utilization")
    aggregate = _mapping(accepted["aggregate"], "accepted.aggregate")
    unused: dict[str, float] = {}
    for controller in ("Fixed Rotation", "PPO"):
        metrics = _mapping(aggregate[controller], f"accepted.aggregate.{controller}")
        unused[controller] = _number(metrics["available_mwh"]) - _number(metrics["served_mwh"])
    inactive_localities = len(LOCALITIES) - FixedRotationController.active_localities
    return _SectionContract(
        "Baseline interpretation",
        fragments=(
            _ClaimFragment(
                "baseline interpretation",
                f"Fixed Rotation serves {_number_word(FixedRotationController.active_localities)} "
                "consecutive localities at Full Service for each "
                f"{_number_word(FixedRotationController.block_hours)}-hour block and turns the "
                f"other {_number_word(inactive_localities)} "
                "Off.",
            ),
            _ClaimFragment(
                "baseline interpretation",
                "are "
                f"{_human_join([f'{value:.2%}' for value in critical])}, while utilization rises "
                f"from {' to '.join(f'{value:.2%}' for value in utilization)} across "
                f"{_human_join([level.title() for level in shortage_levels])} shortages.",
            ),
            _ClaimFragment(
                "baseline interpretation",
                f"Mean unused available energy is {unused['Fixed Rotation']:,.1f} MWh for Fixed "
                f"Rotation versus {unused['PPO']:,.1f} MWh for PPO.",
            ),
        ),
    )


def _interpretation_contract(forecast: Mapping[str, Any]) -> _SectionContract:
    methods = _mapping(forecast["methods"], "forecast.methods")
    persistence = _mapping(methods["persistence_1h"], "forecast.methods.persistence_1h")
    window_hours = _integer(persistence["window_hours"], "persistence_1h.window_hours")
    return _SectionContract(
        "Interpretation against the proposal questions",
        fragments=(
            _ClaimFragment(
                "forecast interpretation",
                f"{_number_word(window_hours)}-hour persistence beats the "
                "display-only trailing mean on all stored synthetic MAE/RMSE measures.",
            ),
            _ClaimFragment(
                "proposal question labels",
                "Data and rules needed for realism (RQ3):",
            ),
        ),
    )


def _limitations_contract(
    sensitivity: Mapping[str, Any],
    locality_count: int,
    protocol: Mapping[str, Any],
) -> _SectionContract:
    variants = tuple(_mapping(value, "sensitivity variant") for value in sensitivity["variants"])
    factors = sorted({_number(variant["factor"]) for variant in variants})
    service_tier_count = 3
    action_count = service_tier_count**locality_count
    plus_percent = (factors[-1] - 1.0) * 100
    return _SectionContract(
        "Limitations of the result",
        fragments=(
            _ClaimFragment(
                "dispatch limitation",
                f"When requests exceed supply, {_number_word(1)} proportional factor",
            ),
            _ClaimFragment(
                "model scalability",
                f"The actor enumerates `{service_tier_count}^N` actions "
                f"(`{service_tier_count}^{locality_count} = {_count(action_count)}` here), "
                f"limiting scalability beyond {_number_word(locality_count)} localities.",
            ),
            _ClaimFragment(
                "sensitivity",
                f"The +/-{abs(plus_percent):.0f}% one-factor sensitivity",
            ),
            _ClaimFragment(
                "holdout limitation",
                f"{_number_word(len(protocol['seeds'])).title()} documented-seed-disjoint seeds "
                "improve separation and coverage",
            ),
        ),
    )


def _validate_section(
    sections: Mapping[str, str],
    contract: _SectionContract,
) -> None:
    try:
        section = sections[contract.title]
    except KeyError as error:
        raise ReportResultsClaimError(f"missing Results section: {contract.title}") from error
    actual_tables = _markdown_tables(section)
    if len(actual_tables) != len(contract.tables):
        raise ReportResultsClaimError(
            f"{contract.title}: expected {len(contract.tables)} audited table(s), "
            f"found {len(actual_tables)}"
        )
    for actual, expected in zip(actual_tables, contract.tables, strict=True):
        if actual.header != expected.header or actual.rows != expected.rows:
            raise ReportResultsClaimError(
                f"{expected.category}: Markdown table does not match verified evidence"
            )
    _consume_claims_and_reject_unbound_numbers(
        _without_markdown_tables(section),
        contract.fragments,
        category=contract.title,
    )


def _validate_results_document_schema(
    markdown: str,
    sections: Mapping[str, str],
    contracts: Sequence[_SectionContract],
) -> None:
    first_section = re.search(r"^##\s+", markdown, flags=re.MULTILINE)
    if first_section is None:
        raise ReportResultsClaimError("Results preamble: no audited sections found")
    if _normalize(markdown[: first_section.start()]) != "# Results":
        raise ReportResultsClaimError("Results preamble: expected only the canonical title")

    heading_titles = tuple(
        match.group(1).strip()
        for match in re.finditer(r"^##\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
    )
    duplicates = tuple(title for title, count in Counter(heading_titles).items() if count > 1)
    if duplicates:
        raise ReportResultsClaimError(f"duplicate Results section(s): {', '.join(duplicates)}")

    actual_titles = tuple(sections)
    expected_titles = tuple(contract.title for contract in contracts)
    unexpected = tuple(title for title in actual_titles if title not in expected_titles)
    if unexpected:
        raise ReportResultsClaimError(f"unexpected Results section(s): {', '.join(unexpected)}")
    if actual_titles != expected_titles:
        raise ReportResultsClaimError(
            "Results section schema: required sections are missing or out of order"
        )


def _validate_report_document_schema(
    markdown: str,
    sections: Mapping[str, str],
) -> None:
    heading_matches = tuple(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", markdown, flags=re.MULTILINE))
    level_two_titles = tuple(
        match.group(2).strip() for match in heading_matches if match.group(1) == "##"
    )
    duplicates = tuple(title for title, count in Counter(level_two_titles).items() if count > 1)
    if duplicates:
        raise ReportResultsClaimError(f"duplicate REPORT section(s): {', '.join(duplicates)}")

    actual_titles = tuple(sections)
    unexpected = tuple(title for title in actual_titles if title not in _REPORT_SECTION_ORDER)
    if unexpected:
        raise ReportResultsClaimError(f"unexpected REPORT section(s): {', '.join(unexpected)}")
    if actual_titles != _REPORT_SECTION_ORDER:
        raise ReportResultsClaimError(
            "REPORT section schema: required sections are missing or out of order"
        )

    actual_heading_schema = tuple(
        (len(match.group(1)), match.group(2).strip()) for match in heading_matches
    )
    expected_heading_schema = ((1, _REPORT_TITLE),) + tuple(
        (2, title) for title in _REPORT_SECTION_ORDER
    )
    if actual_heading_schema != expected_heading_schema:
        raise ReportResultsClaimError(
            "unexpected REPORT heading: expected one canonical title and the locked sections"
        )
    abstract_heading = next(
        match
        for match in heading_matches
        if match.group(1) == "##" and match.group(2).strip() == "Abstract"
    )
    _consume_claims_and_reject_unbound_numbers(
        markdown[: abstract_heading.start()],
        _REPORT_PREAMBLE_FRAGMENTS,
        category="REPORT preamble",
    )
    _validate_report_content_digest("preamble", markdown[: abstract_heading.start()])


def _validate_report_content_digest(label: str, text: str) -> None:
    expected = _REPORT_NORMALIZED_CONTENT_SHA256[label]
    actual = _canonical_markdown_sha256(text)
    if actual != expected:
        raise ReportResultsClaimError(
            f"REPORT content digest: {label} is not the locked release text"
        )


def _canonical_markdown_sha256(text: str) -> str:
    canonical_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical_newlines.encode("utf-8")).hexdigest()


def _consume_claims_and_reject_unbound_numbers(
    text: str,
    fragments: Sequence[_ClaimFragment],
    *,
    category: str,
) -> None:
    residual = _normalize(text)
    for fragment in fragments:
        expected = _normalize(fragment.text)
        position = residual.find(expected)
        if position < 0:
            raise ReportResultsClaimError(
                f"{fragment.category}: required evidence-derived claim is missing or altered: "
                f"{expected}"
            )
        residual = residual[:position] + residual[position + len(expected) :]
    unsupported = [
        *_DIGIT_BEARING_TOKEN.findall(residual),
        *_NUMBER_WORD_TOKEN.findall(residual),
    ]
    if unsupported:
        raise ReportResultsClaimError(
            f"{category}: unbound numeric claim(s): {', '.join(unsupported)}"
        )


def _markdown_tables(text: str) -> tuple[_TableContract, ...]:
    tables: list[_TableContract] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not _is_table_line(lines[index]):
            index += 1
            continue
        raw_rows: list[tuple[str, ...]] = []
        while index < len(lines) and _is_table_line(lines[index]):
            raw_rows.append(_table_cells(lines[index]))
            index += 1
        rows = tuple(row for row in raw_rows if not _is_separator_row(row))
        if rows:
            tables.append(_TableContract("", rows[0], rows[1:]))
    return tuple(tables)


def _without_markdown_tables(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _is_table_line(line))


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _table_cells(line: str) -> tuple[str, ...]:
    return tuple(_normalize(cell) for cell in line.strip().strip("|").split("|"))


def _is_separator_row(row: Sequence[str]) -> bool:
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in row)


def _level_two_sections(markdown: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", markdown, flags=re.MULTILINE))
    return {
        match.group(1).strip(): markdown[
            match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        ]
        for index, match in enumerate(matches)
    }


def _subsection(
    markdown: str,
    title: str,
    *,
    expected_next_level_two: str,
) -> str:
    match = re.search(rf"^###\s+{re.escape(title)}\s*$", markdown, flags=re.MULTILINE)
    if match is None:
        raise ReportResultsClaimError(f"missing Methodology subsection: {title}")
    following = re.search(
        r"^(#{2,3})\s+(.+?)\s*$",
        markdown[match.end() :],
        flags=re.MULTILINE,
    )
    if (
        following is None
        or following.group(1) != "##"
        or following.group(2).strip() != expected_next_level_two
    ):
        raise ReportResultsClaimError(
            "training parameters: unexpected training subsection or missing next chapter"
        )
    end = match.end() + following.start()
    return markdown[match.end() : end]


def _cell_metric_means(
    cells: Sequence[Mapping[str, Any]],
    shortage_levels: Sequence[str],
    metric: str,
) -> tuple[float, ...]:
    means: list[float] = []
    for level in shortage_levels:
        values = [
            _number(_mapping(cell["Fixed Rotation"], "cell.Fixed Rotation")[metric])
            for cell in cells
            if str(cell["shortage_level"]) == level
        ]
        if not values:
            raise ReportResultsClaimError(
                f"baseline interpretation: no accepted cells for shortage level {level}"
            )
        means.append(fmean(values))
    return tuple(means)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportResultsClaimError(f"verified input field {field} is not a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReportResultsClaimError(f"verified input field {field} is not a sequence")
    return value


def _number(value: Any) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ReportResultsClaimError("verified input contains a non-numeric claim value")
    return float(value)


def _integer(value: Any, field: str) -> int:
    number = _number(value)
    if not number.is_integer():
        raise ReportResultsClaimError(f"verified input field {field} is not an integer")
    return int(number)


def _count(value: Any) -> str:
    return f"{_integer(value, 'count'):,}"


def _number_word(value: int) -> str:
    try:
        return _NUMBER_WORDS[value]
    except KeyError as error:
        raise ReportResultsClaimError(f"cannot render count {value} as a report word") from error


def _human_join(values: Sequence[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _plain_number(value: Any) -> str:
    return format(_number(value), ".15g")


def _normalize(value: str) -> str:
    return " ".join(value.split())
