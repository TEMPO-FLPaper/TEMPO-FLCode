"""Portable reference presets, centralized parameter help, and strict validation."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

METHODS = {"tempo": "TEMPO-FL", "ditto": "Ditto", "hfl": "Simple HFL", "flat": "Flat FedAvg"}
PAGES = [
    "Dataset and partitions",
    "Model and initialization",
    "Training",
    "Hierarchy and aggregation",
    "Calibration and routing",
    "Drift and promotion",
    "Evaluation and outputs",
]
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
FIXED = {
    "experiment_version",
    "dataset_name",
    "trashnet_zip_url",
    "image_extensions",
    "train_initial_ratio",
    "validation_ratio",
    "online_ratio",
    "test_ratio",
    "specialization_search_mode",
    "num_clients",
    "num_contexts",
    "clients_per_context",
    "context_names",
    "planned_classes",
    "shift_transform_names",
    "project_root",
    "auto_resume",
}
HIDDEN = {
    "project_root",
    "auto_resume",
    "specialization_show_details",
    "specialization_print_per_epoch",
    "specialization_print_per_class_each_epoch",
}


def project_root():
    if os.environ.get("TEMPO_FL_HOME"):
        return Path(os.environ["TEMPO_FL_HOME"]).expanduser().resolve()
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").is_file() and (path / "TEMPO_FL.pdf").is_file():
            return path
    return Path(__file__).resolve().parents[2]


def preset_path(method: str, preset: str = "reference") -> Path:
    """Resolve a packaged method preset independently of the working directory."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if preset not in {"reference", "verification"}:
        raise ValueError(f"Unknown preset: {preset}")
    folder = "tempo_fl" if method == "tempo" else method
    return Path(__file__).resolve().parent / folder / f"{preset}.json"


def reference(method):
    return json.loads(
        preset_path(method).read_text(encoding="utf-8")
    )


def category(key):
    if key.startswith("specialization_"):
        return PAGES[2]
    if key.startswith(
        (
            "promotion_",
            "shift_",
            "temporary_",
            "persistent_",
            "new_normal_",
            "teacher_",
            "absolute_",
            "reliance_",
            "persistence_",
            "higher_resolution",
            "max_teacher",
            "min_teacher",
            "commissioning_",
        )
    ):
        return PAGES[5]
    if key.startswith(
        (
            "threshold_",
            "l1_threshold",
            "l2_threshold",
            "l3_threshold",
            "target_selective",
            "min_accept",
            "reliability_",
        )
    ):
        return PAGES[4]
    if key.startswith(
        (
            "bootstrap_",
            "gate_",
            "trust_",
            "topdown_",
            "l2_blend",
            "l3_blend",
            "l2_exploration",
            "l3_exploration",
            "l2_hard",
            "l3_hard",
            "l2_min",
            "l3_min",
            "l2_rollback",
            "l3_rollback",
            "l3_context",
            "num_context",
            "clients_per",
            "context_names",
        )
    ):
        return PAGES[3]
    if key in {
        "image_size",
        "mobilenet_alpha",
        "dropout",
        "trainable_tail_layers",
        "v12_2_generic_checkpoint_path",
        "generic_source_project_root",
    }:
        return PAGES[1]
    if key.startswith(
        (
            "generic_",
            "ditto_",
            "fedavg_",
            "l2_",
            "tf_data_",
            "gradient_",
            "early_stopping",
            "reduce_lr",
        )
    ) or key in {"seed", "num_rounds"}:
        return PAGES[2]
    if key.startswith(("evaluation_", "events_", "export_", "prefer_", "tflite_", "count_")):
        return PAGES[6]
    return PAGES[0]


EXPLANATIONS = {
    "seed": (
        "Seeds Python, NumPy, TensorFlow, partitions and per-round sampling. Changing it changes allocations and initialization. Cross-platform bitwise equality is not guaranteed.",
        "integer seed",
    ),
    "num_rounds": (
        "Operational communication rounds after preparation. The phase calendar remains fixed at rounds 20, 22, 40 and 65; reduced runs do not compress it.",
        "rounds",
    ),
    "image_size": (
        "Square bilinear image resize before MobileNetV2 preprocessing to [-1,1]. Larger sizes increase computation and can change the ImageNet weight file.",
        "pixels",
    ),
    "mobilenet_alpha": (
        "MobileNetV2 width multiplier with ImageNet weights. Larger values increase channels and memory use; accuracy effects require measurement.",
        "multiplier",
    ),
    "trainable_tail_layers": (
        "Final N backbone layers are trainable after head training. The source graph calls the backbone with training=False, but actual nested-model BatchNorm behavior depends on Keras. Verification compares its moving statistics to the original functions. Larger counts increase trainable parameters.",
        "layers",
    ),
    "dropout": (
        "Drop probability after global average pooling and before six logits. Active in training only. Larger values increase regularization noise.",
        "fraction",
    ),
    "ditto_lambda": (
        "Personalized loss is cross entropy + lambda/2 times squared distance to the round-start global weights over trainable variables. Global updates have no proximal penalty. Increasing lambda strengthens the global anchor.",
        "coefficient",
    ),
    "l2_prox_mu": (
        "Shadow objective adds mu/2 times squared distance to received L2 weights. The expression uses all weights; only trainable variables are updated.",
        "coefficient",
    ),
    "promotion_kd_lambda": (
        "Multiplier on the temperature-squared KL loss in separate teacher minibatch updates. Replay uses separate cross-entropy updates; it is not multiplied by (1-lambda).",
        "loss multiplier",
    ),
    "promotion_kd_temperature": (
        "Temperature for raw teacher and student logits. Training KL is multiplied by temperature squared; the KL acceptance helper is not.",
        "temperature",
    ),
    "trust_region_preferred_exploration_alpha": (
        "Tie-break preference among equally scoring safe exploratory blends. Alpha interpolates weights; it is not a confidence threshold or an explicit trust-region radius.",
        "blend coefficient",
    ),
    "higher_resolution_threshold": (
        "Minimum pooled higher-level acceptance among escalated events over a complete window. Resolution means confidence acceptance; audit labels measure correctness separately.",
        "fraction",
    ),
    "persistence_rounds": (
        "Bounded monitoring-window length. The trigger uses pooled reliance, current-round reliance, pooled resolution, escalation counts and recent teachers. It does not require every individual round rate to exceed its threshold.",
        "rounds",
    ),
    "planned_classes": (
        "Fixed reference ownership plan. Multinomial allocation favors owned classes while retaining background examples. Models output all six labels.",
        "class lists",
    ),
    "dataset_mode": (
        "Download resized TrashNet or choose a local copy with the same six class folders. No second dataset is supported.",
        "choice",
    ),
    "custom_dataset_dir": (
        "Local directory containing cardboard, glass, metal, paper, plastic and trash folders. Required in local-folder mode.",
        "directory",
    ),
    "calibration_fraction_of_validation": (
        "Stratified fraction of global validation used for temperatures and thresholds. The remainder supplies model-selection gates. Neither is the final test.",
        "fraction",
    ),
    "validation_subsplit_fraction": (
        "Flat FedAvg development split retained for alignment. It does not calibrate inference.",
        "fraction",
    ),
    "v12_2_generic_checkpoint_path": (
        "Optional matched TEMPO generic .weights.h5 file. As in the reference, Ditto trains generic stages first, then replaces weights if this is supplied. Empty means recreate the protocol.",
        "optional file",
    ),
    "generic_source_project_root": (
        "Optional TEMPO run directory containing 00_config/generic_seed_v12_2.weights.h5. Flat FedAvg loads it after generic training. Empty uses its own generic weights.",
        "optional directory",
    ),
    "tf_data_prefetch_batches": (
        "Bounded prefetch depth. Larger values consume more RAM without guaranteed speed gains. Keep small for memory-safe execution.",
        "batches",
    ),
    "gradient_clip_norm": (
        "Global norm limit for manual updates. Fit and promotion retain the reference Adam clipnorm behavior. Larger limits permit larger gradients.",
        "norm",
    ),
    "events_per_client_per_round": (
        "Operational draws per client per round. Sampling can repeat images; events are not unique test samples. Baselines retain this as unused metadata.",
        "events",
    ),
    "export_tflite_at_end": (
        "Exports method-specific models after requested rounds. INT8 uses generic training images as representatives; supported conversion failures fall back to dynamic-range quantization.",
        "boolean",
    ),
}

EXPLANATIONS.update({
    'planned_class_assignment_weight':('Relative multinomial weight for assigning examples of a class to its planned owners, after the minimum owned allocation. Increasing it relative to the background weight concentrates class support among owners. It does not change class IDs.','relative allocation weight'),
    'background_class_assignment_weight':('Relative allocation weight for clients that do not own a class. Increasing it distributes more background examples to other clients and reduces ownership concentration. At least one allocation weight must be positive.','relative allocation weight'),
    'min_owned_samples_per_split_when_possible':('Reserves this many shuffled examples for each owner before multinomial allocation, only when the split contains enough examples for all owners. Larger values use more of a small split for guaranteed owner support.','images per owner per class'),
    'min_client_train_samples':('Minimum allowed private training support after planned-class partitioning. A smaller client causes an actionable error; examples are not moved between clients to satisfy this check. Zero disables the size requirement.','images per client'),
    'generic_use_class_weights':('Uses clipped inverse-frequency weights for generic training. The balanced reference seed normally gives equal class weights. This does not introduce additional training examples.','boolean'),
    'generic_class_weight_min':('Lower clip bound for inverse-frequency generic class weights. Raising it limits down-weighting of frequent classes. It must not exceed the maximum; balanced seeds normally have weight one.','loss multiplier'),
    'generic_class_weight_max':('Upper clip bound for inverse-frequency generic class weights. Lowering it limits up-weighting of rare classes. It must not be below the minimum; balanced seeds normally have weight one.','loss multiplier'),
    'early_stopping_patience':('Generic head/fine-tuning epochs without improved validation loss before EarlyStopping. Best weights are restored. Larger patience allows longer training on plateaus.','epochs'),
    'reduce_lr_patience':('Generic epochs without improved validation loss before ReduceLROnPlateau lowers the rate. Larger values wait longer before reducing it. The source minimum rate is 1e-6.','epochs'),
    'reduce_lr_factor':('Multiplier applied to the generic learning rate on a validation-loss plateau. Smaller factors reduce the rate more sharply. Must be greater than zero and below one.','multiplier'),
    'specialization_background_fraction':('Samples background examples up to this fraction of balanced owned-class support, capped by available background data. Increasing it includes more background training alongside oversampled owned classes.','fraction of balanced owned support'),
    'specialization_min_owned_score_gain':('Retained notebook configuration field. The designated specialization loop does not consume it; changing it has no effect. Actual selection uses specialization_min_score_delta and minimum-owned-recall safety.','unused source metadata'),
    'specialization_min_score_delta':('Minimum composite owned-selection score gain needed to replace the best snapshot. Larger values require a larger gain and can increase plateau counters; minimum-owned-recall preservation must also pass.','absolute score gain'),
    'specialization_owned_macro_recall_weight':('Coefficient on present-owned-class macro recall in specialist selection. Larger values emphasize equal-class owned recall relative to owned accuracy and fixed-six-class all-local F1. Coefficients are not automatically normalized.','selection-score coefficient'),
    'specialization_owned_accuracy_weight':('Coefficient on owned accuracy in specialist selection. Larger values emphasize support-weighted owned correctness relative to the other score terms. Coefficients are not automatically normalized.','selection-score coefficient'),
    'specialization_all_local_f1_weight':('Coefficient on fixed-six-class all-local Macro-F1 in fast specialist selection. Larger values increase its influence relative to owned recall/accuracy; unsupported classes retain the source zero convention.','selection-score coefficient'),
    'specialization_stop_when_target_met':('Allows early stopping once the current candidate reaches both owned accuracy and owned macro-recall targets after the minimum epochs and passes minimum-recall safety. It does not bypass best-snapshot selection.','boolean'),
    'specialization_target_owned_accuracy':('Owned accuracy target used with macro recall for target-based stopping. Higher targets can require more epochs, up to the configured maximum; they do not guarantee final accuracy.','fraction'),
    'specialization_target_owned_macro_recall':('Present-owned-class macro recall target used with owned accuracy for stopping after the minimum epochs. Higher targets make this stopping condition harder to meet.','fraction'),
    'specialization_lr_patience':('Non-improving owned-score epochs before adaptive specialist LR reduction. Larger values wait longer before multiplying the rate by its reduction factor.','epochs'),
    'specialization_early_stop_patience':('Non-improving owned-score epochs required for plateau stopping, after minimum epochs and only once the minimum learning rate has been reached.','epochs'),
    'bootstrap_use_owned_quality_weights':('Permits an initial L2 candidate weighted by owned training support and validation owned recall. It is selected only if its gate score is at least the uniform specialist candidate; disabling this selects uniform L2 averaging.','boolean'),
    'bootstrap_quality_floor':('Lower bound on quality in owned-support weighting and context bootstrap quality. Raising the floor reduces the relative disadvantage of low-quality models. The source still compares candidates on validation.','quality-score floor'),
    'bootstrap_l3_context_quality_power':('Exponent on context quality when weighting initial L3 by context support. Larger exponents emphasize quality differences. The quality-weighted and uniform candidates are still compared by the gate.','exponent'),
    'l3_context_quality_power':('Exponent on pre-top-down context gate scores in operational L3 support/quality aggregation. Larger values emphasize relative context quality. It is independent of inference confidence thresholds.','exponent'),
    'target_selective_accuracy':('Desired accepted-sample accuracy in calibration threshold search. Increasing it makes candidate feasibility stricter, but the reference tradeoff/quantile fallbacks can return a threshold that does not meet the target.','fraction'),
    'target_selective_lower_bound':('Minimum Wilson lower bound in threshold candidate feasibility, using the reference z=1.0. Increasing it makes the filter stricter. Fallbacks and clipping still apply; this is not a guaranteed 95% bound.','fraction'),
    'min_accept_coverage':('Minimum calibration coverage used when choosing a threshold. Larger values favor accepting more calibration inputs. Fallbacks and final clipping can change achieved coverage, which is reported separately.','fraction'),
    'threshold_grid_step':('Spacing between searched calibration confidence thresholds. Smaller steps examine more candidates and cost more evaluations; they do not guarantee better operational accuracy.','probability step'),
    'reliability_bin_edges':('Strictly increasing confidence-bin boundaries covering [0,1]. They define shrunk empirical correctness estimates for audit and forced-cascade diagnostics. They do not replace confidence-based operational acceptance.','probability boundaries'),
    'reliability_prior_strength':('Pseudo-count strength shrinking bin correctness toward overall calibration accuracy. Larger values reduce the influence of small-bin observations. Reliability is audit/forced-diagnostic information, not an operational label signal.','pseudo samples'),
    'teacher_confidence_margin':('Added to a higher model acceptance threshold before clipping to the teacher min/max. A larger margin usually admits fewer teacher samples; it does not change operational acceptance itself.','probability margin'),
    'teacher_threshold_min':('Lower clip bound on teacher confidence eligibility. Raising it can shrink the teacher buffer. It must not exceed teacher_threshold_max and is distinct from the operational routing thresholds.','probability'),
    'teacher_threshold_max':('Upper clip bound on the adaptive teacher confidence threshold. Lowering it can admit more otherwise accepted higher-level examples; it is not a model-blending coefficient.','probability'),
    'max_teacher_buffer':('Maximum recent teacher items retained per client in a bounded deque. Older entries are evicted when full. Larger capacity uses more memory and must accommodate the minimum promotion sample count.','teacher items per client'),
    'teacher_memory_rounds':('Maximum teacher age included in a promotion attempt: current round minus recorded round is at most this value. Longer memory admits older evidence, which can be less current under drift.','rounds'),
    'min_teacher_samples_for_promotion':('Minimum number of recent confident teacher items before an attempt can train. Increasing it delays or prevents attempts when confident higher-level coverage is sparse; the bounded buffer must be at least this large.','teacher items'),
    'promotion_replay_samples':('Maximum local private-training examples sampled for replay during a promotion attempt, capped by available support. Larger replay sets add retention updates and computation. Operational/final-test labels are not used.','images per attempt'),
    'promotion_min_teacher_kl_improvement':('Required decrease in raw-logit teacher/student KL on teacher items, using the acceptance helper without T-squared scaling. Larger values make candidate acceptance stricter. Preservation and acceptance-gain gates must also pass.','mean KL decrease'),
    'promotion_min_acceptance_gain':('Required increase in student confidence acceptance on teacher paths at the existing L1 temperature/threshold. It is unlabeled acceptance, not correctness. Larger values make acceptance stricter.','absolute acceptance-rate gain'),
    'promotion_min_escalated_events_window':('Minimum pooled escalated-event count in a full persistence window. Higher values require more evidence before a promotion can trigger. Events may repeat images and are not unique-sample support.','events per window'),
    'absolute_reliance_floor':('Minimum reliance demand for promotion, combined by max with baseline reliance plus the configured increase. Raising it requires more escalation in both the pooled window and current round.','fraction'),
    'reliance_increase_over_baseline':('Increase above the commissioning/rebaselined reliance required for promotion, subject to the absolute floor. Larger values require more sustained demand; pooled resolution and teacher gates remain separate.','absolute fraction'),
    'commissioning_rounds':('Rounds used to accumulate the initial reliance baseline; also a scheduled shift-profile discovery boundary. Changing it does not rescale the fixed scenario calendar in scenario_for_round.','rounds'),
    'persistent_profile_refresh_round':('Second round at which the source rebuilds eligible hard-and-resolvable shift profiles. Changing it moves discovery, not the fixed phase calendar. The default is round 40.','round index'),
    'shift_target_pool_size':('Target total strict, source-unique examples for each client shift profile, divided into adaptation and scenario-evaluation roles. Larger pools demand more discovered higher-level advantage. Unsupported clients remain controls.','source examples per client'),
    'shift_source_eval_fraction':('Fraction of online source paths reserved for the shift-specific scenario-evaluation role before discovery. Larger fractions leave fewer adaptation source paths. Normal online draws are not made globally disjoint by this split.','fraction'),
    'shift_min_stream_pool_size':('Minimum selected strict adaptation examples for eligibility. Larger requirements can classify more clients as controls. Offline benchmark construction uses labels; routing and promotion do not.','source examples'),
    'shift_min_eval_pool_size':('Minimum selected strict scenario-evaluation examples for eligibility. Increasing it can reduce eligibility. These are constructed online holdout sources, not the final-test split.','source examples'),
    'shift_max_fraction_per_class':('First-pass class cap while selecting ranked source-unique shift candidates. If the target remains unfilled, the reference fills from remaining candidates, so this is not an unconditional final class-ratio guarantee.','fraction of target pool'),
    'temporary_shift_hard_fraction':('Fraction of operational draws taken from an eligible hard pool in fixed temporary rounds 21–22. More hard draws increase exposure to the constructed scenario; no accuracy benefit is guaranteed.','fraction of draws'),
    'persistent_shift_hard_fraction':('Base hard-draw fraction in persistent rounds 41–65. Discovery can raise the per-client fraction to exceed baseline-adjusted reliance demand, capped by the source rule.','fraction of draws'),
    'new_normal_hard_fraction':('Hard-pool draw fraction for eligible clients after round 65. Raising it changes the constructed exposure mix; controls remain on normal client pools.','fraction of draws'),
    'evaluation_samples_per_client':('TEMPO operational-evaluation draws per client at evaluation rounds, sampled using the separate source salt. Increasing it increases evaluation work, not the unique final-test set. This is unused metadata in the baseline loops.','draws per client'),
    'evaluation_every':('Regular evaluation cadence in communication rounds, in addition to the source forced evaluation rounds. Smaller values evaluate more often; TEMPO also refreshes gates at its scheduled boundaries.','rounds'),
    'prefer_full_int8_tflite':('Requests full INT8 using the source representative data and INT8 input/output types. Disabling it takes the reference dynamic-range fallback branch. The Keras 3 serialization bridge changes only the converter input API.','boolean'),
    'tflite_representative_samples':('Maximum prefix of generic training images used for INT8 calibration, capped by available generic data. Larger values increase conversion work; neither online nor final-test images are calibration data.','images'),
    'tf_data_num_parallel_calls':('Bounded map concurrency for image decode/augmentation/preprocessing. Increasing it may improve throughput but uses more CPU and buffering; private-pool limits still apply.','parallel map calls'),
    'tf_data_private_threadpool_size':('Private thread pool size for each tf.data pipeline. Larger pools can increase concurrency and contention; the source keeps intra-op parallelism at one.','threads'),
})

SHARED_PARAMETER_KEYS={
    'seed','dataset_name','dataset_mode','custom_dataset_dir','trashnet_zip_url','image_extensions',
    'train_initial_ratio','validation_ratio','online_ratio','test_ratio','num_clients','planned_classes',
    'planned_class_assignment_weight','background_class_assignment_weight','min_owned_samples_per_split_when_possible',
    'min_client_train_samples','image_size','mobilenet_alpha','dropout','trainable_tail_layers','gradient_clip_norm',
    'generic_train_per_class','generic_validation_per_class','generic_internal_test_per_class',
    'generic_head_epochs','generic_finetune_epochs','generic_batch_size','generic_lr_head','generic_lr_finetune',
    'early_stopping_patience','reduce_lr_patience','reduce_lr_factor','generic_use_class_weights',
    'generic_class_weight_min','generic_class_weight_max','tf_data_num_parallel_calls',
    'tf_data_prefetch_batches','tf_data_private_threadpool_size','num_rounds',
}


from implementation.config.help_text import extra_explanations
EXPLANATIONS.update(extra_explanations())


def bounds(key, default):
    if key == "seed":
        return 0, 2**31 - 1
    if key == "image_size":
        return 32, 512
    if key == "mobilenet_alpha":
        return 0.35, 1.4
    if key == "trainable_tail_layers":
        return 1, 154
    if isinstance(default, int):
        return (
            0
            if any(
                s in key
                for s in ("min_client", "min_owned", "commissioning", "cooldown", "refresh")
            )
            else 1
        ), 100000
    if key == "dropout":
        return 0, 0.99
    if key == "threshold_grid_step":
        return 1e-6, 1
    if key in {"reduce_lr_factor", "specialization_lr_reduce_factor"}:
        return 1e-6, 0.999999999
    if key.endswith(("_lr", "_lr_head", "_lr_finetune")) or key in {
        "specialization_initial_lr",
        "specialization_min_lr",
    }:
        return 1e-9, 1
    if key in {"promotion_kd_temperature", "reliability_prior_strength", "gradient_clip_norm"}:
        return 1e-6, 1000
    if "weight" in key:
        return 0, 100
    if any(
        s in key
        for s in (
            "ratio",
            "fraction",
            "drop",
            "threshold",
            "coverage",
            "accuracy",
            "recall",
            "gain",
            "score_delta",
            "factor",
            "alpha",
            "improvement",
            "reliance",
            "floor",
        )
    ):
        return 0, 1
    return 0, 10000


@dataclass(frozen=True)
class Parameter:
    key: str
    default: Any
    page: str
    editable: bool
    explanation: str
    unit: str

    @property
    def label(self):
        return self.key.replace("_", " ").capitalize()

    def help(self):
        constraint = "Same JSON type as the reference value."
        if isinstance(self.default, (int, float)) and not isinstance(self.default, bool):
            lo, hi = bounds(self.key, self.default)
            constraint = f"Valid range: {lo:g} to {hi:g}."
        if not self.editable:
            constraint += " Fixed executable-reference setting."
        sources = Path(__file__).with_name("parameter_sources.json")
        locations = json.loads(sources.read_text(encoding="utf-8")).get(self.key, []) if sources.exists() else []
        source = "; ".join(locations) or "Method CONFIG (cell 3); see README.md"
        return f"{self.label}\n\n{self.explanation}\n\nType: {type(self.default).__name__}. Unit: {self.unit}. {constraint}\n\nReference default: {json.dumps(self.default)}\n\nOperates in: {self.page}. Source: {source}\n\nChanging a setting creates a departure from Reference. Effects on accuracy require measurement."


def metadata(method):
    result = []
    for key, default in reference(method)["settings"].items():
        if key in HIDDEN:
            continue
        text, unit = EXPLANATIONS.get(key, ("", "reference units"))
        label = key.replace("_", " ")
        if not text:
            if "lr" in key:
                text = f"Learning rate for {label}. Larger steps can accelerate changes and increase instability. Interacts with gradient clipping and the local update count."
                unit = "step size"
            elif "batch" in key:
                text = f"Controls {label}. Larger batches change sampling and gradients and require more RAM."
                unit = "images or batches"
            elif any(s in key for s in ("epochs", "steps", "patience", "cooldown", "rounds")):
                text = f"Controls {label}. Larger counts increase training or waiting time; the reference stopping and acceptance rules still apply."
                unit = (
                    "epochs"
                    if "epoch" in key
                    else "steps"
                    if "step" in key
                    else "rounds"
                    if "round" in key
                    else "checks"
                )
            elif "threshold" in key or "coverage" in key:
                text = f"Controls {label} in calibration or routing. Higher confidence thresholds generally reduce coverage. Calibration samples, clipping and fallback selection rules interact with this value."
                unit = "fraction"
            elif any(s in key for s in ("drop", "improvement", "gain")):
                text = f"Controls {label} in validation safeguards. Larger allowed drops relax preservation; larger required gains make acceptance stricter. Values are absolute fractions."
                unit = "absolute fraction"
            elif "blend" in key or "candidates" in key:
                text = f"Candidate weight-blending coefficients for {label}. Each candidate is tested on model-gate validation. Larger coefficients move farther toward the proposed weights."
                unit = "coefficient list"
            elif key.startswith("generic_"):
                text = f"Controls {label} for the balanced generic model. Data come only from initial training; larger seed sample counts leave less private federated data."
            else:
                text = f"Controls {label} in {category(key).lower()}. It participates in the source calculation together with the related settings. See the reference mapping for exact interactions; no accuracy gain is guaranteed."
        result.append(Parameter(key, default, category(key), key not in FIXED, text, unit))
    return result


@dataclass
class RunConfig:
    method: str
    settings: dict
    output_root: str
    dataset_cache: str
    preset: str = "Reference"
    schema_version: int = 1
    protocol_label: str = "research"

    def __post_init__(self):
        if not isinstance(self.settings, dict):
            raise ValueError("Settings must be a JSON object for the selected method.")
        for key in ("output_root", "dataset_cache"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key}: a nonempty directory path is required.")
            value = Path(getattr(self, key)).expanduser()
            if not value.is_absolute():
                value = project_root() / value
            setattr(self, key, str(value.resolve()))
        for key in ("custom_dataset_dir", "v12_2_generic_checkpoint_path", "generic_source_project_root"):
            if self.settings.get(key):
                value = Path(self.settings[key]).expanduser()
                self.settings[key] = str((value if value.is_absolute() else project_root() / value).resolve())

    @classmethod
    def defaults(cls, method="tempo"):
        root = project_root()
        return cls(
            method,
            deepcopy(reference(method)["settings"]),
            str(root / "runs"),
            str(root / "datasets"),
        )

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported configuration schema.")
        if self.protocol_label not in {"research", "verification"}:
            raise ValueError("Protocol label must be research or verification.")
        if not isinstance(self.preset, str) or not self.preset.strip():
            raise ValueError("Preset name is required.")
        base = reference(self.method)["settings"]
        if self.settings.keys() != base.keys():
            raise ValueError(
                "Configuration keys do not match the selected method. Restore Reference defaults."
            )
        errors = []
        s = self.settings
        for key, value in s.items():
            default = base[key]
            if type(value) is not type(default):
                errors.append(f"{key}: expected {type(default).__name__}")
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lo, hi = bounds(key, default)
                if not math.isfinite(value) or not lo <= value <= hi:
                    errors.append(f"{key}: expected {lo:g}..{hi:g}")
            if key in FIXED and value != default:
                errors.append(f"{key}: fixed reference value {default!r}")
        if errors:
            raise ValueError("\n".join(errors))
        for lo, hi in [
            ("generic_class_weight_min", "generic_class_weight_max"),
            ("specialization_min_lr", "specialization_initial_lr"),
            ("specialization_min_epochs", "specialization_max_epochs"),
            ("threshold_grid_min", "threshold_grid_max"),
            ("teacher_threshold_min", "teacher_threshold_max"),
        ]:
            if lo in s and s[lo] > s[hi]:
                errors.append(f"{lo} must not exceed {hi}")
        if (
            "max_teacher_buffer" in s
            and s["min_teacher_samples_for_promotion"] > s["max_teacher_buffer"]
        ):
            errors.append("Promotion teacher minimum cannot exceed its bounded buffer capacity.")
        for key in ("l1_threshold_clip_min", "l2_threshold_clip_min", "l3_threshold_clip_min"):
            if key in s and s[key] > s["threshold_clip_max"]:
                errors.append(f"{key} must not exceed threshold_clip_max.")
        if s["planned_class_assignment_weight"] + s["background_class_assignment_weight"] <= 0:
            errors.append("At least one class-assignment weight must be positive.")
        for key in ("l2_blend_candidates", "l3_blend_candidates", "topdown_l3_to_l2_candidates"):
            if key in s and (
                not s[key] or any(type(x) not in (int, float) or not 0 < x <= 1 for x in s[key])
            ):
                errors.append(f"{key}: nonempty coefficients in (0,1] required")
        if "reliability_bin_edges" in s:
            e = s["reliability_bin_edges"]
            if (
                len(e) < 2
                or any(type(x) not in (int, float) for x in e)
                or any(not math.isfinite(x) for x in e)
                or any(a >= b for a, b in zip(e, e[1:]))
                or e[0] > 0
                or e[-1] < 1
            ):
                errors.append("Reliability bins must strictly increase and cover [0,1].")
        if s["mobilenet_alpha"] not in {0.35, 0.5, 0.75, 1.0, 1.3, 1.4}:
            errors.append("ImageNet alpha must be 0.35, 0.5, 0.75, 1.0, 1.3 or 1.4.")
        fraction = s.get("calibration_fraction_of_validation", s.get("validation_subsplit_fraction"))
        if not 0 < fraction < 1:
            errors.append("Validation subsplit fraction must be strictly between zero and one.")
        if s["dataset_mode"] not in {"trashnet_resized", "local_folder"}:
            errors.append("Unsupported dataset mode.")
        if s["dataset_mode"] == "local_folder":
            p = Path(s["custom_dataset_dir"]).expanduser()
            if not p.is_dir() or not all((p / c).is_dir() for c in CLASSES):
                errors.append("Local TrashNet root must contain all six reference class folders.")
        for key in ("v12_2_generic_checkpoint_path", "generic_source_project_root"):
            if s.get(key):
                p = Path(s[key]).expanduser()
                if key == "generic_source_project_root":
                    p = p / "00_config/generic_seed_v12_2.weights.h5"
                if not p.is_file():
                    errors.append(f"{key}: requested generic weights do not exist.")
        for label, value in [
            ("Output root", self.output_root),
            ("Dataset cache", self.dataset_cache),
        ]:
            if not value.strip():
                errors.append(f"{label} is required.")
            elif Path(value).exists() and not Path(value).is_dir():
                errors.append(f"{label} must be a directory.")
        if errors:
            raise ValueError("\n".join(errors))

    def departures(self):
        base = reference(self.method)["settings"]
        return {
            k: {"reference": base[k], "selected": v}
            for k, v in self.settings.items()
            if v != base[k]
        }

    def to_dict(self):
        identity = json.loads(Path(__file__).with_name("reference_sources.json").read_text(encoding="utf-8"))[self.method]
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "preset": self.preset,
            "protocol_label": self.protocol_label,
            "reference_notebook": reference(self.method)["reference_notebook"],
            "reference_sha256": identity["sha256"],
            "settings": self.settings,
            "output_root": str(Path(self.output_root).expanduser().resolve()),
            "dataset_cache": str(Path(self.dataset_cache).expanduser().resolve()),
            "departures": self.departures(),
        }

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("Configuration must be a JSON object.")
        if type(d.get("schema_version", 1)) is not int or d.get("schema_version", 1) != 1:
            raise ValueError("Unsupported configuration schema.")
        identity = json.loads(Path(__file__).with_name("reference_sources.json").read_text(encoding="utf-8"))[d["method"]]
        if d.get("reference_notebook", identity["notebook"]) != identity["notebook"] or d.get("reference_sha256", identity["sha256"]) != identity["sha256"]:
            raise ValueError("Configuration reference identity does not match the selected implementation.")
        if "output_root" not in d:
            c = cls.defaults(d["method"])
            c.settings = d["settings"]
            c.__post_init__()
            c.validate()
            return c
        c = cls(
            **{
                k: d[k]
                for k in (
                    "method",
                    "settings",
                    "output_root",
                    "dataset_cache",
                    "preset",
                    "schema_version",
                    "protocol_label",
                )
            }
        )
        c.validate()
        return c

    def fingerprint(self):
        data = {
            "method": self.method,
            "schema_version": self.schema_version,
            "settings": self.settings,
            "protocol_label": self.protocol_label,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
