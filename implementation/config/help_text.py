"""Context and dependencies for configurable reference calculations."""


def extra_explanations() -> dict:
    result = {}

    def put(key, text, unit):
        result[key] = (text, unit)

    for key, role in [
        ("generic_train_per_class", "generic supervised training"),
        ("generic_validation_per_class", "generic early stopping and LR scheduling"),
        ("generic_internal_test_per_class", "evaluation-only generic internal testing"),
    ]:
        put(
            key,
            f"Balanced examples per class reserved from initial training for {role}. All reserved generic examples are removed before private client allocation. Larger counts leave fewer private examples; each class must have enough initial-training support.",
            "images per class",
        )
    for key, stage in [
        ("generic_head_epochs", "frozen-backbone head training"),
        ("generic_finetune_epochs", "trainable-tail fine tuning"),
    ]:
        put(
            key,
            f"Maximum epochs of generic {stage}. EarlyStopping and ReduceLROnPlateau still operate on the generic validation subset. Larger maxima permit more updates but need not be reached. Head training precedes fine tuning.",
            "epochs",
        )
    for key, stage in [
        ("generic_lr_head", "generic head"),
        ("generic_lr_finetune", "generic fine-tuning"),
        ("specialization_initial_lr", "initial owned-class specialization"),
        ("l2_shadow_lr", "received L2 shadow local SGD"),
        ("ditto_global_lr", "Ditto global-path local SGD"),
        ("ditto_personal_lr", "Ditto personalized-path local SGD"),
        ("fedavg_local_lr", "FedAvg client local SGD"),
        ("promotion_lr", "candidate replay/KD Adam"),
    ]:
        put(
            key,
            f"Learning rate for {stage}. It scales optimizer steps and interacts with batch construction, gradient_clip_norm and local update counts. Changing it changes learning dynamics; no performance improvement is implied. Ditto's personal anchor remains the round-start global weights regardless of these rates.",
            "optimizer step size",
        )
    for key, stage in [
        ("generic_batch_size", "generic fit stages"),
        ("specialization_batch_size", "balanced owned/background specialization"),
        ("l2_shadow_batch_size", "received L2 shadow updates"),
        ("ditto_batch_size", "both Ditto update paths, on the same selected batch"),
        ("fedavg_local_batch_size", "plain FedAvg local updates"),
    ]:
        put(
            key,
            f"Images per minibatch in {stage}. Larger values increase memory and change gradient estimates. Deterministic/class-balanced sampling and each method's local-step budget still apply.",
            "images per minibatch",
        )
    for key, role in [
        ("l2_local_steps_per_round", "L2-shadow updates per client"),
        ("ditto_local_steps_per_round", "paired global-then-personal updates per client"),
        ("fedavg_local_steps_per_round", "FedAvg local updates per client"),
    ]:
        put(
            key,
            f"Number of {role} in each communication round. Larger values perform more private-data gradient steps and change the update before aggregation; all eight clients still participate.",
            "local steps per round",
        )
    for key in (
        "l2_class_balanced_batches",
        "ditto_class_balanced_batches",
        "fedavg_class_balanced_batches",
    ):
        put(
            key,
            "Selects class-balanced batch construction rather than the reference deterministic unbalanced sampler for this method's operational local updates. Both retain seed, round and client offsets. It affects batch composition, not ownership or final-test partitions.",
            "boolean",
        )
    put(
        "specialization_min_lr",
        "Lowest adaptive specialist learning rate. After a selection-score plateau the rate is multiplied by specialization_lr_reduce_factor and clipped here. Plateau early stopping requires this floor to be reached; it must not exceed specialization_initial_lr.",
        "optimizer step size",
    )
    put(
        "specialization_lr_reduce_factor",
        "Multiplier applied after specialization_lr_patience non-improving epochs, clipped at specialization_min_lr. A smaller factor reaches the rate floor faster; selection and recall safeguards remain active.",
        "multiplier",
    )
    put(
        "specialization_min_epochs",
        "Minimum specialist epochs before target-based or plateau stopping can occur. Must not exceed specialization_max_epochs. Increasing it requires more training before stopping even if current targets are met.",
        "epochs",
    )
    put(
        "specialization_max_epochs",
        "Maximum specialist epochs per client. Best-snapshot selection uses owned validation metrics and minimum-recall safety; increasing this limit permits additional updates but does not guarantee a better accepted snapshot.",
        "epochs",
    )
    put(
        "specialization_max_owned_min_recall_drop",
        "Allowed drop in minimum supported owned-class recall when choosing a specialist snapshot. Larger values relax this safeguard; score improvement and the configured stopping conditions remain separate.",
        "absolute recall fraction",
    )
    for key in ("l2_blend_candidates", "l3_blend_candidates", "topdown_l3_to_l2_candidates"):
        put(
            key,
            "Convex model-weight interpolation coefficients: (1-alpha)*old + alpha*proposed. All listed candidates are evaluated on the appropriate model-gate validation set with recall/ECE safeguards. Larger alpha moves farther toward the proposed model. This is not a confidence threshold or an explicit trust-region radius.",
            "nonempty coefficient list in (0,1]",
        )
    for key, quantity in [
        ("gate_score_f1_weight", "macro-F1"),
        ("gate_score_accuracy_weight", "accuracy"),
        ("gate_score_balanced_accuracy_weight", "balanced accuracy"),
        ("gate_score_nll_quality_weight", "exp(-negative-log-likelihood)"),
        ("gate_score_calibration_weight", "1 minus ECE"),
    ]:
        put(
            key,
            f"Coefficient on {quantity} in the aggregation model-gate score. It combines with the other gate_score weights without automatic normalization. Changing relative weights changes candidate ranking and the meaning of fixed score-drop/gain tolerances.",
            "score coefficient",
        )
    for level in ("l2", "l3"):
        put(
            f"{level}_exploration_score_drop",
            f"Maximum model-gate score loss for a safe exploratory {level.upper()} blend when no safely improving candidate exists. Recall and ECE safeguards must also pass. Larger values admit more score loss; ties use the preferred exploration alpha.",
            "absolute composite score",
        )
        put(
            f"{level}_hard_score_drop",
            f"Hard model-gate score-loss allowance tested together with recall and ECE for {level.upper()} candidates. Improving selection additionally requires {level}_min_improvement; a hard-safe candidate alone is not sufficient for acceptance.",
            "absolute composite score",
        )
        put(
            f"{level}_min_improvement",
            f"Minimum {level.upper()} candidate score gain for improving selection and best-anchor replacement. Larger values require greater composite-score improvement. A safe exploratory path remains available under its separate tolerance.",
            "absolute composite score",
        )
        put(
            f"{level}_rollback_score_drop",
            f"Maximum selected {level.upper()} score loss relative to its stored best anchor before rollback. Increasing it permits a larger drop before rollback. Anchor scores and recalibration retain the notebook update order.",
            "absolute composite score",
        )
    for prefix, role in [
        ("gate", "hierarchical aggregation"),
        ("promotion", "promotion candidate acceptance"),
    ]:
        put(
            f"{prefix}_base_supported_class_recall_drop",
            f"Base recall-loss tolerance per supported validation class for {role}. The reference adds support-dependent uncertainty and caps it by {prefix}_max_adaptive_class_recall_drop. Larger tolerances relax preservation, independently of owned-class and overall gates.",
            "absolute recall fraction",
        )
        put(
            f"{prefix}_max_adaptive_class_recall_drop",
            f"Cap on the support-adaptive allowed recall loss per validation class for {role}. It bounds the base tolerance plus uncertainty allowance. Increasing it can relax protection for low-support classes; unsupported classes are handled as in the source helper.",
            "absolute recall fraction",
        )
    put(
        "gate_max_ece_increase",
        "Maximum ECE increase from the current model to an aggregation/top-down candidate on model-gate validation. Larger values relax calibration preservation; score and per-class recall conditions must also pass.",
        "absolute ECE fraction",
    )
    for key in ("threshold_grid_min", "threshold_grid_max"):
        put(
            key,
            "Lower or upper confidence-search endpoint before final clipping. Must be ordered with the other endpoint. The grid, Wilson/coverage feasibility, tradeoff fallback and quantile fallback all retain source behavior. Changing endpoints can change chosen thresholds and routing coverage without guaranteeing target accuracy.",
            "confidence probability",
        )
    for key in (
        "l1_threshold_clip_min",
        "l2_threshold_clip_min",
        "l3_threshold_clip_min",
        "threshold_clip_max",
    ):
        put(
            key,
            "Final lower bound for the named model level, or common upper bound, applied to the selected confidence threshold. Every level lower bound must not exceed threshold_clip_max. Clipping can change the achieved calibration accuracy/coverage relative to search feasibility.",
            "confidence probability",
        )
    put(
        "promotion_epochs",
        "Epochs for candidate promotion. Each epoch first runs private replay CE minibatches, then separate teacher KD minibatches; teacher loss is lambda*T^2*KL. Larger values perform more updates before the same acceptance/preservation gates. The source fixes minibatch size at 16.",
        "epochs per attempt",
    )
    for key, metric in [
        ("promotion_max_local_f1_drop", "all-local macro-F1"),
        ("promotion_max_local_accuracy_drop", "all-local accuracy"),
        ("promotion_max_owned_accuracy_drop", "owned accuracy"),
        ("promotion_max_owned_macro_recall_drop", "present-owned macro recall"),
        ("promotion_max_owned_min_recall_drop", "minimum supported owned recall"),
    ]:
        put(
            key,
            f"Maximum allowed loss in {metric} from current L1 to the promotion candidate on private model-gate validation. Increasing it relaxes this preservation requirement. All other preservation checks, teacher-KL improvement and unlabeled acceptance-gain conditions must also pass.",
            "absolute metric fraction",
        )
    for key, event in [
        ("promotion_accept_cooldown_rounds", "accepted promotion"),
        ("promotion_reject_cooldown_rounds", "promotion attempt, including rejected attempts"),
    ]:
        put(
            key,
            f"Minimum rounds since the previous {event} before another attempt is permitted. Monitoring continues while the cooldown is active. Larger values delay eligibility; commissioning/rebaselining and persistence/teacher requirements remain separate.",
            "rounds",
        )
    put(
        "promotion_rebaseline_rounds",
        "Reliance observations collected after an accepted promotion before replacing its baseline. During rebaselining, promotion detection returns early. Completion clears monitoring windows; larger values extend this post-promotion observation stage.",
        "rounds",
    )
    for key, direction in [
        ("count_l2_to_l3_communication", "L2-to-L3 upload"),
        ("count_l3_to_l2_communication", "L3-to-L2 feedback"),
    ]:
        put(
            key,
            f"Includes simulated full-weight bytes for {direction} in communication accounting. It does not disable the corresponding aggregation or feedback and is not a measured network-traffic toggle.",
            "boolean accounting switch",
        )
    return result
