"""Reference-derived training/aggregation. See README.md."""

import numpy as np
from implementation.shared.training.aggregation import Operations as SharedOperations


class Operations(SharedOperations):
    # Reference cells (zero-based): {'tempo': 20}
    def model_gate_score(self, metrics):
        nll_quality = float(np.exp(-max(float(metrics["loss"]), 0.0)))
        calibration_quality = float(np.clip(1.0 - float(metrics["ece"]), 0.0, 1.0))
        return float(
            self.CONFIG["gate_score_f1_weight"] * metrics["macro_f1"]
            + self.CONFIG["gate_score_accuracy_weight"] * metrics["accuracy"]
            + self.CONFIG["gate_score_balanced_accuracy_weight"] * metrics["balanced_accuracy"]
            + self.CONFIG["gate_score_nll_quality_weight"] * nll_quality
            + self.CONFIG["gate_score_calibration_weight"] * calibration_quality
        )

    # Reference cells (zero-based): {'tempo': 20}
    def quality_weighted_average_weights(
        self, weight_sets, sample_counts, quality_scores, power=1.0
    ):
        sample_counts = np.asarray(sample_counts, dtype=np.float64)
        quality_scores = np.asarray(quality_scores, dtype=np.float64)
        quality_scores = np.clip(quality_scores, 0.0001, None)
        combined = sample_counts * np.power(quality_scores, float(power))
        return (self.weighted_average_weights(weight_sets, combined), combined)

    # Reference cells (zero-based): {'tempo': 20}
    def blend_weights(self, old_weights, new_weights, new_weight):
        new_weight = float(new_weight)
        old_weight = 1.0 - new_weight
        blended = []
        for old_tensor, new_tensor in zip(old_weights, new_weights):
            value = old_weight * old_tensor.astype(np.float64) + new_weight * new_tensor.astype(
                np.float64
            )
            blended.append(value.astype(old_tensor.dtype))
        return blended

    # Reference cells (zero-based): {'tempo': 20}
    def adaptive_allowed_recall_drop(self, support_count, base_drop, max_drop):
        if support_count <= 0:
            return float(max_drop)
        one_sample_drop = 1.0 / float(support_count)
        return float(min(max_drop, max(base_drop, one_sample_drop)))

    # Reference cells (zero-based): {'tempo': 20}
    def recall_retention_is_safe(
        self, old_metrics, candidate_metrics, gate_df, base_drop=None, max_drop=None
    ):
        if base_drop is None:
            base_drop = self.CONFIG["gate_base_supported_class_recall_drop"]
        if max_drop is None:
            max_drop = self.CONFIG["gate_max_adaptive_class_recall_drop"]
        details = []
        safe = True
        class_counts = gate_df["label_id"].value_counts().to_dict()
        for class_id, support in sorted(class_counts.items()):
            class_name = self.CLASS_NAMES[int(class_id)]
            old_recall = old_metrics[f"recall_{class_name}"]
            new_recall = candidate_metrics[f"recall_{class_name}"]
            allowed_drop = self.adaptive_allowed_recall_drop(int(support), base_drop, max_drop)
            recall_delta = new_recall - old_recall
            class_safe = recall_delta >= -allowed_drop
            safe = safe and class_safe
            details.append(
                {
                    "class_name": class_name,
                    "support": int(support),
                    "old_recall": old_recall,
                    "new_recall": new_recall,
                    "recall_delta": recall_delta,
                    "allowed_drop": allowed_drop,
                    "safe": class_safe,
                }
            )
        return (safe, details)

    # Reference cells (zero-based): {'tempo': 20}
    def select_trust_region_blend(
        self,
        old_weights,
        proposed_weights,
        gate_df,
        temperature,
        candidate_alphas,
        exploration_score_drop,
        hard_score_drop,
        min_improvement,
    ):
        old_metrics = self.evaluate_weights(old_weights, gate_df, temperature)
        old_score = self.model_gate_score(old_metrics)
        candidates = []
        for alpha in candidate_alphas:
            candidate_weights = self.blend_weights(old_weights, proposed_weights, alpha)
            candidate_metrics = self.evaluate_weights(candidate_weights, gate_df, temperature)
            candidate_score = self.model_gate_score(candidate_metrics)
            recall_safe, recall_details = self.recall_retention_is_safe(
                old_metrics, candidate_metrics, gate_df
            )
            ece_safe = (
                candidate_metrics["ece"]
                <= old_metrics["ece"] + self.CONFIG["gate_max_ece_increase"]
            )
            hard_safe = (
                candidate_score >= old_score - float(hard_score_drop) and recall_safe and ece_safe
            )
            exploration_safe = (
                candidate_score >= old_score - float(exploration_score_drop)
                and recall_safe
                and ece_safe
            )
            candidates.append(
                {
                    "weights": candidate_weights,
                    "alpha": float(alpha),
                    "metrics": candidate_metrics,
                    "score": candidate_score,
                    "score_delta": candidate_score - old_score,
                    "hard_safe": hard_safe,
                    "exploration_safe": exploration_safe,
                    "recall_details": recall_details,
                    "update_norm": self.weight_delta_l2(candidate_weights, old_weights),
                }
            )
        improving = [
            row
            for row in candidates
            if row["hard_safe"] and row["score_delta"] >= float(min_improvement)
        ]
        if improving:
            chosen = sorted(improving, key=lambda row: (row["score"], -row["alpha"]), reverse=True)[
                0
            ]
            mode = "improvement"
        else:
            exploratory = [row for row in candidates if row["exploration_safe"]]
            if exploratory:
                best_exploration_score = max((row["score"] for row in exploratory))
                near_best = [
                    row for row in exploratory if best_exploration_score - row["score"] <= 1e-12
                ]
                chosen = sorted(
                    near_best,
                    key=lambda row: abs(
                        row["alpha"] - self.CONFIG["trust_region_preferred_exploration_alpha"]
                    ),
                )[0]
                mode = "exploration"
            else:
                return {
                    "weights": [w.copy() for w in old_weights],
                    "alpha": 0.0,
                    "metrics": old_metrics,
                    "score": old_score,
                    "old_metrics": old_metrics,
                    "old_score": old_score,
                    "score_delta": 0.0,
                    "update_norm": 0.0,
                    "mode": "rejected",
                    "accepted": False,
                }
        return {
            **chosen,
            "old_metrics": old_metrics,
            "old_score": old_score,
            "mode": mode,
            "accepted": True,
        }

    # Reference cells (zero-based): {'tempo': 47}
    def safe_topdown_l3_to_l2(self, context_name, context_weights, l3_weights, state):
        gate_df = self.context_dataframe(self.client_gate_local, context_name)
        temperature = state["l2_temperature"][context_name]
        base_metrics = self.evaluate_weights(context_weights, gate_df, temperature)
        best = {
            "weights": [w.copy() for w in context_weights],
            "alpha": 0.0,
            "metrics": base_metrics,
            "score": self.model_gate_score(base_metrics),
        }
        for alpha in self.CONFIG["topdown_l3_to_l2_candidates"]:
            candidate = self.blend_weights(context_weights, l3_weights, alpha)
            metrics = self.evaluate_weights(candidate, gate_df, temperature)
            score = self.model_gate_score(metrics)
            recall_safe, _ = self.recall_retention_is_safe(base_metrics, metrics, gate_df)
            ece_safe = metrics["ece"] <= base_metrics["ece"] + self.CONFIG["gate_max_ece_increase"]
            if recall_safe and ece_safe and (score > best["score"] + 1e-08):
                best = {
                    "weights": [w.copy() for w in candidate],
                    "alpha": float(alpha),
                    "metrics": metrics,
                    "score": float(score),
                }
        best["score_delta"] = float(best["score"] - self.model_gate_score(base_metrics))
        return best
