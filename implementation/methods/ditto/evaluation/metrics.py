"""Reference-derived evaluation/metrics. See README.md."""

import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from implementation.shared.evaluation.metrics import Operations as SharedOperations


class Operations(SharedOperations):
    # Reference cells (zero-based): {'ditto': 19}
    def evaluate_general_stage_rows__ditto(
        self, stage, stage_detail, stage_order, model_specs, test_set=None
    ):
        if test_set is None:
            test_set = self.general_test_set
        rows = []
        for spec in model_specs:
            model_type = self._ditto_model_type(spec)
            model_id = str(spec.get("model_id", "Global" if model_type == "Global" else ""))
            metrics = self.evaluate_weights(spec["weights"], test_set, spec.get("temperature", 1.0))
            rows.append(
                {
                    "stage": stage,
                    "stage_detail": stage_detail,
                    "stage_order": int(stage_order),
                    "level": model_type,
                    "model_type": model_type,
                    "model_id": model_id,
                    "client_id": spec.get("client_id", ""),
                    "context": spec.get("context", ""),
                    "test_set": "general_test_set",
                    "metric_scope": "all_six_classes",
                    "test_samples": int(len(test_set)),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "ece": metrics["ece"],
                }
            )
        return rows

    # Reference cells (zero-based): {'ditto': 19}
    def evaluate_owned_stage_rows__ditto(
        self, stage, stage_detail, stage_order, model_specs, test_sets=None
    ):
        if test_sets is None:
            test_sets = self.owned_class_test_set
        rows = []
        for spec in model_specs:
            cid = spec["client_id"]
            model_type = self._ditto_model_type(spec)
            model_id = str(spec.get("model_id", "Global" if model_type == "Global" else cid))
            test_df = test_sets[cid]
            metrics = self.evaluate_owned_specialist(
                spec["weights"], cid, test_df, spec.get("temperature", 1.0)
            )
            rows.append(
                {
                    "stage": stage,
                    "stage_detail": stage_detail,
                    "stage_order": int(stage_order),
                    "evaluated_model_level": model_type,
                    "model_type": model_type,
                    "model_id": model_id,
                    "client_id": cid,
                    "context": self.CLIENT_TO_CONTEXT[cid],
                    "planned_classes": ",".join(self.CLIENT_PLANNED_CLASSES[cid]),
                    "test_set": "owned_class_test_set",
                    "metric_scope": "planned_owned_classes",
                    "test_samples": int(len(test_df)),
                    "owned_test_samples": int(metrics["owned_eval_support"]),
                    "accuracy": metrics["owned_accuracy"],
                    "macro_f1": metrics["owned_macro_f1"],
                    "owned_accuracy": metrics["owned_accuracy"],
                    "owned_macro_f1": metrics["owned_macro_f1"],
                    "owned_macro_recall": metrics["owned_macro_recall"],
                    "owned_min_class_recall": metrics["owned_min_class_recall"],
                }
            )
        return rows

    # Reference cells (zero-based): {'ditto': 19}
    def _ditto_model_type(self, spec):
        explicit = spec.get("model_type")
        if explicit:
            return str(explicit)
        if str(spec.get("model_id", "")) == "Generic":
            return "Global"
        return "Personalized"
