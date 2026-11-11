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
    # Reference cells (zero-based): {'flat': 18}
    def evaluate_owned_model(self, weights, cid, df, temperature=1.0):
        owned_df, other_df = self.owned_and_other_dfs(cid, df)
        all_metrics = self.evaluate_weights(weights, df, temperature) if len(df) > 0 else None
        y_owned, pred_owned, _ = self.predictions_from_weights(weights, owned_df, temperature)
        owned_accuracy = float(accuracy_score(y_owned, pred_owned)) if len(y_owned) > 0 else np.nan
        owned_macro_recall = self.macro_recall_on_present_classes(y_owned, pred_owned)
        owned_class_rows = self.classwise_accuracy_rows(
            weights,
            owned_df,
            "owned_eval",
            cid,
            self.CLIENT_TO_CONTEXT[cid],
            temperature=temperature,
            train_reference_df=self.client_train[cid],
        )
        owned_class_rows = [
            row for row in owned_class_rows if row["class_name"] in self.CLIENT_PLANNED_CLASSES[cid]
        ]
        supported_owned_recalls = [
            row["class_accuracy_recall"]
            for row in owned_class_rows
            if row["evaluation_support"] > 0 and (not np.isnan(row["class_accuracy_recall"]))
        ]
        owned_min_recall = (
            float(np.min(supported_owned_recalls)) if supported_owned_recalls else np.nan
        )
        y_other, pred_other, _ = self.predictions_from_weights(weights, other_df, temperature)
        other_accuracy = float(accuracy_score(y_other, pred_other)) if len(y_other) > 0 else np.nan
        other_macro_recall = self.macro_recall_on_present_classes(y_other, pred_other)
        return {
            "client_id": cid,
            "context": self.CLIENT_TO_CONTEXT[cid],
            "planned_classes": ",".join(self.CLIENT_PLANNED_CLASSES[cid]),
            "owned_eval_support": len(owned_df),
            "owned_accuracy": owned_accuracy,
            "owned_macro_recall": owned_macro_recall,
            "owned_macro_f1": self.macro_f1_on_present_classes(y_owned, pred_owned),
            "owned_min_class_recall": owned_min_recall,
            "other_eval_support": len(other_df),
            "other_accuracy_secondary": other_accuracy,
            "other_macro_recall_secondary": other_macro_recall,
            "all_local_accuracy_secondary": all_metrics["accuracy"] if all_metrics else np.nan,
            "all_local_macro_f1_secondary": all_metrics["macro_f1"] if all_metrics else np.nan,
            "owned_class_rows": owned_class_rows,
        }

    # Reference cells (zero-based): {'flat': 18}
    def evaluate_owned_stage_rows__flat(
        self, stage, stage_detail, stage_order, model_specs, test_sets=None
    ):
        if test_sets is None:
            test_sets = self.owned_class_test_set
        rows = []
        for spec in model_specs:
            cid = spec["client_id"]
            test_df = test_sets[cid]
            metrics = self.evaluate_owned_model(
                spec["weights"], cid, test_df, spec.get("temperature", 1.0)
            )
            rows.append(
                {
                    "stage": stage,
                    "stage_detail": stage_detail,
                    "stage_order": int(stage_order),
                    "evaluated_model_level": spec.get("evaluated_model_level", "Client"),
                    "model_id": spec.get("model_id", cid),
                    "client_id": cid,
                    "context": self.CLIENT_TO_CONTEXT[cid],
                    "planned_classes": ",".join(self.CLIENT_PLANNED_CLASSES[cid]),
                    "test_set": "owned_class_test_set",
                    "metric_scope": "planned_owned_classes",
                    "test_samples": int(len(test_df)),
                    "owned_test_samples": int(metrics["owned_eval_support"]),
                    "owned_accuracy": metrics["owned_accuracy"],
                    "owned_macro_f1": metrics["owned_macro_f1"],
                    "owned_macro_recall": metrics["owned_macro_recall"],
                    "owned_min_class_recall": metrics["owned_min_class_recall"],
                }
            )
        return rows
