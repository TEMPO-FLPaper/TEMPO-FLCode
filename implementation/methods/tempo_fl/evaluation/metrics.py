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
    # Reference cells (zero-based): {'tempo': 20}
    def specialization_selection_score(self, metrics):
        owned_macro = (
            0.0 if np.isnan(metrics["owned_macro_recall"]) else metrics["owned_macro_recall"]
        )
        owned_accuracy = 0.0 if np.isnan(metrics["owned_accuracy"]) else metrics["owned_accuracy"]
        all_f1 = (
            0.0
            if np.isnan(metrics["all_local_macro_f1_secondary"])
            else metrics["all_local_macro_f1_secondary"]
        )
        return float(
            self.CONFIG["specialization_owned_macro_recall_weight"] * owned_macro
            + self.CONFIG["specialization_owned_accuracy_weight"] * owned_accuracy
            + self.CONFIG["specialization_all_local_f1_weight"] * all_f1
        )

    # Reference cells (zero-based): {'tempo': 20}
    def make_specialization_training_df(self, cid, seed):
        local_df = self.client_train[cid].copy().reset_index(drop=True)
        planned_classes = self.CLIENT_PLANNED_CLASSES[cid]
        owned_parts = []
        owned_counts = {
            class_name: int(np.sum(local_df["label"] == class_name))
            for class_name in planned_classes
        }
        if any((count <= 0 for count in owned_counts.values())):
            raise RuntimeError(
                f"V12 specialization cannot train {cid}: planned-class counts={owned_counts}"
            )
        target_per_owned_class = max(owned_counts.values())
        for class_index, class_name in enumerate(planned_classes):
            class_df = local_df[local_df["label"] == class_name].copy()
            balanced_class_df = class_df.sample(
                n=target_per_owned_class,
                replace=len(class_df) < target_per_owned_class,
                random_state=seed + 101 + class_index,
            )
            owned_parts.append(balanced_class_df)
        owned_balanced_df = pd.concat(owned_parts, ignore_index=True)
        other_df = local_df[~local_df["label"].isin(planned_classes)].copy()
        if len(other_df) > 0 and self.CONFIG["specialization_background_fraction"] > 0:
            target_other = min(
                len(other_df),
                max(
                    1,
                    int(
                        round(
                            len(owned_balanced_df)
                            * self.CONFIG["specialization_background_fraction"]
                        )
                    ),
                ),
            )
            other_sample = other_df.sample(n=target_other, random_state=seed + 501)
            training_df = pd.concat([owned_balanced_df, other_sample], ignore_index=True)
        else:
            training_df = owned_balanced_df.copy()
        return training_df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    # Reference cells (zero-based): {'tempo': 20}
    def client_owned_quality_weight(self, cid, metrics):
        planned_train_count = int(
            self.client_train[cid]["label"].isin(self.CLIENT_PLANNED_CLASSES[cid]).sum()
        )
        quality = max(
            float(metrics["owned_macro_recall"])
            if not np.isnan(metrics["owned_macro_recall"])
            else 0.0,
            self.CONFIG["bootstrap_quality_floor"],
        )
        return float(planned_train_count * quality)

    # Reference cells (zero-based): {'tempo': 20}
    def evaluate_owned_specialist_fast(self, weights, cid, df, temperature=1.0):
        if len(df) == 0:
            return {
                "client_id": cid,
                "context": self.CLIENT_TO_CONTEXT[cid],
                "planned_classes": ",".join(self.CLIENT_PLANNED_CLASSES[cid]),
                "owned_eval_support": 0,
                "owned_accuracy": np.nan,
                "owned_macro_recall": np.nan,
                "owned_macro_f1": np.nan,
                "owned_min_class_recall": np.nan,
                "all_local_accuracy_secondary": np.nan,
                "all_local_macro_f1_secondary": np.nan,
                "per_owned_class": {},
            }
        y_true, y_pred, _ = self.predictions_from_weights(weights, df, temperature)
        planned_ids = np.asarray(
            [self.CLASS_TO_ID[class_name] for class_name in self.CLIENT_PLANNED_CLASSES[cid]],
            dtype=np.int32,
        )
        owned_mask = np.isin(y_true, planned_ids)
        y_owned = y_true[owned_mask]
        pred_owned = y_pred[owned_mask]
        owned_accuracy = float(accuracy_score(y_owned, pred_owned)) if len(y_owned) > 0 else np.nan
        owned_macro_recall = self.macro_recall_on_present_classes(y_owned, pred_owned)
        per_owned_class = {}
        class_recalls = []
        for class_name in self.CLIENT_PLANNED_CLASSES[cid]:
            class_id = self.CLASS_TO_ID[class_name]
            class_mask = y_true == class_id
            support = int(class_mask.sum())
            if support > 0:
                recall_value = float(np.mean(y_pred[class_mask] == class_id))
                class_recalls.append(recall_value)
            else:
                recall_value = np.nan
            per_owned_class[class_name] = {"support": support, "recall": recall_value}
        owned_min_class_recall = float(np.min(class_recalls)) if class_recalls else np.nan
        all_local_accuracy = float(accuracy_score(y_true, y_pred))
        all_local_macro_f1 = self.fixed_label_macro_f1(y_true, y_pred)
        return {
            "client_id": cid,
            "context": self.CLIENT_TO_CONTEXT[cid],
            "planned_classes": ",".join(self.CLIENT_PLANNED_CLASSES[cid]),
            "owned_eval_support": int(owned_mask.sum()),
            "owned_accuracy": owned_accuracy,
            "owned_macro_recall": owned_macro_recall,
            "owned_macro_f1": self.macro_f1_on_present_classes(y_owned, pred_owned),
            "owned_min_class_recall": owned_min_class_recall,
            "all_local_accuracy_secondary": all_local_accuracy,
            "all_local_macro_f1_secondary": all_local_macro_f1,
            "per_owned_class": per_owned_class,
        }

    # Reference cells (zero-based): {'tempo': 20}
    def print_fast_specialization_metrics(self, prefix, metrics):
        class_parts = []
        for class_name in self.CLIENT_PLANNED_CLASSES[metrics["client_id"]]:
            details = metrics["per_owned_class"].get(class_name, {})
            recall_value = details.get("recall", np.nan)
            support = details.get("support", 0)
            class_parts.append(
                f"{class_name}={recall_value:.3f}(n={support})"
                if not np.isnan(recall_value)
                else f"{class_name}=NA(n={support})"
            )
        self.log(
            f"{prefix} | OwnedAcc={metrics['owned_accuracy']:.3f} | OwnedMacroRecall={metrics['owned_macro_recall']:.3f} | OwnedMacroF1={metrics['owned_macro_f1']:.3f} | MinOwnedRecall={metrics['owned_min_class_recall']:.3f} | "
            + " | ".join(class_parts)
        )
