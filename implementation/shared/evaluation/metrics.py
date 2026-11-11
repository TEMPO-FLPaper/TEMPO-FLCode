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

class Operations:
    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def predict_logits_from_weights(self, weights, df, batch_size=32):
        if len(df) == 0:
            return (np.empty((0, self.NUM_CLASSES)), np.array([], dtype=np.int32))
        self.eval_model.set_weights(weights)
        ds = self.make_tf_dataset(df, batch_size=batch_size, training=False, shuffle=False)
        logits = self.eval_model.predict(ds, verbose=0)
        labels = df["label_id"].to_numpy(dtype=np.int32)
        return (logits, labels)

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def expected_calibration_error(self, probabilities, y_true, n_bins=10):
        if len(y_true) == 0:
            return np.nan
        confidence = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        correctness = (predictions == y_true).astype(np.float64)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for left, right in zip(edges[:-1], edges[1:]):
            if right == 1.0:
                mask = (confidence >= left) & (confidence <= right)
            else:
                mask = (confidence >= left) & (confidence < right)
            if mask.sum() == 0:
                continue
            ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
        return float(ece)

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def metrics_from_logits(self, logits, y_true, temperature=1.0):
        if len(y_true) == 0:
            result = {
                "loss": np.nan,
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "balanced_accuracy": np.nan,
                "macro_precision": np.nan,
                "macro_recall": np.nan,
                "ece": np.nan,
            }
            for class_name in self.CLASS_NAMES:
                result[f"recall_{class_name}"] = np.nan
            return result
        scaled = logits / float(temperature)
        probabilities = tf.nn.softmax(scaled, axis=-1).numpy()
        predictions = probabilities.argmax(axis=1)
        loss = (
            tf.keras.losses.sparse_categorical_crossentropy(y_true, scaled, from_logits=True)
            .numpy()
            .mean()
        )
        result = {
            "loss": float(loss),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "macro_precision": float(
                precision_score(y_true, predictions, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_true, predictions, average="macro", zero_division=0)
            ),
            "ece": self.expected_calibration_error(probabilities, y_true),
        }
        class_recall = recall_score(
            y_true, predictions, labels=np.arange(self.NUM_CLASSES), average=None, zero_division=0
        )
        for class_id, class_name in enumerate(self.CLASS_NAMES):
            result[f"recall_{class_name}"] = float(class_recall[class_id])
        return result

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def evaluate_weights(self, weights, df, temperature=1.0):
        logits, labels = self.predict_logits_from_weights(weights, df)
        return self.metrics_from_logits(logits, labels, temperature)

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def predictions_from_weights(self, weights, df, temperature=1.0):
        logits, y_true = self.predict_logits_from_weights(weights, df)
        if len(y_true) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.empty((0, self.NUM_CLASSES), dtype=np.float32),
            )
        probabilities = tf.nn.softmax(logits / float(temperature), axis=-1).numpy()
        y_pred = probabilities.argmax(axis=1).astype(np.int32)
        return (y_true.astype(np.int32), y_pred, probabilities.astype(np.float32))

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def classwise_accuracy_rows(
        self, weights, df, stage, model_id, context="", temperature=1.0, train_reference_df=None
    ):
        y_true, y_pred, _ = self.predictions_from_weights(weights, df, temperature)
        rows = []
        train_counts = (
            train_reference_df["label"].value_counts()
            if train_reference_df is not None and len(train_reference_df) > 0
            else pd.Series(dtype=np.int64)
        )
        for class_id, class_name in enumerate(self.CLASS_NAMES):
            mask = y_true == class_id
            support = int(mask.sum())
            correct = int(np.sum(y_pred[mask] == class_id)) if support > 0 else 0
            class_accuracy = float(correct / support) if support > 0 else np.nan
            train_count = int(train_counts.get(class_name, 0))
            rows.append(
                {
                    "stage": stage,
                    "model_id": model_id,
                    "context": context,
                    "class_name": class_name,
                    "class_id": int(class_id),
                    "train_count_at_node": train_count,
                    "evaluation_support": support,
                    "correct": correct,
                    "class_accuracy_recall": class_accuracy,
                    "class_available_at_node": train_count > 0,
                }
            )
        return rows

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def overall_stage_row(self, weights, df, stage, model_id, context="", temperature=1.0):
        metrics = self.evaluate_weights(weights, df, temperature)
        return {
            "stage": stage,
            "model_id": model_id,
            "context": context,
            "evaluation_samples": len(df),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "ece": metrics["ece"],
            "loss": metrics["loss"],
        }

    # Reference cells (zero-based): {'tempo': 41, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def fixed_label_macro_f1(self, y_true, y_pred):
        return float(
            f1_score(
                y_true, y_pred, labels=np.arange(self.NUM_CLASSES), average="macro", zero_division=0
            )
        )

    # Reference cells (zero-based): {'tempo': 41, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def fixed_label_balanced_accuracy(self, y_true, y_pred):
        recalls = recall_score(
            y_true, y_pred, labels=np.arange(self.NUM_CLASSES), average=None, zero_division=0
        )
        return float(np.mean(recalls))

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def subset_for_classes(self, df, class_names):
        return df[df["label"].isin(list(class_names))].copy().reset_index(drop=True)

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def owned_and_other_dfs(self, cid, df):
        owned_classes = self.CLIENT_PLANNED_CLASSES[cid]
        owned_df = self.subset_for_classes(df, owned_classes)
        other_df = df[~df["label"].isin(owned_classes)].copy().reset_index(drop=True)
        return (owned_df, other_df)

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def macro_recall_on_present_classes(self, y_true, y_pred):
        if len(y_true) == 0:
            return np.nan
        present = np.unique(y_true)
        return float(recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0))

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18}
    def evaluate_owned_specialist(self, weights, cid, df, temperature=1.0):
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

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def macro_f1_on_present_classes(self, y_true, y_pred):
        if len(y_true) == 0:
            return np.nan
        present = np.unique(y_true)
        return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def _stage_metric_row_key(self, row):
        return (
            str(row.get("stage_detail", "")),
            str(row.get("level", row.get("evaluated_model_level", ""))),
            str(row.get("model_id", "")),
            str(row.get("client_id", "")),
            str(row.get("context", "")),
            str(row.get("test_set", "")),
        )

    # Reference cells (zero-based): {'tempo': 20, 'ditto': 18, 'hfl': 18, 'flat': 18}
    def upsert_stage_metric_rows(self, state, state_key, rows):
        new_rows = [dict(row) for row in rows]
        new_keys = {self._stage_metric_row_key(row) for row in new_rows}
        existing_rows = list(state.get(state_key, []))
        state[state_key] = [
            row for row in existing_rows if self._stage_metric_row_key(row) not in new_keys
        ] + new_rows

    # Reference cells (zero-based): {'tempo': 20, 'hfl': 18, 'flat': 18}
    def evaluate_general_stage_rows__tempo_hfl_flat(
        self, stage, stage_detail, stage_order, model_specs, test_set=None
    ):
        if test_set is None:
            test_set = self.general_test_set
        rows = []
        for spec in model_specs:
            metrics = self.evaluate_weights(spec["weights"], test_set, spec.get("temperature", 1.0))
            rows.append(
                {
                    "stage": stage,
                    "stage_detail": stage_detail,
                    "stage_order": int(stage_order),
                    "level": spec["level"],
                    "model_id": spec["model_id"],
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

    # Reference cells (zero-based): {'tempo': 20, 'hfl': 18}
    def evaluate_owned_stage_rows__tempo_hfl(
        self, stage, stage_detail, stage_order, model_specs, test_sets=None
    ):
        if test_sets is None:
            test_sets = self.owned_class_test_set
        rows = []
        for spec in model_specs:
            cid = spec["client_id"]
            test_df = test_sets[cid]
            metrics = self.evaluate_owned_specialist(
                spec["weights"], cid, test_df, spec.get("temperature", 1.0)
            )
            rows.append(
                {
                    "stage": stage,
                    "stage_detail": stage_detail,
                    "stage_order": int(stage_order),
                    "evaluated_model_level": spec.get("evaluated_model_level", "L1"),
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
