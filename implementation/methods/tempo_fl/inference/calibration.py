"""Reference-derived inference/calibration. See README.md."""

import numpy as np

import pandas as pd

import tensorflow as tf

from scipy.optimize import minimize_scalar

class Operations:
    # Reference cells (zero-based): {'tempo': 27}
    def fit_temperature(self, weights, df):
        logits, labels = self.predict_logits_from_weights(weights, df)
        if len(labels) == 0:
            return 1.0
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)

        def nll(temperature):
            temperature = float(max(temperature, 0.001))
            scaled = logits / temperature
            loss = (
                tf.keras.losses.sparse_categorical_crossentropy(labels, scaled, from_logits=True)
                .numpy()
                .mean()
            )
            return float(loss)

        result = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
        return float(result.x)

    # Reference cells (zero-based): {'tempo': 27}
    def wilson_lower_bound(self, successes, total, z=1.0):
        if total <= 0:
            return np.nan
        p = float(successes) / float(total)
        denominator = 1.0 + z * z / float(total)
        centre = p + z * z / (2.0 * float(total))
        margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * float(total))) / float(total))
        return float((centre - margin) / denominator)

    # Reference cells (zero-based): {'tempo': 27}
    def threshold_floor_for_level(self, level):
        if level == "L1":
            return float(self.CONFIG["l1_threshold_clip_min"])
        if level == "L2":
            return float(self.CONFIG["l2_threshold_clip_min"])
        return float(self.CONFIG["l3_threshold_clip_min"])

    # Reference cells (zero-based): {'tempo': 27}
    def derive_accept_threshold(
        self, weights, df, temperature, level, target_accuracy=None, min_coverage=None
    ):
        if target_accuracy is None:
            target_accuracy = self.CONFIG["target_selective_accuracy"]
        if min_coverage is None:
            min_coverage = self.CONFIG["min_accept_coverage"]
        logits, labels = self.predict_logits_from_weights(weights, df)
        floor = self.threshold_floor_for_level(level)
        if len(labels) == 0:
            return (
                floor,
                {
                    "threshold": floor,
                    "coverage": np.nan,
                    "selective_accuracy": np.nan,
                    "selective_accuracy_lower_bound": np.nan,
                    "overall_accuracy": np.nan,
                    "selection_rule": "empty_calibration",
                },
            )
        probabilities = tf.nn.softmax(logits / float(temperature), axis=-1).numpy()
        confidence = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        correct = predictions == labels
        grid = np.arange(
            max(self.CONFIG["threshold_grid_min"], floor),
            self.CONFIG["threshold_grid_max"] + 1e-09,
            self.CONFIG["threshold_grid_step"],
        )
        candidates = []
        for threshold in grid:
            accepted = confidence >= threshold
            accepted_count = int(accepted.sum())
            coverage = float(accepted.mean())
            if accepted_count == 0:
                selective_accuracy = np.nan
                lower_bound = np.nan
            else:
                correct_count = int(correct[accepted].sum())
                selective_accuracy = float(correct_count / accepted_count)
                lower_bound = self.wilson_lower_bound(correct_count, accepted_count, z=1.0)
            candidates.append(
                {
                    "threshold": float(threshold),
                    "coverage": coverage,
                    "accepted_count": accepted_count,
                    "selective_accuracy": selective_accuracy,
                    "selective_accuracy_lower_bound": lower_bound,
                }
            )
        feasible = [
            row
            for row in candidates
            if row["coverage"] >= min_coverage
            and (not np.isnan(row["selective_accuracy"]))
            and (row["selective_accuracy"] >= target_accuracy)
            and (
                row["selective_accuracy_lower_bound"] >= self.CONFIG["target_selective_lower_bound"]
            )
        ]
        if feasible:
            best = sorted(
                feasible,
                key=lambda row: (
                    row["coverage"],
                    row["selective_accuracy"],
                    row["selective_accuracy_lower_bound"],
                    -row["threshold"],
                ),
                reverse=True,
            )[0]
            selection_rule = "max_coverage_under_reliability_constraints"
        else:
            coverage_feasible = [
                row
                for row in candidates
                if row["coverage"] >= min_coverage and (not np.isnan(row["selective_accuracy"]))
            ]
            if coverage_feasible:
                best = sorted(
                    coverage_feasible,
                    key=lambda row: (
                        0.65 * row["selective_accuracy"]
                        + 0.2
                        * (
                            0.0
                            if np.isnan(row["selective_accuracy_lower_bound"])
                            else row["selective_accuracy_lower_bound"]
                        )
                        + 0.15 * row["coverage"],
                        row["coverage"],
                    ),
                    reverse=True,
                )[0]
                selection_rule = "best_accuracy_coverage_tradeoff"
            else:
                raw_threshold = float(np.quantile(confidence, max(0.0, 1.0 - min_coverage)))
                raw_threshold = max(raw_threshold, floor)
                accepted = confidence >= raw_threshold
                accepted_count = int(accepted.sum())
                correct_count = int(correct[accepted].sum())
                best = {
                    "threshold": raw_threshold,
                    "coverage": float(accepted.mean()),
                    "accepted_count": accepted_count,
                    "selective_accuracy": float(correct_count / max(accepted_count, 1)),
                    "selective_accuracy_lower_bound": self.wilson_lower_bound(
                        correct_count, accepted_count, z=1.0
                    ),
                }
                selection_rule = "coverage_quantile_fallback"
        clipped_threshold = float(
            np.clip(best["threshold"], floor, self.CONFIG["threshold_clip_max"])
        )
        accepted = confidence >= clipped_threshold
        accepted_count = int(accepted.sum())
        if accepted_count > 0:
            correct_count = int(correct[accepted].sum())
            final_selective_accuracy = float(correct_count / accepted_count)
            final_lower_bound = self.wilson_lower_bound(correct_count, accepted_count, z=1.0)
        else:
            final_selective_accuracy = np.nan
            final_lower_bound = np.nan
        diagnostics = {
            "threshold": clipped_threshold,
            "coverage": float(accepted.mean()),
            "accepted_count": accepted_count,
            "selective_accuracy": final_selective_accuracy,
            "selective_accuracy_lower_bound": final_lower_bound,
            "overall_accuracy": float(correct.mean()),
            "target_selective_accuracy": target_accuracy,
            "target_selective_lower_bound": self.CONFIG["target_selective_lower_bound"],
            "minimum_coverage": min_coverage,
            "selection_rule": selection_rule,
        }
        return (clipped_threshold, diagnostics)

    # Reference cells (zero-based): {'tempo': 27}
    def fit_reliability_table(self, weights, df, temperature, level, model_id, round_id):
        logits, labels = self.predict_logits_from_weights(weights, df)
        edges = np.asarray(self.CONFIG["reliability_bin_edges"], dtype=np.float64)
        if len(labels) == 0:
            return {
                "level": level,
                "model_id": model_id,
                "round": round_id,
                "overall_accuracy": 0.5,
                "edges": edges.tolist(),
                "bins": [],
            }
        probabilities = tf.nn.softmax(logits / float(temperature), axis=-1).numpy()
        confidence = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        correct = (predictions == labels).astype(np.float64)
        overall_accuracy = float(correct.mean())
        prior_strength = float(self.CONFIG["reliability_prior_strength"])
        bins = []
        for bin_id in range(len(edges) - 1):
            left = float(edges[bin_id])
            right = float(edges[bin_id + 1])
            if bin_id == len(edges) - 2:
                mask = (confidence >= left) & (confidence <= right)
            else:
                mask = (confidence >= left) & (confidence < right)
            n = int(mask.sum())
            successes = float(correct[mask].sum())
            posterior_accuracy = float(
                (successes + prior_strength * overall_accuracy) / (n + prior_strength)
            )
            bins.append(
                {
                    "bin_id": bin_id,
                    "left": left,
                    "right": right,
                    "samples": n,
                    "correct": int(successes),
                    "posterior_accuracy": posterior_accuracy,
                }
            )
        return {
            "level": level,
            "model_id": model_id,
            "round": round_id,
            "overall_accuracy": overall_accuracy,
            "edges": edges.tolist(),
            "bins": bins,
        }

    # Reference cells (zero-based): {'tempo': 27}
    def estimate_correctness_probability(self, confidence, reliability_table):
        confidence = np.asarray(confidence, dtype=np.float64)
        result = np.full(
            confidence.shape,
            float(reliability_table.get("overall_accuracy", 0.5)),
            dtype=np.float64,
        )
        for row in reliability_table.get("bins", []):
            left = float(row["left"])
            right = float(row["right"])
            if row["bin_id"] == len(reliability_table["bins"]) - 1:
                mask = (confidence >= left) & (confidence <= right)
            else:
                mask = (confidence >= left) & (confidence < right)
            result[mask] = float(row["posterior_accuracy"])
        return result.astype(np.float32)

    # Reference cells (zero-based): {'tempo': 27}
    def refresh_reliability_models(self, state, round_id):
        rows = []
        state["reliability_models"] = {"L1": {}, "L2": {}, "L3": None}
        for cid in self.CLIENT_IDS:
            table = self.fit_reliability_table(
                state["l1_base_weights"][cid],
                self.client_calibration_local[cid],
                state["l1_temperature"][cid],
                "L1",
                cid,
                round_id,
            )
            state["reliability_models"]["L1"][cid] = table
            for bin_row in table["bins"]:
                rows.append(
                    {
                        "round": round_id,
                        "level": "L1",
                        "model_id": cid,
                        "overall_accuracy": table["overall_accuracy"],
                        **bin_row,
                    }
                )
        for context_name in self.CONFIG["context_names"]:
            context_calibration_df = pd.concat(
                [
                    self.client_calibration_local[cid]
                    for cid in self.CONTEXT_TO_CLIENTS[context_name]
                    if len(self.client_calibration_local[cid]) > 0
                ],
                ignore_index=True,
            )
            table = self.fit_reliability_table(
                state["l2_base_weights"][context_name],
                context_calibration_df,
                state["l2_temperature"][context_name],
                "L2",
                context_name,
                round_id,
            )
            state["reliability_models"]["L2"][context_name] = table
            for bin_row in table["bins"]:
                rows.append(
                    {
                        "round": round_id,
                        "level": "L2",
                        "model_id": context_name,
                        "overall_accuracy": table["overall_accuracy"],
                        **bin_row,
                    }
                )
        l3_table = self.fit_reliability_table(
            state["l3_base_weights"],
            self.global_calibration_df,
            state["l3_temperature"],
            "L3",
            "Campus",
            round_id,
        )
        state["reliability_models"]["L3"] = l3_table
        for bin_row in l3_table["bins"]:
            rows.append(
                {
                    "round": round_id,
                    "level": "L3",
                    "model_id": "Campus",
                    "overall_accuracy": l3_table["overall_accuracy"],
                    **bin_row,
                }
            )
        state["reliability_history"].extend(rows)
        return rows

    # Reference cells (zero-based): {'tempo': 27}
    def calibrate_and_gate(self, weights, df, level, model_id, round_id):
        temperature = self.fit_temperature(weights, df)
        threshold, diagnostics = self.derive_accept_threshold(weights, df, temperature, level=level)
        record = {
            "round": round_id,
            "level": level,
            "model_id": model_id,
            "temperature": temperature,
            **diagnostics,
        }
        return (temperature, threshold, record)

    # Reference cells (zero-based): {'tempo': 27}
    def refresh_all_gates(self, state, round_id):
        records = []
        for cid in self.CLIENT_IDS:
            state["l1_temperature"][cid], state["l1_threshold"][cid], record = (
                self.calibrate_and_gate(
                    state["l1_base_weights"][cid],
                    self.client_calibration_local[cid],
                    "L1",
                    cid,
                    round_id,
                )
            )
            records.append(record)
        for context_name in self.CONFIG["context_names"]:
            context_calibration_df = pd.concat(
                [
                    self.client_calibration_local[cid]
                    for cid in self.CONTEXT_TO_CLIENTS[context_name]
                    if len(self.client_calibration_local[cid]) > 0
                ],
                ignore_index=True,
            )
            state["l2_temperature"][context_name], state["l2_threshold"][context_name], record = (
                self.calibrate_and_gate(
                    state["l2_base_weights"][context_name],
                    context_calibration_df,
                    "L2",
                    context_name,
                    round_id,
                )
            )
            records.append(record)
        state["l3_temperature"], state["l3_threshold"], record = self.calibrate_and_gate(
            state["l3_base_weights"], self.global_calibration_df, "L3", "Campus", round_id
        )
        records.append(record)
        state["threshold_diagnostics"].extend(records)
        reliability_rows = self.refresh_reliability_models(state, round_id)
        self.log("\nAdaptive confidence gates + empirical reliability:")
        for record in records:
            self.log(
                f"  {record['level']:2s} {record['model_id']:12s} | T={record['temperature']:.3f} | tau={record['threshold']:.3f} | coverage={record['coverage']:.3f} | selectiveAcc={record['selective_accuracy']:.3f} | LB={record['selective_accuracy_lower_bound']:.3f}"
            )
        return (records, reliability_rows)
