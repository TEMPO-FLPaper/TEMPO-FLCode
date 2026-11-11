"""Reference-derived inference/routing. See README.md."""

import math

import numpy as np

import tensorflow as tf

from sklearn.metrics import accuracy_score

class Operations:
    # Reference cells (zero-based): {'tempo': 41}
    def adaptive_teacher_threshold(self, accept_threshold):
        return float(
            np.clip(
                accept_threshold + self.CONFIG["teacher_confidence_margin"],
                self.CONFIG["teacher_threshold_min"],
                self.CONFIG["teacher_threshold_max"],
            )
        )

    # Reference cells (zero-based): {'tempo': 41}
    def predict_logits_and_probabilities(self, weights, df, temperature):
        if len(df) == 0:
            return (
                np.empty((0, self.NUM_CLASSES), dtype=np.float32),
                np.empty((0, self.NUM_CLASSES), dtype=np.float32),
            )
        logits, _ = self.predict_logits_from_weights(weights, df)
        probabilities = tf.nn.softmax(logits / float(temperature), axis=-1).numpy()
        return (logits.astype(np.float32), probabilities.astype(np.float32))

    # Reference cells (zero-based): {'tempo': 41}
    def predict_probabilities(self, weights, df, temperature):
        _, probabilities = self.predict_logits_and_probabilities(weights, df, temperature)
        return probabilities

    # Reference cells (zero-based): {'tempo': 41}
    def cascade_predict_df(self, cid, events, round_id, state, collect_teachers=False):
        context_name = self.CLIENT_TO_CONTEXT[cid]
        l1_weights = state["l1_base_weights"][cid]
        l2_shadow = state["shadow_l2_weights"][cid]
        l3_shadow = state["shadow_l3_weights"][cid]
        tau_l1 = state["l1_threshold"][cid]
        tau_l2 = state["l2_threshold"][context_name]
        tau_l3 = state["l3_threshold"]
        logits1, p1 = self.predict_logits_and_probabilities(
            l1_weights, events, state["l1_temperature"][cid]
        )
        pred1 = p1.argmax(axis=1)
        conf1 = p1.max(axis=1)
        rel1 = self.estimate_correctness_probability(conf1, state["reliability_models"]["L1"][cid])
        l1_accept = conf1 >= tau_l1
        accepted = l1_accept.copy()
        accepted_level = np.where(l1_accept, 1, 0).astype(np.int32)
        used_level = np.ones(len(events), dtype=np.int32)
        forced_pred = pred1.copy()
        forced_conf = conf1.copy()
        forced_reliability = rel1.copy()
        operational_pred = np.full(len(events), -1, dtype=np.int32)
        operational_conf = np.zeros(len(events), dtype=np.float32)
        operational_pred[l1_accept] = pred1[l1_accept]
        operational_conf[l1_accept] = conf1[l1_accept]
        low_l1_indices = np.where(~l1_accept)[0]
        l2_resolved = np.zeros(len(events), dtype=bool)
        l3_resolved = np.zeros(len(events), dtype=bool)
        l2_expected_gain = np.full(len(events), np.nan, dtype=np.float32)
        l3_expected_gain = np.full(len(events), np.nan, dtype=np.float32)
        l2_reliability = np.full(len(events), np.nan, dtype=np.float32)
        l3_reliability = np.full(len(events), np.nan, dtype=np.float32)
        if len(low_l1_indices) > 0:
            l2_df = events.iloc[low_l1_indices].reset_index(drop=True)
            logits2, p2 = self.predict_logits_and_probabilities(
                l2_shadow, l2_df, state["l2_temperature"][context_name]
            )
            pred2 = p2.argmax(axis=1)
            conf2 = p2.max(axis=1)
            rel2 = self.estimate_correctness_probability(
                conf2, state["reliability_models"]["L2"][context_name]
            )
            rel1_local = rel1[low_l1_indices]
            gain2 = rel2 - rel1_local
            l2_reliability[low_l1_indices] = rel2
            l2_expected_gain[low_l1_indices] = gain2
            used_level[low_l1_indices] = 2
            better_forced_l2 = rel2 > forced_reliability[low_l1_indices]
            if np.any(better_forced_l2):
                global_better = low_l1_indices[better_forced_l2]
                forced_pred[global_better] = pred2[better_forced_l2]
                forced_conf[global_better] = conf2[better_forced_l2]
                forced_reliability[global_better] = rel2[better_forced_l2]
            l2_accept = conf2 >= tau_l2
            for local_idx, is_accepted in enumerate(l2_accept):
                global_idx = low_l1_indices[local_idx]
                if not is_accepted:
                    continue
                accepted[global_idx] = True
                accepted_level[global_idx] = 2
                l2_resolved[global_idx] = True
                operational_pred[global_idx] = pred2[local_idx]
                operational_conf[global_idx] = conf2[local_idx]
                if collect_teachers and conf2[local_idx] >= self.adaptive_teacher_threshold(tau_l2):
                    state["teacher_buffers"][cid].append(
                        {
                            "path": str(events.iloc[global_idx]["path"]),
                            "teacher_logits": logits2[local_idx].astype(np.float32),
                            "teacher_level": 2,
                            "estimated_correctness_audit": float(rel2[local_idx]),
                            "estimated_gain_audit": float(gain2[local_idx]),
                            "round": int(round_id),
                        }
                    )
            l3_local_candidates = np.where(~l2_accept)[0]
            l3_indices = low_l1_indices[l3_local_candidates]
            if len(l3_indices) > 0:
                l3_df = events.iloc[l3_indices].reset_index(drop=True)
                logits3, p3 = self.predict_logits_and_probabilities(
                    l3_shadow, l3_df, state["l3_temperature"]
                )
                pred3 = p3.argmax(axis=1)
                conf3 = p3.max(axis=1)
                rel3 = self.estimate_correctness_probability(
                    conf3, state["reliability_models"]["L3"]
                )
                used_level[l3_indices] = 3
                rel1_l3 = rel1[l3_indices]
                rel2_l3 = l2_reliability[l3_indices]
                rel2_l3 = np.where(np.isnan(rel2_l3), rel1_l3, rel2_l3)
                reference_rel = np.maximum(rel1_l3, rel2_l3)
                gain3 = rel3 - reference_rel
                l3_reliability[l3_indices] = rel3
                l3_expected_gain[l3_indices] = gain3
                better_forced_l3 = rel3 > forced_reliability[l3_indices]
                if np.any(better_forced_l3):
                    global_better = l3_indices[better_forced_l3]
                    forced_pred[global_better] = pred3[better_forced_l3]
                    forced_conf[global_better] = conf3[better_forced_l3]
                    forced_reliability[global_better] = rel3[better_forced_l3]
                l3_accept = conf3 >= tau_l3
                for local_idx, is_accepted in enumerate(l3_accept):
                    global_idx = l3_indices[local_idx]
                    if not is_accepted:
                        continue
                    accepted[global_idx] = True
                    accepted_level[global_idx] = 3
                    l3_resolved[global_idx] = True
                    operational_pred[global_idx] = pred3[local_idx]
                    operational_conf[global_idx] = conf3[local_idx]
                    if collect_teachers and conf3[local_idx] >= self.adaptive_teacher_threshold(
                        tau_l3
                    ):
                        state["teacher_buffers"][cid].append(
                            {
                                "path": str(events.iloc[global_idx]["path"]),
                                "teacher_logits": logits3[local_idx].astype(np.float32),
                                "teacher_level": 3,
                                "estimated_correctness_audit": float(rel3[local_idx]),
                                "estimated_gain_audit": float(gain3[local_idx]),
                                "round": int(round_id),
                            }
                        )
        unknown_mask = ~accepted
        l1_to_l2 = int(np.sum(used_level >= 2))
        l2_to_l3 = int(np.sum(used_level >= 3))
        higher_resolved = int(np.sum(l2_resolved | l3_resolved))
        reliance_rate = float(l1_to_l2 / max(len(events), 1))
        higher_resolution_rate = float(higher_resolved / max(l1_to_l2, 1))
        resolved_gain_values = np.concatenate(
            [l2_expected_gain[l2_resolved], l3_expected_gain[l3_resolved]]
        )
        mean_resolved_expected_gain = (
            float(np.nanmean(resolved_gain_values)) if len(resolved_gain_values) > 0 else np.nan
        )
        summary = {
            "events": int(len(events)),
            "l1_to_l2_count": l1_to_l2,
            "l2_to_l3_count": l2_to_l3,
            "higher_resolved_count": higher_resolved,
            "l1_shadow_reliance_rate": reliance_rate,
            "higher_resolution_rate": higher_resolution_rate,
            "unknown_rate": float(unknown_mask.mean()),
            "average_inference_depth": float(used_level.mean()),
            "l1_threshold_used": tau_l1,
            "l2_threshold_used": tau_l2,
            "l3_threshold_used": tau_l3,
            "l2_teacher_threshold_used": self.adaptive_teacher_threshold(tau_l2),
            "l3_teacher_threshold_used": self.adaptive_teacher_threshold(tau_l3),
            "mean_resolved_expected_gain_audit": mean_resolved_expected_gain,
            "mean_resolved_expected_gain": mean_resolved_expected_gain,
        }
        return {
            "summary": summary,
            "forced_pred": forced_pred,
            "forced_conf": forced_conf,
            "forced_reliability": forced_reliability,
            "final_pred": forced_pred,
            "final_conf": forced_conf,
            "operational_pred": operational_pred,
            "operational_conf": operational_conf,
            "used_level": used_level,
            "accepted_level": accepted_level,
            "unknown_mask": unknown_mask,
            "l1_confidence": conf1,
            "l1_reliability": rel1,
            "l2_reliability": l2_reliability,
            "l3_reliability": l3_reliability,
            "l2_expected_gain": l2_expected_gain,
            "l3_expected_gain": l3_expected_gain,
        }

    # Reference cells (zero-based): {'tempo': 41}
    def operational_cascade_for_client(self, cid, round_id, state):
        events = self.sample_operational_events(cid, round_id)
        result = self.cascade_predict_df(cid, events, round_id, state, collect_teachers=True)
        summary = {
            "round": round_id,
            "scenario": self.scenario_for_round(round_id),
            "client_id": cid,
            "context": self.CLIENT_TO_CONTEXT[cid],
            **result["summary"],
            "teacher_buffer_size": len(state["teacher_buffers"][cid]),
        }
        event_rows = []
        for i, row in events.iterrows():
            is_unknown = bool(result["unknown_mask"][i])
            event_rows.append(
                {
                    "round": round_id,
                    "scenario": self.scenario_for_round(round_id),
                    "client_id": cid,
                    "context": self.CLIENT_TO_CONTEXT[cid],
                    "event_index": i,
                    "path": str(row["path"]),
                    "stream_source": row.get("stream_source", "unknown"),
                    "true_label_eval_only": row["label"],
                    "operational_prediction": "UNKNOWN"
                    if is_unknown
                    else self.CLASS_NAMES[int(result["operational_pred"][i])],
                    "forced_prediction_eval_only": self.CLASS_NAMES[int(result["forced_pred"][i])],
                    "forced_confidence": float(result["forced_conf"][i]),
                    "used_level": int(result["used_level"][i]),
                    "accepted_level": int(result["accepted_level"][i]),
                    "l1_confidence": float(result["l1_confidence"][i]),
                    "l1_reliability": float(result["l1_reliability"][i]),
                    "l2_expected_gain": float(result["l2_expected_gain"][i])
                    if not np.isnan(result["l2_expected_gain"][i])
                    else np.nan,
                    "l3_expected_gain": float(result["l3_expected_gain"][i])
                    if not np.isnan(result["l3_expected_gain"][i])
                    else np.nan,
                    "unknown": is_unknown,
                }
            )
        return (summary, event_rows)

    # Reference cells (zero-based): {'tempo': 41}
    def evaluate_operational_cascade(self, cid, round_id, state):
        evaluation_df = self.sample_operational_evaluation(cid, round_id)
        result = self.cascade_predict_df(
            cid, evaluation_df, round_id, state, collect_teachers=False
        )
        y_true = evaluation_df["label_id"].to_numpy(dtype=np.int32)
        l1_logits, _ = self.predict_logits_from_weights(
            state["l1_base_weights"][cid], evaluation_df
        )
        l1_prob = tf.nn.softmax(l1_logits / float(state["l1_temperature"][cid]), axis=-1).numpy()
        l1_pred = l1_prob.argmax(axis=1)
        l1_accuracy = float(accuracy_score(y_true, l1_pred))
        l1_macro_f1 = self.fixed_label_macro_f1(y_true, l1_pred)
        l1_balanced_accuracy = self.fixed_label_balanced_accuracy(y_true, l1_pred)
        forced_pred = result["forced_pred"]
        forced_accuracy = float(accuracy_score(y_true, forced_pred))
        forced_macro_f1 = self.fixed_label_macro_f1(y_true, forced_pred)
        forced_balanced_accuracy = self.fixed_label_balanced_accuracy(y_true, forced_pred)
        operational_pred = result["operational_pred"]
        operational_accuracy = float(accuracy_score(y_true, operational_pred))
        operational_macro_f1 = self.fixed_label_macro_f1(y_true, operational_pred)
        operational_balanced_accuracy = self.fixed_label_balanced_accuracy(y_true, operational_pred)
        accepted_mask = ~result["unknown_mask"]
        coverage = float(accepted_mask.mean())
        escalated_mask = result["used_level"] >= 2
        higher_accepted_mask = result["accepted_level"] >= 2
        higher_correct_mask = higher_accepted_mask & (operational_pred == y_true)
        true_higher_resolution_rate = float(
            higher_correct_mask.sum() / max(escalated_mask.sum(), 1)
        )
        higher_accept_correctness = (
            float(np.mean(operational_pred[higher_accepted_mask] == y_true[higher_accepted_mask]))
            if higher_accepted_mask.sum() > 0
            else np.nan
        )
        beneficial_override_mask = higher_accepted_mask & (operational_pred == y_true) & l1_pred
        harmful_override_mask = higher_accepted_mask & operational_pred & (l1_pred == y_true)
        higher_override_count = int(higher_accepted_mask.sum())
        beneficial_override_count = int(beneficial_override_mask.sum())
        harmful_override_count = int(harmful_override_mask.sum())
        harmful_override_rate = float(harmful_override_count / max(higher_override_count, 1))
        override_net_correctness_gain = float(
            (beneficial_override_count - harmful_override_count) / max(len(y_true), 1)
        )
        if accepted_mask.sum() > 0:
            selective_accuracy = float(
                accuracy_score(y_true[accepted_mask], operational_pred[accepted_mask])
            )
            selective_macro_f1 = self.fixed_label_macro_f1(
                y_true[accepted_mask], operational_pred[accepted_mask]
            )
        else:
            selective_accuracy = np.nan
            selective_macro_f1 = np.nan
        return {
            "operational_eval_samples": len(evaluation_df),
            "l1_only_operational_accuracy": l1_accuracy,
            "l1_only_operational_macro_f1": l1_macro_f1,
            "l1_only_operational_balanced_accuracy": l1_balanced_accuracy,
            "operational_accuracy": operational_accuracy,
            "operational_macro_f1": operational_macro_f1,
            "operational_balanced_accuracy": operational_balanced_accuracy,
            "operational_coverage": coverage,
            "operational_selective_accuracy": selective_accuracy,
            "operational_selective_macro_f1": selective_macro_f1,
            "forced_cascade_accuracy": forced_accuracy,
            "forced_cascade_macro_f1": forced_macro_f1,
            "forced_cascade_balanced_accuracy": forced_balanced_accuracy,
            "cascade_accuracy_gain_over_l1": operational_accuracy - l1_accuracy,
            "cascade_macro_f1_gain_over_l1": operational_macro_f1 - l1_macro_f1,
            "forced_cascade_accuracy_gain_over_l1": forced_accuracy - l1_accuracy,
            "forced_cascade_macro_f1_gain_over_l1": forced_macro_f1 - l1_macro_f1,
            "operational_eval_reliance_rate": float(result["summary"]["l1_shadow_reliance_rate"]),
            "operational_eval_higher_resolution_rate": float(
                result["summary"]["higher_resolution_rate"]
            ),
            "operational_eval_true_higher_resolution_rate": true_higher_resolution_rate,
            "operational_eval_higher_accept_correctness": higher_accept_correctness,
            "operational_eval_higher_override_count": higher_override_count,
            "operational_eval_beneficial_override_count": beneficial_override_count,
            "operational_eval_harmful_override_count": harmful_override_count,
            "operational_eval_harmful_override_rate": harmful_override_rate,
            "operational_eval_override_net_correctness_gain": override_net_correctness_gain,
            "operational_eval_unknown_rate": float(result["unknown_mask"].mean()),
            "operational_eval_mean_depth": float(result["used_level"].mean()),
            "operational_eval_mean_resolved_expected_gain": result["summary"][
                "mean_resolved_expected_gain_audit"
            ],
        }

    # Reference cells (zero-based): {'tempo': 41}
    def update_v12_2_operational_counters(self, cid, event_rows, state):
        counters = state["client_operational_counters"][cid]
        total = len(event_rows)
        specialist_accepted = 0
        specialist_correct = 0
        shadow_invoked = 0
        shadow_replaced = 0
        shadow_correct = 0
        l2_replaced = 0
        l2_correct = 0
        l3_replaced = 0
        l3_correct = 0
        unknown = 0
        for row in event_rows:
            accepted_level = int(row["accepted_level"])
            used_level = int(row["used_level"])
            true_label = str(row["true_label_eval_only"])
            prediction = str(row["operational_prediction"])
            is_correct = prediction == true_label
            if accepted_level == 1:
                specialist_accepted += 1
                if is_correct:
                    specialist_correct += 1
            if used_level >= 2:
                shadow_invoked += 1
            if accepted_level >= 2:
                shadow_replaced += 1
                if is_correct:
                    shadow_correct += 1
            if accepted_level == 2:
                l2_replaced += 1
                if is_correct:
                    l2_correct += 1
            if accepted_level == 3:
                l3_replaced += 1
                if is_correct:
                    l3_correct += 1
            if accepted_level == 0:
                unknown += 1
        counters["total_events"] += total
        counters["specialist_accepted_events"] += specialist_accepted
        counters["specialist_correct_events_audit"] += specialist_correct
        counters["shadow_invocations"] += shadow_invoked
        counters["shadow_replacements"] += shadow_replaced
        counters["shadow_correct_replacements_audit"] += shadow_correct
        counters["l2_replacements"] += l2_replaced
        counters["l2_correct_replacements_audit"] += l2_correct
        counters["l3_replacements"] += l3_replaced
        counters["l3_correct_replacements_audit"] += l3_correct
        counters["unknown_events"] += unknown
        if shadow_invoked == 0:
            counters["rounds_specialist_only"] += 1
        if specialist_accepted >= math.ceil(total / 2):
            counters["rounds_specialist_dominant"] += 1
        if shadow_invoked > 0:
            counters["rounds_shadow_invoked"] += 1
        if shadow_replaced > 0:
            counters["rounds_shadow_replaced"] += 1
        if shadow_correct > 0:
            counters["rounds_shadow_effective_audit"] += 1
        return {
            "round_total_events": total,
            "round_specialist_accepted": specialist_accepted,
            "round_specialist_correct_audit": specialist_correct,
            "round_shadow_invocations": shadow_invoked,
            "round_shadow_replacements": shadow_replaced,
            "round_shadow_correct_audit": shadow_correct,
            "round_l2_replacements": l2_replaced,
            "round_l3_replacements": l3_replaced,
            "round_unknown_events": unknown,
        }
