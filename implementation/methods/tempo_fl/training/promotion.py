"""Reference-derived training/promotion. See README.md."""

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

class Operations:
    # Reference cells (zero-based): {'tempo': 28}
    def required_promotion_reliance(self, baseline):
        return max(
            self.CONFIG["absolute_reliance_floor"],
            float(baseline) + self.CONFIG["reliance_increase_over_baseline"],
        )

    # Reference cells (zero-based): {'tempo': 28}
    def required_hard_fraction_for_reliance(self, baseline, required_reliance):
        denominator = max(1.0 - float(baseline), 1e-08)
        return float(np.clip((float(required_reliance) - float(baseline)) / denominator, 0.0, 1.0))

    # Reference cells (zero-based): {'tempo': 45}
    def train_promotion_candidate(self, base_weights, teacher_items, replay_df):
        candidate = self.build_tempo_model(self.NUM_CLASSES)
        candidate.set_weights(base_weights)
        optimizer = keras.optimizers.Adam(
            self.CONFIG["promotion_lr"], clipnorm=self.CONFIG["gradient_clip_norm"]
        )
        sparse_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        teacher_paths = [item["path"] for item in teacher_items]
        teacher_logits_all = (
            np.stack([item["teacher_logits"] for item in teacher_items]).astype(np.float32)
            if teacher_items
            else None
        )
        replay_paths = replay_df["path"].astype(str).tolist()
        replay_labels = replay_df["label_id"].astype(np.int32).to_numpy()
        batch_size = 16
        temperature = float(self.CONFIG["promotion_kd_temperature"])
        for _ in range(self.CONFIG["promotion_epochs"]):
            for start in range(0, len(replay_paths), batch_size):
                selected = np.arange(start, min(start + batch_size, len(replay_paths)))
                mini_df = pd.DataFrame(
                    {
                        "path": [replay_paths[i] for i in selected],
                        "label_id": [replay_labels[i] for i in selected],
                    }
                )
                x_batch, y_batch = self.load_batch_tensor(mini_df, training=True)
                with tf.GradientTape() as tape:
                    logits = candidate(x_batch, training=True)
                    replay_loss = sparse_ce(y_batch, logits)
                gradients = tape.gradient(replay_loss, candidate.trainable_variables)
                optimizer.apply_gradients(
                    (
                        (g, v)
                        for g, v in zip(gradients, candidate.trainable_variables)
                        if g is not None
                    )
                )
            if teacher_items:
                for start in range(0, len(teacher_paths), batch_size):
                    selected = np.arange(start, min(start + batch_size, len(teacher_paths)))
                    mini_df = pd.DataFrame(
                        {
                            "path": [teacher_paths[i] for i in selected],
                            "label_id": [0 for _ in selected],
                        }
                    )
                    x_batch, _ = self.load_batch_tensor(mini_df, training=True)
                    teacher_logits = tf.convert_to_tensor(
                        teacher_logits_all[selected], dtype=tf.float32
                    )
                    with tf.GradientTape() as tape:
                        student_logits = candidate(x_batch, training=True)
                        teacher_prob_t = tf.nn.softmax(teacher_logits / temperature, axis=-1)
                        student_log_prob_t = tf.nn.log_softmax(
                            student_logits / temperature, axis=-1
                        )
                        kd_loss = (
                            tf.reduce_mean(
                                tf.reduce_sum(
                                    teacher_prob_t
                                    * (
                                        tf.math.log(tf.clip_by_value(teacher_prob_t, 1e-08, 1.0))
                                        - student_log_prob_t
                                    ),
                                    axis=-1,
                                )
                            )
                            * temperature**2
                        )
                        total_loss = self.CONFIG["promotion_kd_lambda"] * kd_loss
                    gradients = tape.gradient(total_loss, candidate.trainable_variables)
                    optimizer.apply_gradients(
                        (
                            (g, v)
                            for g, v in zip(gradients, candidate.trainable_variables)
                            if g is not None
                        )
                    )
        return [w.copy() for w in candidate.get_weights()]

    # Reference cells (zero-based): {'tempo': 46}
    def teacher_kl_loss(self, weights, teacher_items, temperature=None):
        if not teacher_items:
            return np.nan
        if temperature is None:
            temperature = self.CONFIG["promotion_kd_temperature"]
        df = pd.DataFrame(
            {
                "path": [item["path"] for item in teacher_items],
                "label_id": [0 for _ in teacher_items],
            }
        )
        student_logits, _ = self.predict_logits_from_weights(weights, df)
        teacher_logits = np.stack([item["teacher_logits"] for item in teacher_items]).astype(
            np.float64
        )
        t = float(temperature)
        teacher_prob = tf.nn.softmax(teacher_logits / t, axis=-1).numpy()
        student_prob = tf.nn.softmax(student_logits / t, axis=-1).numpy()
        teacher_prob = np.clip(teacher_prob, 1e-08, 1.0)
        student_prob = np.clip(student_prob, 1e-08, 1.0)
        kl = np.sum(teacher_prob * (np.log(teacher_prob) - np.log(student_prob)), axis=1)
        return float(np.mean(kl))

    # Reference cells (zero-based): {'tempo': 46}
    def teacher_sample_acceptance_rate(self, weights, teacher_items, l1_temperature, l1_threshold):
        if not teacher_items:
            return np.nan
        df = pd.DataFrame(
            {
                "path": [item["path"] for item in teacher_items],
                "label_id": [0 for _ in teacher_items],
            }
        )
        logits, _ = self.predict_logits_from_weights(weights, df)
        probabilities = tf.nn.softmax(logits / float(l1_temperature), axis=-1).numpy()
        confidence = probabilities.max(axis=1)
        return float(np.mean(confidence >= float(l1_threshold)))

    # Reference cells (zero-based): {'tempo': 46}
    def maybe_promote_l1(
        self,
        cid,
        round_id,
        reliance_rate,
        resolution_rate,
        total_event_count,
        escalated_count,
        resolved_count,
        state,
    ):
        result = {
            "baseline_reliance": state["baseline_reliance"][cid],
            "required_reliance_for_promotion": np.nan,
            "rebaseline_active": state["rebaseline_active"][cid],
            "rebaseline_completed": False,
            "promotion_triggered": False,
            "promotion_accepted": False,
            "shift_eligible_client": np.nan,
            "control_client_block": False,
            "promotion_pre_local_f1": np.nan,
            "promotion_post_local_f1": np.nan,
            "promotion_delta_local_f1": np.nan,
            "promotion_pre_local_accuracy": np.nan,
            "promotion_post_local_accuracy": np.nan,
            "promotion_delta_local_accuracy": np.nan,
            "promotion_pre_owned_accuracy": np.nan,
            "promotion_post_owned_accuracy": np.nan,
            "promotion_delta_owned_accuracy": np.nan,
            "promotion_pre_owned_macro_recall": np.nan,
            "promotion_post_owned_macro_recall": np.nan,
            "promotion_delta_owned_macro_recall": np.nan,
            "promotion_owned_retention_ok": False,
            "teacher_kl_before": np.nan,
            "teacher_kl_after": np.nan,
            "teacher_kl_improvement": np.nan,
            "teacher_acceptance_before": np.nan,
            "teacher_acceptance_after": np.nan,
            "teacher_acceptance_gain": np.nan,
            "promotion_accept_cooldown_active": False,
            "promotion_reject_cooldown_active": False,
            "rounds_since_last_promotion": round_id - state["last_promotion_round"][cid],
            "rounds_since_last_attempt": round_id - state["last_promotion_attempt_round"][cid],
            "recent_teacher_samples": 0,
        }
        if round_id <= self.CONFIG["commissioning_rounds"]:
            state["baseline_reliance_samples"][cid].append(float(reliance_rate))
            state["baseline_reliance"][cid] = float(
                np.mean(state["baseline_reliance_samples"][cid])
            )
            result["baseline_reliance"] = state["baseline_reliance"][cid]
            return result
        if state["rebaseline_active"][cid]:
            state["rebaseline_samples"][cid].append(float(reliance_rate))
            if len(state["rebaseline_samples"][cid]) >= self.CONFIG["promotion_rebaseline_rounds"]:
                state["baseline_reliance"][cid] = float(np.mean(state["rebaseline_samples"][cid]))
                state["rebaseline_active"][cid] = False
                state["rebaseline_samples"][cid].clear()
                state["rebaseline_start_round"][cid] = None
                state["reliance_history"][cid].clear()
                state["resolution_history"][cid].clear()
                state["escalation_count_history"][cid].clear()
                state["resolved_count_history"][cid].clear()
                state["total_event_count_history"][cid].clear()
                result["rebaseline_completed"] = True
            result["baseline_reliance"] = state["baseline_reliance"][cid]
            result["rebaseline_active"] = state["rebaseline_active"][cid]
            return result
        baseline = float(state["baseline_reliance"][cid])
        if np.isnan(baseline):
            baseline = 0.0
        required_reliance = self.required_promotion_reliance(baseline)
        result["baseline_reliance"] = baseline
        result["required_reliance_for_promotion"] = required_reliance
        profile = state["shift_profiles"].get(cid)
        if profile is not None:
            result["shift_eligible_client"] = bool(profile.get("eligible", False))
            if not profile.get("eligible", False):
                result["control_client_block"] = False
        state["reliance_history"][cid].append(float(reliance_rate))
        state["resolution_history"][cid].append(float(resolution_rate))
        state["escalation_count_history"][cid].append(int(escalated_count))
        state["resolved_count_history"][cid].append(int(resolved_count))
        state["total_event_count_history"][cid].append(int(total_event_count))
        since_accept = round_id - state["last_promotion_round"][cid]
        if since_accept < self.CONFIG["promotion_accept_cooldown_rounds"]:
            result["promotion_accept_cooldown_active"] = True
            return result
        since_attempt = round_id - state["last_promotion_attempt_round"][cid]
        if since_attempt < self.CONFIG["promotion_reject_cooldown_rounds"]:
            result["promotion_reject_cooldown_active"] = True
            return result
        total_events_window = int(sum(state["total_event_count_history"][cid]))
        total_escalated_window = int(sum(state["escalation_count_history"][cid]))
        total_resolved_window = int(sum(state["resolved_count_history"][cid]))
        pooled_reliance_rate = float(total_escalated_window / max(total_events_window, 1))
        pooled_resolution_rate = float(total_resolved_window / max(total_escalated_window, 1))
        persistent_reliance = (
            len(state["total_event_count_history"][cid]) == self.CONFIG["persistence_rounds"]
            and pooled_reliance_rate >= required_reliance
            and (float(reliance_rate) >= required_reliance)
        )
        persistent_resolution = (
            len(state["resolution_history"][cid]) == self.CONFIG["persistence_rounds"]
            and total_escalated_window >= self.CONFIG["promotion_min_escalated_events_window"]
            and (pooled_resolution_rate >= self.CONFIG["higher_resolution_threshold"])
        )
        result["promotion_window_total_event_count"] = total_events_window
        result["promotion_window_escalated_count"] = total_escalated_window
        result["promotion_window_reliance_rate"] = pooled_reliance_rate
        result["promotion_window_resolved_count"] = total_resolved_window
        result["promotion_window_resolution_rate"] = pooled_resolution_rate
        recent_teacher_items = [
            item
            for item in state["teacher_buffers"][cid]
            if round_id - int(item["round"]) <= self.CONFIG["teacher_memory_rounds"]
        ]
        result["recent_teacher_samples"] = len(recent_teacher_items)
        enough_teachers = (
            len(recent_teacher_items) >= self.CONFIG["min_teacher_samples_for_promotion"]
        )
        if not (persistent_reliance and persistent_resolution and enough_teachers):
            return result
        state["last_promotion_attempt_round"][cid] = round_id
        result["promotion_triggered"] = True
        before_weights = state["l1_base_weights"][cid]
        before_metrics = self.evaluate_weights(
            before_weights, self.client_gate_local[cid], state["l1_temperature"][cid]
        )
        teacher_items = list(recent_teacher_items)
        teacher_kl_before = self.teacher_kl_loss(before_weights, teacher_items)
        acceptance_before = self.teacher_sample_acceptance_rate(
            before_weights, teacher_items, state["l1_temperature"][cid], state["l1_threshold"][cid]
        )
        replay_count = min(self.CONFIG["promotion_replay_samples"], len(self.client_train[cid]))
        replay_df = self.client_train[cid].sample(
            replay_count, random_state=self.CONFIG["seed"] + round_id + int(cid[1:])
        )
        candidate_weights = self.train_promotion_candidate(before_weights, teacher_items, replay_df)
        candidate_metrics = self.evaluate_weights(
            candidate_weights, self.client_gate_local[cid], state["l1_temperature"][cid]
        )
        before_owned_metrics = self.evaluate_owned_specialist(
            before_weights,
            cid,
            self.client_gate_local[cid],
            temperature=state["l1_temperature"][cid],
        )
        candidate_owned_metrics = self.evaluate_owned_specialist(
            candidate_weights,
            cid,
            self.client_gate_local[cid],
            temperature=state["l1_temperature"][cid],
        )
        owned_accuracy_retained = (
            candidate_owned_metrics["owned_accuracy"]
            >= before_owned_metrics["owned_accuracy"]
            - self.CONFIG["promotion_max_owned_accuracy_drop"]
        )
        owned_macro_retained = (
            candidate_owned_metrics["owned_macro_recall"]
            >= before_owned_metrics["owned_macro_recall"]
            - self.CONFIG["promotion_max_owned_macro_recall_drop"]
        )
        before_owned_min = before_owned_metrics["owned_min_class_recall"]
        candidate_owned_min = candidate_owned_metrics["owned_min_class_recall"]
        owned_min_retained = (
            np.isnan(before_owned_min)
            or np.isnan(candidate_owned_min)
            or candidate_owned_min
            >= before_owned_min - self.CONFIG["promotion_max_owned_min_recall_drop"]
        )
        owned_retention_ok = owned_accuracy_retained and owned_macro_retained and owned_min_retained
        teacher_kl_after = self.teacher_kl_loss(candidate_weights, teacher_items)
        acceptance_after = self.teacher_sample_acceptance_rate(
            candidate_weights,
            teacher_items,
            state["l1_temperature"][cid],
            state["l1_threshold"][cid],
        )
        delta_f1 = candidate_metrics["macro_f1"] - before_metrics["macro_f1"]
        kl_improvement = teacher_kl_before - teacher_kl_after
        acceptance_gain = acceptance_after - acceptance_before
        retention_ok = (
            delta_f1 >= -self.CONFIG["promotion_max_local_f1_drop"]
            and candidate_metrics["accuracy"]
            >= before_metrics["accuracy"] - self.CONFIG["promotion_max_local_accuracy_drop"]
        )
        class_recall_retention_ok, promotion_recall_details = self.recall_retention_is_safe(
            before_metrics,
            candidate_metrics,
            self.client_gate_local[cid],
            base_drop=self.CONFIG["promotion_base_supported_class_recall_drop"],
            max_drop=self.CONFIG["promotion_max_adaptive_class_recall_drop"],
        )
        worst_supported_recall_drop = min(
            [detail["recall_delta"] for detail in promotion_recall_details] or [0.0]
        )
        teacher_improved = kl_improvement >= self.CONFIG["promotion_min_teacher_kl_improvement"]
        reliance_proxy_improved = acceptance_gain >= self.CONFIG["promotion_min_acceptance_gain"]
        accepted = (
            retention_ok
            and class_recall_retention_ok
            and owned_retention_ok
            and teacher_improved
            and reliance_proxy_improved
        )
        result.update(
            {
                "promotion_class_recall_retention_ok": class_recall_retention_ok,
                "promotion_worst_supported_recall_drop": worst_supported_recall_drop,
                "promotion_accepted": accepted,
                "promotion_pre_local_f1": before_metrics["macro_f1"],
                "promotion_post_local_f1": candidate_metrics["macro_f1"],
                "promotion_delta_local_f1": delta_f1,
                "promotion_pre_local_accuracy": before_metrics["accuracy"],
                "promotion_post_local_accuracy": candidate_metrics["accuracy"],
                "promotion_delta_local_accuracy": candidate_metrics["accuracy"]
                - before_metrics["accuracy"],
                "promotion_pre_owned_accuracy": before_owned_metrics["owned_accuracy"],
                "promotion_post_owned_accuracy": candidate_owned_metrics["owned_accuracy"],
                "promotion_delta_owned_accuracy": candidate_owned_metrics["owned_accuracy"]
                - before_owned_metrics["owned_accuracy"],
                "promotion_pre_owned_macro_recall": before_owned_metrics["owned_macro_recall"],
                "promotion_post_owned_macro_recall": candidate_owned_metrics["owned_macro_recall"],
                "promotion_delta_owned_macro_recall": candidate_owned_metrics["owned_macro_recall"]
                - before_owned_metrics["owned_macro_recall"],
                "promotion_owned_retention_ok": owned_retention_ok,
                "teacher_kl_before": teacher_kl_before,
                "teacher_kl_after": teacher_kl_after,
                "teacher_kl_improvement": kl_improvement,
                "teacher_acceptance_before": acceptance_before,
                "teacher_acceptance_after": acceptance_after,
                "teacher_acceptance_gain": acceptance_gain,
            }
        )
        if accepted:
            state["l1_base_weights"][cid] = candidate_weights
            state["promotion_count"][cid] += 1
            state["last_promotion_round"][cid] = round_id
            state["l1_temperature"][cid], state["l1_threshold"][cid], gate_record = (
                self.calibrate_and_gate(
                    candidate_weights, self.client_calibration_local[cid], "L1", cid, round_id
                )
            )
            state["threshold_diagnostics"].append(gate_record)
            promoted_reliability = self.fit_reliability_table(
                candidate_weights,
                self.client_calibration_local[cid],
                state["l1_temperature"][cid],
                "L1",
                cid,
                round_id,
            )
            state["reliability_models"]["L1"][cid] = promoted_reliability
            for bin_row in promoted_reliability["bins"]:
                state["reliability_history"].append(
                    {
                        "round": round_id,
                        "level": "L1",
                        "model_id": cid,
                        "overall_accuracy": promoted_reliability["overall_accuracy"],
                        **bin_row,
                        "refresh_reason": "accepted_promotion",
                    }
                )
            state["rebaseline_active"][cid] = True
            state["rebaseline_samples"][cid].clear()
            state["rebaseline_start_round"][cid] = round_id + 1
            state["teacher_buffers"][cid].clear()
            state["reliance_history"][cid].clear()
            state["resolution_history"][cid].clear()
            state["escalation_count_history"][cid].clear()
            state["resolved_count_history"][cid].clear()
            state["total_event_count_history"][cid].clear()
        state["promotion_history"].append(
            {"round": round_id, "client_id": cid, "context": self.CLIENT_TO_CONTEXT[cid], **result}
        )
        return result
