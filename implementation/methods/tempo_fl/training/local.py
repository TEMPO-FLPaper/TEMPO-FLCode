"""Reference-derived training/local. See README.md."""

import math

import numpy as np

import pandas as pd

import tensorflow as tf
from implementation.shared.training.local import Operations as SharedOperations


class Operations(SharedOperations):
    # Reference cells (zero-based): {'tempo': 21}
    def local_gradient_step(
        self,
        start_weights,
        batch_df,
        learning_rate,
        training=True,
        proximal_reference_weights=None,
        prox_mu=0.0,
    ):
        if len(batch_df) == 0:
            return ([w.copy() for w in start_weights], np.nan, np.nan)
        self.worker_model.set_weights(start_weights)
        x_batch, y_batch = self.load_batch_tensor(batch_df, training=training)
        if proximal_reference_weights is not None and float(prox_mu) > 0:
            reference_tensors = [
                tf.convert_to_tensor(w, dtype=v.dtype)
                for w, v in zip(proximal_reference_weights, self.worker_model.weights)
            ]
        else:
            reference_tensors = None
        with tf.GradientTape() as tape:
            logits = self.worker_model(x_batch, training=True)
            ce_loss = self.classification_loss(y_batch, logits)
            total_loss = ce_loss
            if reference_tensors is not None:
                prox_loss = tf.add_n(
                    [
                        tf.reduce_sum(tf.square(variable - reference))
                        for variable, reference in zip(self.worker_model.weights, reference_tensors)
                    ]
                )
                total_loss = total_loss + 0.5 * float(prox_mu) * prox_loss
        gradients = tape.gradient(total_loss, self.worker_model.trainable_variables)
        valid_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.worker_model.trainable_variables)
            if gradient is not None
        ]
        if valid_pairs:
            gradient_list = [pair[0] for pair in valid_pairs]
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradient_list, self.CONFIG["gradient_clip_norm"]
            )
            for clipped_gradient, (_, variable) in zip(clipped_gradients, valid_pairs):
                variable.assign_sub(tf.cast(learning_rate, variable.dtype) * clipped_gradient)
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_batch), tf.float32))
        return (
            [w.copy() for w in self.worker_model.get_weights()],
            float(total_loss.numpy()),
            float(accuracy.numpy()),
        )

    # Reference cells (zero-based): {'tempo': 42}
    def routine_l1_maintenance(self, cid, round_id, state):
        return {
            "maintenance_attempted": False,
            "maintenance_accepted": False,
            "reason": "Disabled in V12: Participate != Adopt",
        }

    # Reference cells (zero-based): {'tempo': 43}
    def train_l2_shadow(self, cid, round_id, state):
        context_name = self.CLIENT_TO_CONTEXT[cid]
        start_weights = [w.copy() for w in state["shadow_l2_weights"][cid]]
        working_weights = [w.copy() for w in start_weights]
        losses = []
        accuracies = []
        samples_seen = 0
        for local_step in range(self.CONFIG["l2_local_steps_per_round"]):
            if self.CONFIG["l2_class_balanced_batches"]:
                batch_df = self.select_class_balanced_batch(
                    self.client_train[cid],
                    self.CONFIG["l2_shadow_batch_size"],
                    round_id,
                    cid,
                    salt=29 + 101 * local_step,
                )
            else:
                batch_df = self.select_deterministic_batch(
                    self.client_train[cid],
                    self.CONFIG["l2_shadow_batch_size"],
                    round_id,
                    cid,
                    salt=29 + 101 * local_step,
                )
            working_weights, train_loss, train_acc = self.local_gradient_step(
                working_weights,
                batch_df,
                self.CONFIG["l2_shadow_lr"],
                training=True,
                proximal_reference_weights=start_weights,
                prox_mu=self.CONFIG["l2_prox_mu"],
            )
            losses.append(train_loss)
            accuracies.append(train_acc)
            samples_seen += len(batch_df)
        state["shadow_l2_weights"][cid] = [w.copy() for w in working_weights]
        return {
            "weights": working_weights,
            "samples": samples_seen,
            "aggregation_weight": len(self.client_train[cid]),
            "local_steps": self.CONFIG["l2_local_steps_per_round"],
            "loss": float(np.nanmean(losses)),
            "accuracy": float(np.nanmean(accuracies)),
            "delta_l2_norm": self.weight_delta_l2(working_weights, start_weights),
            "download_bytes": 0,
            "upload_bytes": self.weights_size_bytes(working_weights),
            "context": context_name,
        }
