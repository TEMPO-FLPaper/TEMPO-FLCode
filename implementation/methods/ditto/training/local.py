"""Reference-derived training/local. See README.md."""

import math

import numpy as np

import pandas as pd

import tensorflow as tf
from implementation.shared.training.local import Operations as SharedOperations


class Operations(SharedOperations):
    # Reference cells (zero-based): {'ditto': 20}
    def _reference_for_variable(self, variable, reference_weights):
        if reference_weights is None:
            return None
        for index, candidate in enumerate(self.worker_model.weights):
            if id(candidate) == id(variable):
                return reference_weights[index]
        raise RuntimeError(f"Could not map worker variable {variable.name} to reference weights.")

    # Reference cells (zero-based): {'ditto': 20}
    def local_gradient_step_ditto(
        self,
        start_weights,
        batch_df,
        learning_rate,
        reference_weights=None,
        ditto_lambda=0.0,
        training=True,
    ):
        if len(batch_df) == 0:
            return ([w.copy() for w in start_weights], np.nan, np.nan, np.nan)
        self.worker_model.set_weights(start_weights)
        x_batch, y_batch = self.load_batch_tensor(batch_df, training=training)
        with tf.GradientTape() as tape:
            logits = self.worker_model(x_batch, training=True)
            empirical_loss = self.classification_loss(y_batch, logits)
            proximal_loss = tf.constant(0.0, dtype=empirical_loss.dtype)
            if reference_weights is not None and float(ditto_lambda) > 0.0:
                proximal_terms = []
                for variable in self.worker_model.trainable_variables:
                    reference = self._reference_for_variable(variable, reference_weights)
                    proximal_terms.append(
                        tf.reduce_sum(tf.square(variable - tf.cast(reference, variable.dtype)))
                    )
                if proximal_terms:
                    proximal_loss = tf.add_n(proximal_terms)
            total_loss = empirical_loss + 0.5 * float(ditto_lambda) * proximal_loss
        gradients = tape.gradient(total_loss, self.worker_model.trainable_variables)
        valid_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.worker_model.trainable_variables)
            if gradient is not None
        ]
        if valid_pairs:
            clipped_gradients, _ = tf.clip_by_global_norm(
                [pair[0] for pair in valid_pairs], self.CONFIG["gradient_clip_norm"]
            )
            for clipped_gradient, (_, variable) in zip(clipped_gradients, valid_pairs):
                variable.assign_sub(tf.cast(learning_rate, variable.dtype) * clipped_gradient)
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_batch), tf.float32))
        return (
            [w.copy() for w in self.worker_model.get_weights()],
            float(total_loss.numpy()),
            float(accuracy.numpy()),
            float(proximal_loss.numpy()),
        )

    # Reference cells (zero-based): {'ditto': 20}
    def train_ditto_client(self, cid, round_id, global_parent_weights, personal_parent_weights):
        global_working = [w.copy() for w in global_parent_weights]
        personal_working = [w.copy() for w in personal_parent_weights]
        global_losses = []
        global_accuracies = []
        personal_losses = []
        personal_accuracies = []
        personal_proximal_norms = []
        samples_seen = 0
        for local_step in range(self.CONFIG["ditto_local_steps_per_round"]):
            if self.CONFIG["ditto_class_balanced_batches"]:
                batch_df = self.select_class_balanced_batch(
                    self.client_train[cid],
                    self.CONFIG["ditto_batch_size"],
                    round_id,
                    cid,
                    salt=29 + 101 * local_step,
                )
            else:
                batch_df = self.select_deterministic_batch(
                    self.client_train[cid],
                    self.CONFIG["ditto_batch_size"],
                    round_id,
                    cid,
                    salt=29 + 101 * local_step,
                )
            global_working, global_loss, global_accuracy, _ = self.local_gradient_step_ditto(
                global_working,
                batch_df,
                self.CONFIG["ditto_global_lr"],
                reference_weights=None,
                ditto_lambda=0.0,
                training=True,
            )
            personal_working, personal_loss, personal_accuracy, proximal_norm = (
                self.local_gradient_step_ditto(
                    personal_working,
                    batch_df,
                    self.CONFIG["ditto_personal_lr"],
                    reference_weights=global_parent_weights,
                    ditto_lambda=self.CONFIG["ditto_lambda"],
                    training=True,
                )
            )
            global_losses.append(global_loss)
            global_accuracies.append(global_accuracy)
            personal_losses.append(personal_loss)
            personal_accuracies.append(personal_accuracy)
            personal_proximal_norms.append(proximal_norm)
            samples_seen += len(batch_df)
        return {
            "global_weights": global_working,
            "personalized_weights": personal_working,
            "samples": int(samples_seen),
            "aggregation_weight": int(len(self.client_train[cid])),
            "local_steps": int(self.CONFIG["ditto_local_steps_per_round"]),
            "global_loss": float(np.nanmean(global_losses)),
            "global_accuracy": float(np.nanmean(global_accuracies)),
            "personal_loss": float(np.nanmean(personal_losses)),
            "personal_accuracy": float(np.nanmean(personal_accuracies)),
            "mean_proximal_norm": float(np.nanmean(personal_proximal_norms)),
            "global_delta_l2_norm": self.weight_delta_l2(global_working, global_parent_weights),
            "personal_delta_l2_norm": self.weight_delta_l2(
                personal_working, personal_parent_weights
            ),
            "download_bytes": self.weights_size_bytes(global_parent_weights),
            "global_upload_bytes": self.weights_size_bytes(global_working),
            "personal_state_bytes": self.weights_size_bytes(personal_working),
            "context": self.CLIENT_TO_CONTEXT[cid],
        }
