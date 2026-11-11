"""Reference-derived training/local. See README.md."""

import math

import numpy as np

import pandas as pd

import tensorflow as tf

class Operations:
    # Reference cells (zero-based): {'tempo': 21, 'ditto': 20, 'hfl': 19, 'flat': 19}
    def select_deterministic_batch(self, df, batch_size, round_id, client_id, salt=0):
        if len(df) == 0:
            return df.copy()
        numeric_client = int(client_id[1:]) if str(client_id).startswith("C") else 0
        rng = np.random.default_rng(
            self.CONFIG["seed"] + 100000 * int(round_id) + 1000 * numeric_client + int(salt)
        )
        replace = len(df) < batch_size
        size = batch_size if replace else min(batch_size, len(df))
        selected = rng.choice(np.arange(len(df)), size=size, replace=replace)
        return df.iloc[selected].copy().reset_index(drop=True)

    # Reference cells (zero-based): {'tempo': 21, 'ditto': 20, 'hfl': 19, 'flat': 19}
    def select_class_balanced_batch(self, df, batch_size, round_id, client_id, salt=0):
        if len(df) == 0:
            return df.copy()
        available_classes = sorted(df["label_id"].unique().tolist())
        if len(available_classes) <= 1:
            return self.select_deterministic_batch(df, batch_size, round_id, client_id, salt)
        numeric_client = int(client_id[1:]) if str(client_id).startswith("C") else 0
        rng = np.random.default_rng(
            self.CONFIG["seed"] + 200000 * int(round_id) + 2000 * numeric_client + int(salt)
        )
        per_class = max(1, int(math.ceil(batch_size / len(available_classes))))
        selected_frames = []
        for class_id in available_classes:
            class_df = df[df["label_id"] == class_id]
            if len(class_df) == 0:
                continue
            indices = rng.choice(
                np.arange(len(class_df)), size=per_class, replace=len(class_df) < per_class
            )
            selected_frames.append(class_df.iloc[indices])
        batch_df = pd.concat(selected_frames, ignore_index=True)
        if len(batch_df) > batch_size:
            batch_df = batch_df.sample(
                n=batch_size,
                random_state=self.CONFIG["seed"] + round_id + numeric_client + int(salt),
            )
        return batch_df.reset_index(drop=True)

    # Reference cells (zero-based): {'hfl': 19, 'flat': 19}
    def local_gradient_step_fedavg(self, start_weights, batch_df, learning_rate, training=True):
        if len(batch_df) == 0:
            return ([w.copy() for w in start_weights], np.nan, np.nan)
        self.worker_model.set_weights(start_weights)
        x_batch, y_batch = self.load_batch_tensor(batch_df, training=training)
        with tf.GradientTape() as tape:
            logits = self.worker_model(x_batch, training=True)
            total_loss = self.classification_loss(y_batch, logits)
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
        )

    # Reference cells (zero-based): {'hfl': 19, 'flat': 19}
    def train_fedavg_client(self, cid, round_id, parent_weights):
        working_weights = [w.copy() for w in parent_weights]
        losses = []
        accuracies = []
        samples_seen = 0
        for local_step in range(self.CONFIG["fedavg_local_steps_per_round"]):
            if self.CONFIG["fedavg_class_balanced_batches"]:
                batch_df = self.select_class_balanced_batch(
                    self.client_train[cid],
                    self.CONFIG["fedavg_local_batch_size"],
                    round_id,
                    cid,
                    salt=29 + 101 * local_step,
                )
            else:
                batch_df = self.select_deterministic_batch(
                    self.client_train[cid],
                    self.CONFIG["fedavg_local_batch_size"],
                    round_id,
                    cid,
                    salt=29 + 101 * local_step,
                )
            working_weights, loss_value, accuracy_value = self.local_gradient_step_fedavg(
                working_weights, batch_df, self.CONFIG["fedavg_local_lr"], training=True
            )
            losses.append(loss_value)
            accuracies.append(accuracy_value)
            samples_seen += len(batch_df)
        return {
            "weights": working_weights,
            "samples": int(samples_seen),
            "aggregation_weight": int(len(self.client_train[cid])),
            "local_steps": int(self.CONFIG["fedavg_local_steps_per_round"]),
            "loss": float(np.nanmean(losses)),
            "accuracy": float(np.nanmean(accuracies)),
            "delta_l2_norm": self.weight_delta_l2(working_weights, parent_weights),
            "download_bytes": self.weights_size_bytes(parent_weights),
            "upload_bytes": self.weights_size_bytes(working_weights),
            "context": self.CLIENT_TO_CONTEXT[cid],
        }
