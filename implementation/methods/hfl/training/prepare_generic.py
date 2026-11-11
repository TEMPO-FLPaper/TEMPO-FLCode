"""Reference-derived training/prepare_generic_hfl. See README.md."""

import gc

import numpy as np

import pandas as pd

from tensorflow import keras

class Operations:
    # Reference cells (zero-based): {'hfl': 22}
    def prepare_generic(self):
        self._init_state = self.fresh_runtime_state()
        self.set_round_seed(0)
        self.log("=" * 118)
        self.log("NO V12 CHECKPOINT FOUND — RUNNING OWNED-CLASS STAGED INITIALIZATION")
        self.log("=" * 118)
        self.log("\n" + "=" * 118)
        self.log("STAGE 1 — SMALL BALANCED GENERIC MODEL TRAINED ON ALL CLASSES")
        self.log("=" * 118)
        generic_counts_table = pd.DataFrame(
            [
                {
                    "dataset_part": "GenericTrain",
                    "total": len(self.generic_train_df),
                    **{
                        class_name: int(np.sum(self.generic_train_df["label"] == class_name))
                        for class_name in self.CLASS_NAMES
                    },
                },
                {
                    "dataset_part": "GenericValidation",
                    "total": len(self.generic_validation_df),
                    **{
                        class_name: int(np.sum(self.generic_validation_df["label"] == class_name))
                        for class_name in self.CLASS_NAMES
                    },
                },
                {
                    "dataset_part": "GenericInternalTest",
                    "total": len(self.generic_internal_test_df),
                    **{
                        class_name: int(
                            np.sum(self.generic_internal_test_df["label"] == class_name)
                        )
                        for class_name in self.CLASS_NAMES
                    },
                },
            ]
        )
        self.log("\nGENERIC TRAIN / VALIDATION / INTERNAL-TEST COUNTS")
        self.display(generic_counts_table)
        generic_model = self.build_tempo_model(self.NUM_CLASSES)
        backbone = None
        for layer in generic_model.layers:
            if layer.name.startswith("mobilenetv2"):
                backbone = layer
                break
        if backbone is None:
            raise RuntimeError("MobileNetV2 backbone was not found.")
        train_ds = self.make_tf_dataset(
            self.generic_train_df, batch_size=self.CONFIG["generic_batch_size"], training=True
        )
        val_ds = self.make_tf_dataset(
            self.generic_validation_df,
            batch_size=self.CONFIG["generic_batch_size"],
            training=False,
            shuffle=False,
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.CONFIG["early_stopping_patience"],
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=self.CONFIG["reduce_lr_patience"],
                factor=self.CONFIG["reduce_lr_factor"],
                min_lr=1e-06,
                verbose=1,
            ),
        ]
        class_weight = self.generic_class_weights()
        backbone.trainable = False
        self.compile_for_fit(generic_model, self.CONFIG["generic_lr_head"])
        self.log("\n[Generic] Training classifier head on all six classes...")
        generic_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.CONFIG["generic_head_epochs"],
            callbacks=callbacks + [self.training_callback()],
            class_weight=class_weight,
            verbose=0,
        )
        backbone.trainable = True
        for layer in backbone.layers[: -self.CONFIG["trainable_tail_layers"]]:
            layer.trainable = False
        self.compile_for_fit(generic_model, self.CONFIG["generic_lr_finetune"])
        self.log("\n[Generic] Fine-tuning MobileNetV2 tail...")
        generic_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.CONFIG["generic_finetune_epochs"],
            callbacks=callbacks + [self.training_callback()],
            class_weight=class_weight,
            verbose=0,
        )
        self._init_generic_weights = [w.copy() for w in generic_model.get_weights()]
        generic_weights_path = self.DIRS["config"] / "generic_seed_v12_2.weights.h5"
        self.save_weights_to_file(self._init_generic_weights, generic_weights_path)
        generic_general_rows = self.evaluate_general_stage_rows(
            "Generic",
            "Generic_L3_Small_Generic_Model",
            0,
            [
                {
                    "level": "L3",
                    "model_id": "Generic",
                    "context": "Campus",
                    "weights": self._init_generic_weights,
                    "temperature": 1.0,
                }
            ],
        )
        generic_owned_rows = self.evaluate_owned_stage_rows(
            "Generic",
            "Generic_At_Each_Client",
            0,
            [
                {
                    "client_id": cid,
                    "model_id": "Generic",
                    "evaluated_model_level": "Generic",
                    "weights": self._init_generic_weights,
                    "temperature": 1.0,
                }
                for cid in self.CLIENT_IDS
            ],
        )
        self.upsert_stage_metric_rows(
            self._init_state, "general_accuracy_rows", generic_general_rows
        )
        self.upsert_stage_metric_rows(self._init_state, "owned_accuracy_rows", generic_owned_rows)
        self.log("\nV12.2 GENERAL TEST — GENERIC MODEL")
        self.display(
            pd.DataFrame(generic_general_rows)[
                ["level", "model_id", "test_set", "test_samples", "accuracy", "macro_f1"]
            ]
        )
        self.log("\nV12.2 OWNED TEST — GENERIC MODEL AT EACH CLIENT")
        self.display(
            pd.DataFrame(generic_owned_rows)[
                [
                    "client_id",
                    "context",
                    "planned_classes",
                    "test_samples",
                    "owned_test_samples",
                    "owned_accuracy",
                    "owned_macro_f1",
                ]
            ]
        )
        generic_internal_summary = self.overall_stage_row(
            self._init_generic_weights,
            self.generic_internal_test_df,
            "Stage1_Generic_Internal_Test",
            "GenericSeed",
            "AllClasses",
            temperature=1.0,
        )
        generic_internal_class_rows = self.classwise_accuracy_rows(
            self._init_generic_weights,
            self.generic_internal_test_df,
            "Stage1_Generic_Internal_Test",
            "GenericSeed",
            "AllClasses",
            temperature=1.0,
            train_reference_df=self.generic_train_df,
        )
        generic_global_dev_summary = self.overall_stage_row(
            self._init_generic_weights,
            self.global_validation_df,
            "Stage1_Generic_Global_Development",
            "GenericSeed",
            "AllClasses",
            temperature=1.0,
        )
        self._init_state["generic_stage_summary"] = dict(generic_internal_summary)
        self._init_state["generic_global_class_rows"] = list(generic_internal_class_rows)
        self.log("\nGENERIC MODEL — BALANCED INTERNAL TEST")
        self.display(pd.DataFrame([generic_internal_summary]))
        generic_internal_class_df = pd.DataFrame(generic_internal_class_rows)
        self.log("\nGENERIC MODEL — PER-CLASS ACCURACY/RECALL ON ALL SIX CLASSES")
        self.display(
            generic_internal_class_df[
                ["class_name", "train_count_at_node", "evaluation_support", "class_accuracy_recall"]
            ]
        )
        self.show_confusion_matrix_for_weights(
            self._init_generic_weights,
            self.generic_internal_test_df,
            "V12 Small Generic Model — Balanced Internal Test",
            self.DIRS["figures"] / "stage1_generic_allclass_confusion_v12_2.png",
        )
        self.log("\nGENERIC MODEL — LARGER GLOBAL DEVELOPMENT VALIDATION")
        self.display(pd.DataFrame([generic_global_dev_summary]))
        del generic_model
        gc.collect()
        self.log("\n[FedAvg] Released generic training model before FedAvg bootstrap.")
        self.print_memory_snapshot("[Before FedAvg bootstrap]")
        self.log("\nINITIAL FEDAVG BOOTSTRAP — GENERIC -> L1 -> L2 -> L3")
