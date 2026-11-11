"""Reference-derived training/specialize_clients_tempo. See README.md."""

import gc, time

import numpy as np

import pandas as pd

import tensorflow as tf

class Operations:
    # Reference cells (zero-based): {'tempo': 29}
    def specialize_clients(self):
        post_rows = []
        post_all_class_rows = []
        specialization_search_rows = []
        specialization_runtime_rows = []
        self._init_post_owned_metrics = {}
        specialization_model = self.build_tempo_model(self.NUM_CLASSES)
        self.compile_for_fit(specialization_model, self.CONFIG["specialization_initial_lr"])

        def reset_reusable_specialization_optimizer(optimizer, learning_rate):
            try:
                for variable in optimizer.variables:
                    variable.assign(tf.zeros_like(variable))
            except Exception:
                pass
            try:
                optimizer.learning_rate.assign(float(learning_rate))
            except Exception:
                optimizer.learning_rate = float(learning_rate)

        for cid in self.CLIENT_IDS:
            client_stage_start = time.time()
            context_name = self.CLIENT_TO_CONTEXT[cid]
            planned_classes = self.CLIENT_PLANNED_CLASSES[cid]
            specialization_model.set_weights(self._init_generic_weights)
            reset_reusable_specialization_optimizer(
                specialization_model.optimizer, self.CONFIG["specialization_initial_lr"]
            )
            training_df = self.make_specialization_training_df(
                cid, self.CONFIG["seed"] + 3000 + int(cid[1:])
            )
            candidate_ds = self.make_tf_dataset(
                training_df, batch_size=self.CONFIG["specialization_batch_size"], training=True
            )
            baseline_selection_metrics = self.evaluate_owned_specialist_fast(
                self._init_generic_weights, cid, self.client_gate_local[cid], temperature=1.0
            )
            baseline_report_metrics = self._init_pre_owned_metrics[cid]
            baseline_score = self.specialization_selection_score(baseline_selection_metrics)
            owned_train_counts = {
                class_name: int(np.sum(training_df["label"] == class_name))
                for class_name in planned_classes
            }
            owned_gate_counts = {
                class_name: int(np.sum(self.client_gate_local[cid]["label"] == class_name))
                for class_name in planned_classes
            }
            self.log("\n" + "-" * 118)
            self.log(f"[{cid} | {context_name}] planned={planned_classes}")
            self.log(
                f"  Private train n={len(self.client_train[cid])} | specialization train n={len(training_df)} | owned train counts={owned_train_counts}"
            )
            self.log(
                f"  Selection gate n={len(self.client_gate_local[cid])} | owned gate counts={owned_gate_counts}"
            )
            self.print_fast_specialization_metrics("  Generic baseline", baseline_selection_metrics)
            self.print_memory_snapshot(f"  [{cid} start]")
            best_weights = [w.copy() for w in self._init_generic_weights]
            best_selection_metrics = baseline_selection_metrics
            best_score = float(baseline_score)
            best_epoch = 0
            best_lr = 0.0
            best_source = "generic_baseline"
            no_improvement_epochs = 0
            no_improvement_for_lr = 0
            lr_reductions = 0
            current_lr = float(self.CONFIG["specialization_initial_lr"])
            max_epochs = int(self.CONFIG["specialization_max_epochs"])
            min_epochs = int(self.CONFIG["specialization_min_epochs"])
            baseline_min_recall = baseline_selection_metrics["owned_min_class_recall"]
            for epoch_number in range(1, max_epochs + 1):
                epoch_start = time.time()
                history = specialization_model.fit(
                    candidate_ds, epochs=1, verbose=0, callbacks=[self.training_callback()]
                )
                candidate_weights = [w.copy() for w in specialization_model.get_weights()]
                candidate_metrics = self.evaluate_owned_specialist_fast(
                    candidate_weights, cid, self.client_gate_local[cid], temperature=1.0
                )
                candidate_score = self.specialization_selection_score(candidate_metrics)
                candidate_min_recall = candidate_metrics["owned_min_class_recall"]
                min_recall_safe = (
                    np.isnan(baseline_min_recall)
                    or np.isnan(candidate_min_recall)
                    or candidate_min_recall
                    >= baseline_min_recall - self.CONFIG["specialization_max_owned_min_recall_drop"]
                )
                score_improved = (
                    min_recall_safe
                    and candidate_score
                    >= best_score + self.CONFIG["specialization_min_score_delta"]
                )
                if score_improved:
                    best_weights = [w.copy() for w in candidate_weights]
                    best_selection_metrics = candidate_metrics
                    best_score = float(candidate_score)
                    best_epoch = int(epoch_number)
                    best_lr = float(current_lr)
                    best_source = "local_finetune"
                    no_improvement_epochs = 0
                    no_improvement_for_lr = 0
                else:
                    no_improvement_epochs += 1
                    no_improvement_for_lr += 1
                last_loss = float(history.history.get("loss", [np.nan])[-1])
                last_accuracy = float(history.history.get("accuracy", [np.nan])[-1])
                elapsed = float(time.time() - epoch_start)
                specialization_search_rows.append(
                    {
                        "client_id": cid,
                        "context": context_name,
                        "planned_classes": ",".join(planned_classes),
                        "epoch": int(epoch_number),
                        "learning_rate": float(current_lr),
                        "train_loss": last_loss,
                        "train_accuracy": last_accuracy,
                        "owned_accuracy": candidate_metrics["owned_accuracy"],
                        "owned_macro_recall": candidate_metrics["owned_macro_recall"],
                        "owned_min_class_recall": candidate_metrics["owned_min_class_recall"],
                        "all_local_macro_f1_secondary": candidate_metrics[
                            "all_local_macro_f1_secondary"
                        ],
                        "selection_score": candidate_score,
                        "best_score_so_far": best_score,
                        "score_improved": bool(score_improved),
                        "min_recall_safe": bool(min_recall_safe),
                        "elapsed_sec": elapsed,
                    }
                )
                if self.CONFIG["specialization_print_per_epoch"]:
                    self.log(
                        f"  Epoch {epoch_number:02d}/{max_epochs} | lr={current_lr:.2e} | loss={last_loss:.4f} | trainAcc={last_accuracy:.3f} | score={candidate_score:.4f} | best={best_score:.4f} | time={elapsed:.1f}s"
                    )
                    if self.CONFIG["specialization_print_per_class_each_epoch"]:
                        self.print_fast_specialization_metrics(
                            "    owned detail", candidate_metrics
                        )
                    self.print_memory_snapshot(f"    [{cid} epoch {epoch_number}]")
                if (
                    no_improvement_for_lr >= self.CONFIG["specialization_lr_patience"]
                    and current_lr > self.CONFIG["specialization_min_lr"]
                ):
                    new_lr = max(
                        self.CONFIG["specialization_min_lr"],
                        current_lr * self.CONFIG["specialization_lr_reduce_factor"],
                    )
                    if new_lr < current_lr:
                        try:
                            specialization_model.optimizer.learning_rate.assign(float(new_lr))
                        except Exception:
                            specialization_model.optimizer.learning_rate = float(new_lr)
                        current_lr = float(new_lr)
                        lr_reductions += 1
                        no_improvement_for_lr = 0
                        self.log(f"    LR reduced -> {current_lr:.2e} after owned-score plateau.")
                target_met = (
                    epoch_number >= min_epochs
                    and (not np.isnan(candidate_metrics["owned_accuracy"]))
                    and (not np.isnan(candidate_metrics["owned_macro_recall"]))
                    and (
                        candidate_metrics["owned_accuracy"]
                        >= self.CONFIG["specialization_target_owned_accuracy"]
                    )
                    and (
                        candidate_metrics["owned_macro_recall"]
                        >= self.CONFIG["specialization_target_owned_macro_recall"]
                    )
                    and min_recall_safe
                )
                if self.CONFIG["specialization_stop_when_target_met"] and target_met:
                    self.log(f"  Early stop: owned target reached at epoch {epoch_number}.")
                    break
                if (
                    epoch_number >= min_epochs
                    and no_improvement_epochs >= self.CONFIG["specialization_early_stop_patience"]
                    and (current_lr <= self.CONFIG["specialization_min_lr"])
                ):
                    self.log(
                        f"  Early stop: no owned-score improvement for {no_improvement_epochs} epochs at minimum learning rate."
                    )
                    break
                del candidate_weights
                gc.collect()
            specialization_model.set_weights(best_weights)
            self._init_state["l1_base_weights"][cid] = [w.copy() for w in best_weights]
            self.save_weights_to_file(
                self._init_state["l1_base_weights"][cid],
                self.DIRS["config"] / f"initial_l1_{cid}_v12_2.weights.h5",
            )
            best_metrics = self.evaluate_owned_specialist(
                best_weights, cid, self.client_val_local[cid], temperature=1.0
            )
            self._init_post_owned_metrics[cid] = best_selection_metrics
            overall = self.overall_stage_row(
                best_weights,
                self.client_val_local[cid],
                "Stage3_OwnedClass_Specialist",
                cid,
                context_name,
                temperature=1.0,
            )
            client_elapsed = float(time.time() - client_stage_start)
            post_row = {
                **overall,
                "planned_classes": best_metrics["planned_classes"],
                "selected_source": best_source,
                "selected_learning_rate": best_lr,
                "selected_epoch": best_epoch,
                "lr_reductions": int(lr_reductions),
                "owned_eval_support": best_metrics["owned_eval_support"],
                "owned_accuracy": best_metrics["owned_accuracy"],
                "owned_macro_recall": best_metrics["owned_macro_recall"],
                "owned_min_class_recall": best_metrics["owned_min_class_recall"],
                "owned_accuracy_gain": best_metrics["owned_accuracy"]
                - baseline_report_metrics["owned_accuracy"],
                "owned_macro_recall_gain": best_metrics["owned_macro_recall"]
                - baseline_report_metrics["owned_macro_recall"],
                "other_accuracy_secondary": best_metrics["other_accuracy_secondary"],
                "other_macro_recall_secondary": best_metrics["other_macro_recall_secondary"],
                "specialization_elapsed_sec": client_elapsed,
            }
            post_rows.append(post_row)
            all_rows = self.classwise_accuracy_rows(
                best_weights,
                self.client_val_local[cid],
                "Stage3_OwnedClass_Specialist",
                cid,
                context_name,
                temperature=1.0,
                train_reference_df=self.client_train[cid],
            )
            post_all_class_rows.extend(self.add_planned_flag(all_rows, cid))
            self.log(
                f"  SELECTED -> {best_source} | lr={best_lr:.2e} | epoch={best_epoch} | bestScore={best_score:.4f}"
            )
            self.log(
                f"  Report-set OwnedAcc: {baseline_report_metrics['owned_accuracy']:.3f} -> {best_metrics['owned_accuracy']:.3f} | OwnedMacroRecall: {baseline_report_metrics['owned_macro_recall']:.3f} -> {best_metrics['owned_macro_recall']:.3f}"
            )
            before_class_map = {
                row["class_name"]: row
                for row in self._init_pre_class_df[
                    (self._init_pre_class_df["model_id"] == cid)
                    & (self._init_pre_class_df["planned_owned_class"] == True)
                ].to_dict("records")
            }
            after_class_map = {
                row["class_name"]: row for row in all_rows if row["class_name"] in planned_classes
            }
            for class_name in planned_classes:
                before_value = before_class_map.get(class_name, {}).get(
                    "class_accuracy_recall", np.nan
                )
                after_value = after_class_map.get(class_name, {}).get(
                    "class_accuracy_recall", np.nan
                )
                support = after_class_map.get(class_name, {}).get("evaluation_support", 0)
                self.log(
                    f"    {class_name:10s}: {before_value:.3f} -> {after_value:.3f} (n={support})"
                )
            self.log(f"  Client specialization time: {client_elapsed:.1f}s")
            self.print_memory_snapshot(f"  [{cid} completed]")
            specialization_runtime_rows.append(
                {
                    "client_id": cid,
                    "context": context_name,
                    "planned_classes": ",".join(planned_classes),
                    "elapsed_sec": client_elapsed,
                    "selected_lr": best_lr,
                    "selected_epoch": best_epoch,
                    "lr_reductions": lr_reductions,
                    "selected_owned_accuracy": best_metrics["owned_accuracy"],
                    "selected_owned_macro_recall": best_metrics["owned_macro_recall"],
                    "ram_gb_after_client": self.current_process_ram_gb(),
                    "gpu_gb_after_client": self.current_gpu_memory_gb(),
                }
            )
            del candidate_ds
            del training_df
            del best_weights
            gc.collect()
        del specialization_model
        gc.collect()
        self.print_memory_snapshot("[After Stage 3 reusable model release]")
        initial_l1_specs = [
            {
                "level": "L1",
                "model_id": cid,
                "context": self.CLIENT_TO_CONTEXT[cid],
                "weights": self._init_state["l1_base_weights"][cid],
                "temperature": 1.0,
            }
            for cid in self.CLIENT_IDS
        ]
        initial_l1_general_rows = self.evaluate_general_stage_rows(
            "Initial", "Initial_L1_After_Owned_Class_Specialization", 1, initial_l1_specs
        )
        initial_l1_owned_rows = self.evaluate_owned_stage_rows(
            "Initial",
            "Initial_L1_After_Owned_Class_Specialization",
            1,
            [
                {
                    "client_id": cid,
                    "model_id": cid,
                    "evaluated_model_level": "L1",
                    "weights": self._init_state["l1_base_weights"][cid],
                    "temperature": 1.0,
                }
                for cid in self.CLIENT_IDS
            ],
        )
        self.upsert_stage_metric_rows(
            self._init_state, "general_accuracy_rows", initial_l1_general_rows
        )
        self.upsert_stage_metric_rows(
            self._init_state, "owned_accuracy_rows", initial_l1_owned_rows
        )
        self.log("\nV12.2 GENERAL TEST — INITIAL L1 MODELS")
        self.display(
            pd.DataFrame(initial_l1_general_rows)[
                ["level", "model_id", "context", "test_samples", "accuracy", "macro_f1"]
            ]
        )
        self.log("\nV12.2 OWNED TEST — INITIAL L1 MODELS")
        self.display(
            pd.DataFrame(initial_l1_owned_rows)[
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
        self._init_post_df = pd.DataFrame(post_rows)
        post_class_df = pd.DataFrame(post_all_class_rows)
        self._init_search_df = pd.DataFrame(specialization_search_rows)
        self._init_specialization_runtime_df = pd.DataFrame(specialization_runtime_rows)
        self._init_state["client_post_specialization_rows"] = list(post_rows)
        self._init_state["client_class_stage_rows"].extend(post_all_class_rows)
        self._init_state["v12_specialization_runtime_rows"] = list(specialization_runtime_rows)
        self._init_owned_comparison = (
            self._init_pre_df[
                [
                    "model_id",
                    "context",
                    "planned_classes",
                    "owned_accuracy",
                    "owned_macro_recall",
                    "owned_min_class_recall",
                ]
            ]
            .rename(
                columns={
                    "owned_accuracy": "generic_owned_accuracy",
                    "owned_macro_recall": "generic_owned_macro_recall",
                    "owned_min_class_recall": "generic_owned_min_recall",
                }
            )
            .merge(
                self._init_post_df[
                    [
                        "model_id",
                        "selected_source",
                        "selected_learning_rate",
                        "selected_epoch",
                        "lr_reductions",
                        "owned_accuracy",
                        "owned_macro_recall",
                        "owned_min_class_recall",
                        "owned_accuracy_gain",
                        "owned_macro_recall_gain",
                        "specialization_elapsed_sec",
                    ]
                ].rename(
                    columns={
                        "owned_accuracy": "specialist_owned_accuracy",
                        "owned_macro_recall": "specialist_owned_macro_recall",
                        "owned_min_class_recall": "specialist_owned_min_recall",
                    }
                ),
                on="model_id",
                how="left",
            )
        )
        self.log("\nPRIMARY SPECIALIZATION RESULT — OWNED CLASSES FIRST")
        self.display(self._init_owned_comparison)
        pre_planned = self._init_pre_class_df[self._init_pre_class_df["planned_owned_class"]][
            ["model_id", "context", "class_name", "evaluation_support", "class_accuracy_recall"]
        ].rename(columns={"class_accuracy_recall": "generic_class_accuracy"})
        post_planned = post_class_df[post_class_df["planned_owned_class"]][
            ["model_id", "class_name", "class_accuracy_recall"]
        ].rename(columns={"class_accuracy_recall": "specialist_class_accuracy"})
        self._init_planned_class_comparison = pre_planned.merge(
            post_planned, on=["model_id", "class_name"], how="left"
        )
        self._init_planned_class_comparison["class_accuracy_gain"] = (
            self._init_planned_class_comparison["specialist_class_accuracy"]
            - self._init_planned_class_comparison["generic_class_accuracy"]
        )
        self.log("\nPER-PLANNED-CLASS ACCURACY BEFORE AND AFTER SPECIALIZATION")
        self.display(self._init_planned_class_comparison)
        self.log("\nSPECIALIZATION EPOCH-BY-EPOCH DETAILS")
        self.display(self._init_search_df)
        self.log("\nSPECIALIZATION RUNTIME / MEMORY BY CLIENT")
        self.display(self._init_specialization_runtime_df)
        self.log("\nSECONDARY DIAGNOSTIC — OTHER/ALL CLASSES AFTER SPECIALIZATION")
        self.display(
            self._init_post_df[
                [
                    "model_id",
                    "other_accuracy_secondary",
                    "other_macro_recall_secondary",
                    "accuracy",
                    "macro_f1",
                ]
            ]
        )
        self.log("\n" + "=" * 118)
        self.log("STAGE 4 — BUILD INITIAL L2/L3 BY AGGREGATING SPECIALIZED CLIENT MODELS")
        self.log("=" * 118)
