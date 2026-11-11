"""Reference-derived training/bootstrap_hierarchy_tempo. See README.md."""

import gc

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'tempo': 29}
    def bootstrap_hierarchy(self):
        l2_bootstrap_rows = []
        l2_initial_weights = {}
        l2_context_quality = {}
        for context_name in self.CONFIG["context_names"]:
            cids = self.CONTEXT_TO_CLIENTS[context_name]
            specialist_weights = [self._init_state["l1_base_weights"][cid] for cid in cids]
            uniform_candidate = self.weighted_average_weights(
                specialist_weights, [1.0 for _ in cids]
            )
            quality_weights = [
                self.client_owned_quality_weight(cid, self._init_post_owned_metrics[cid])
                for cid in cids
            ]
            quality_candidate = self.weighted_average_weights(specialist_weights, quality_weights)
            context_gate_df = self.context_dataframe(self.client_gate_local, context_name)
            uniform_metrics = self.evaluate_weights(
                uniform_candidate, context_gate_df, temperature=1.0
            )
            quality_metrics = self.evaluate_weights(
                quality_candidate, context_gate_df, temperature=1.0
            )
            uniform_score = self.model_gate_score(uniform_metrics)
            quality_score = self.model_gate_score(quality_metrics)
            if (
                self.CONFIG["bootstrap_use_owned_quality_weights"]
                and quality_score >= uniform_score
            ):
                selected_weights = [w.copy() for w in quality_candidate]
                selected_method = "owned_quality_weighted_specialists"
                selected_gate_metrics = quality_metrics
            else:
                selected_weights = [w.copy() for w in uniform_candidate]
                selected_method = "uniform_specialist_average"
                selected_gate_metrics = uniform_metrics
            l2_initial_weights[context_name] = selected_weights
            context_report_df = self.context_dataframe(self.client_val_local, context_name)
            generic_context_metrics = self.evaluate_weights(
                self._init_generic_weights, context_report_df, temperature=1.0
            )
            l2_report_metrics = self.evaluate_weights(
                selected_weights, context_report_df, temperature=1.0
            )
            l2_context_quality[context_name] = max(
                self.model_gate_score(selected_gate_metrics), self.CONFIG["bootstrap_quality_floor"]
            )
            row = {
                "level": "L2",
                "model_id": context_name,
                "aggregation_method": selected_method,
                "clients": ",".join(cids),
                "generic_accuracy_same_scope": generic_context_metrics["accuracy"],
                "l2_accuracy": l2_report_metrics["accuracy"],
                "accuracy_gain_vs_generic": l2_report_metrics["accuracy"]
                - generic_context_metrics["accuracy"],
                "generic_macro_f1_same_scope": generic_context_metrics["macro_f1"],
                "l2_macro_f1": l2_report_metrics["macro_f1"],
                "macro_f1_gain_vs_generic": l2_report_metrics["macro_f1"]
                - generic_context_metrics["macro_f1"],
                "generic_balanced_accuracy_same_scope": generic_context_metrics[
                    "balanced_accuracy"
                ],
                "l2_balanced_accuracy": l2_report_metrics["balanced_accuracy"],
            }
            l2_bootstrap_rows.append(row)
        l2_bootstrap_df = pd.DataFrame(l2_bootstrap_rows)
        self.log("\nINITIAL L2 MODELS — SPECIALIST AGGREGATION VS SMALL GENERIC MODEL")
        self.display(l2_bootstrap_df)
        l2_list = [
            l2_initial_weights[context_name] for context_name in self.CONFIG["context_names"]
        ]
        uniform_l3 = self.weighted_average_weights(
            l2_list, [1.0 for _ in self.CONFIG["context_names"]]
        )
        context_sample_counts = [
            sum((len(self.client_train[cid]) for cid in self.CONTEXT_TO_CLIENTS[context_name]))
            for context_name in self.CONFIG["context_names"]
        ]
        quality_l3, effective_l3_weights = self.quality_weighted_average_weights(
            l2_list,
            context_sample_counts,
            [l2_context_quality[context_name] for context_name in self.CONFIG["context_names"]],
            power=self.CONFIG["bootstrap_l3_context_quality_power"],
        )
        uniform_l3_gate = self.evaluate_weights(
            uniform_l3, self.global_model_gate_df, temperature=1.0
        )
        quality_l3_gate = self.evaluate_weights(
            quality_l3, self.global_model_gate_df, temperature=1.0
        )
        if self.model_gate_score(quality_l3_gate) >= self.model_gate_score(uniform_l3_gate):
            selected_l3 = [w.copy() for w in quality_l3]
            l3_method = "quality_weighted_L2_aggregation"
        else:
            selected_l3 = [w.copy() for w in uniform_l3]
            l3_method = "uniform_L2_aggregation"
        generic_global_metrics = self.evaluate_weights(
            self._init_generic_weights, self.global_validation_df, temperature=1.0
        )
        l3_global_metrics = self.evaluate_weights(
            selected_l3, self.global_validation_df, temperature=1.0
        )
        l3_bootstrap_df = pd.DataFrame(
            [
                {
                    "level": "L3",
                    "model_id": "Campus",
                    "aggregation_method": l3_method,
                    "generic_accuracy_same_scope": generic_global_metrics["accuracy"],
                    "l3_accuracy": l3_global_metrics["accuracy"],
                    "accuracy_gain_vs_generic": l3_global_metrics["accuracy"]
                    - generic_global_metrics["accuracy"],
                    "generic_macro_f1_same_scope": generic_global_metrics["macro_f1"],
                    "l3_macro_f1": l3_global_metrics["macro_f1"],
                    "macro_f1_gain_vs_generic": l3_global_metrics["macro_f1"]
                    - generic_global_metrics["macro_f1"],
                    "generic_balanced_accuracy_same_scope": generic_global_metrics[
                        "balanced_accuracy"
                    ],
                    "l3_balanced_accuracy": l3_global_metrics["balanced_accuracy"],
                    "balanced_accuracy_gain_vs_generic": l3_global_metrics["balanced_accuracy"]
                    - generic_global_metrics["balanced_accuracy"],
                }
            ]
        )
        self.log("\nINITIAL L3 CAMPUS — SPECIALIST HIERARCHICAL AGGREGATION VS SMALL GENERIC MODEL")
        self.display(l3_bootstrap_df)
        if l3_global_metrics["macro_f1"] <= generic_global_metrics["macro_f1"]:
            self.log(
                "[Hierarchy warning] Initial L3 Macro-F1 did not exceed the small generic model. V12 reports this honestly; the 100-round HFL stage must demonstrate whether federated learning subsequently improves it."
            )
        self._init_state["l2_base_weights"] = {
            context_name: [w.copy() for w in l2_initial_weights[context_name]]
            for context_name in self.CONFIG["context_names"]
        }
        self._init_state["l3_base_weights"] = [w.copy() for w in selected_l3]
        initial_l2_general_rows = self.evaluate_general_stage_rows(
            "Initial",
            "Initial_L2_After_Specialist_Aggregation",
            2,
            [
                {
                    "level": "L2",
                    "model_id": context_name,
                    "context": context_name,
                    "weights": self._init_state["l2_base_weights"][context_name],
                    "temperature": 1.0,
                }
                for context_name in self.CONFIG["context_names"]
            ],
        )
        initial_l3_general_rows = self.evaluate_general_stage_rows(
            "Initial",
            "Initial_L3_After_L2_Aggregation",
            3,
            [
                {
                    "level": "L3",
                    "model_id": "Campus",
                    "context": "Campus",
                    "weights": self._init_state["l3_base_weights"],
                    "temperature": 1.0,
                }
            ],
        )
        self.upsert_stage_metric_rows(
            self._init_state,
            "general_accuracy_rows",
            initial_l2_general_rows + initial_l3_general_rows,
        )
        self.log("\nV12.2 GENERAL TEST — INITIAL L2 MODELS")
        self.display(
            pd.DataFrame(initial_l2_general_rows)[
                ["level", "model_id", "context", "test_samples", "accuracy", "macro_f1"]
            ]
        )
        self.log("\nV12.2 GENERAL TEST — INITIAL L3 MODEL")
        self.display(
            pd.DataFrame(initial_l3_general_rows)[
                ["level", "model_id", "context", "test_samples", "accuracy", "macro_f1"]
            ]
        )
        for context_name in self.CONFIG["context_names"]:
            self.save_weights_to_file(
                self._init_state["l2_base_weights"][context_name],
                self.DIRS["config"] / f"initial_l2_{context_name}_v12_2.weights.h5",
            )
        self.save_weights_to_file(
            self._init_state["l3_base_weights"],
            self.DIRS["config"] / "initial_l3_aggregated_v12_2.weights.h5",
        )
        for context_name in self.CONFIG["context_names"]:
            self._init_state["l2_version"][context_name] = 1
        self._init_state["l3_version"] = 1
        for cid in self.CLIENT_IDS:
            context_name = self.CLIENT_TO_CONTEXT[cid]
            self._init_state["shadow_l2_weights"][cid] = [
                w.copy() for w in self._init_state["l2_base_weights"][context_name]
            ]
            self._init_state["shadow_l3_weights"][cid] = [
                w.copy() for w in self._init_state["l3_base_weights"]
            ]
            self._init_state["shadow_l2_version"][cid] = 1
            self._init_state["shadow_l3_version"][cid] = 1
        self.refresh_all_gates(self._init_state, round_id=0)
        self._init_state["l2_best_weights"] = {}
        hierarchy_initial_rows = []
        for context_name in self.CONFIG["context_names"]:
            context_gate_df = self.context_dataframe(self.client_gate_local, context_name)
            metrics = self.evaluate_weights(
                self._init_state["l2_base_weights"][context_name],
                context_gate_df,
                self._init_state["l2_temperature"][context_name],
            )
            score = self.model_gate_score(metrics)
            self._init_state["l2_best_score"][context_name] = score
            self._init_state["l2_best_weights"][context_name] = [
                w.copy() for w in self._init_state["l2_base_weights"][context_name]
            ]
            hierarchy_initial_rows.append(
                {
                    "level": "L2",
                    "model_id": context_name,
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "gate_score": score,
                }
            )
        l3_gate_metrics = self.evaluate_weights(
            self._init_state["l3_base_weights"],
            self.global_model_gate_df,
            self._init_state["l3_temperature"],
        )
        self._init_state["l3_best_score"] = self.model_gate_score(l3_gate_metrics)
        self._init_state["l3_best_weights"] = [
            w.copy() for w in self._init_state["l3_base_weights"]
        ]
        self._init_state["initial_l2_best_score"] = {
            context_name: float(self._init_state["l2_best_score"][context_name])
            for context_name in self.CONFIG["context_names"]
        }
        self._init_state["initial_l3_best_score"] = float(self._init_state["l3_best_score"])
        hierarchy_initial_rows.append(
            {
                "level": "L3",
                "model_id": "Campus",
                "macro_f1": l3_gate_metrics["macro_f1"],
                "accuracy": l3_gate_metrics["accuracy"],
                "balanced_accuracy": l3_gate_metrics["balanced_accuracy"],
                "gate_score": self._init_state["l3_best_score"],
            }
        )
        self._init_state["hierarchy_initial_rows"] = list(hierarchy_initial_rows)
        self._init_state["v12_owned_specialization_comparison"] = (
            self._init_owned_comparison.to_dict("records")
        )
        self._init_state["v12_planned_class_comparison"] = (
            self._init_planned_class_comparison.to_dict("records")
        )
        self._init_state["v12_l2_bootstrap_rows"] = l2_bootstrap_df.to_dict("records")
        self._init_state["v12_l3_bootstrap_rows"] = l3_bootstrap_df.to_dict("records")
        initial_workbook = self.DIRS["master_metrics"] / "V12_INITIAL_STAGES_AND_HIERARCHY.xlsx"
        with pd.ExcelWriter(initial_workbook, engine="openpyxl") as writer:
            self._init_generic_counts_table.to_excel(
                writer, sheet_name="Generic_Counts", index=False
            )
            pd.DataFrame([self._init_generic_internal_summary]).to_excel(
                writer, sheet_name="Generic_InternalTest", index=False
            )
            self._init_generic_internal_class_df.to_excel(
                writer, sheet_name="Generic_PerClass", index=False
            )
            self._init_pre_df.to_excel(writer, sheet_name="Before_Specialization", index=False)
            self._init_post_df.to_excel(writer, sheet_name="After_Specialization", index=False)
            self._init_owned_comparison.to_excel(writer, sheet_name="OwnedClass_Gain", index=False)
            self._init_planned_class_comparison.to_excel(
                writer, sheet_name="PlannedClass_Gain", index=False
            )
            self._init_search_df.to_excel(writer, sheet_name="Specialization_Search", index=False)
            self._init_specialization_runtime_df.to_excel(
                writer, sheet_name="Specialization_Memory", index=False
            )
            l2_bootstrap_df.to_excel(writer, sheet_name="Initial_L2_vs_Generic", index=False)
            l3_bootstrap_df.to_excel(writer, sheet_name="Initial_L3_vs_Generic", index=False)
            pd.DataFrame(hierarchy_initial_rows).to_excel(
                writer, sheet_name="Hierarchy_Round0", index=False
            )
        round_tables = {
            "GenericCounts": self._init_generic_counts_table,
            "GenericPerClass": self._init_generic_internal_class_df,
            "OwnedSpecialization": self._init_owned_comparison,
            "PlannedClassGain": self._init_planned_class_comparison,
            "SpecializationMemory": self._init_specialization_runtime_df,
            "InitialL2": l2_bootstrap_df,
            "InitialL3": l3_bootstrap_df,
            "Hierarchy": pd.DataFrame(hierarchy_initial_rows),
            "Thresholds": pd.DataFrame(self._init_state["threshold_diagnostics"]),
        }
        self._init_state["current_round"] = 0
        self.save_round_checkpoint(0, self._init_state, round_tables)
        gc.collect()
        self.print_memory_snapshot("[After V12 staged initialization]")
        self.log("\nV12 staged initialization is complete. The next cell starts federated round 1.")
        return self._init_state
