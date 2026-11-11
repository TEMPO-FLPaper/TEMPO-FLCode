"""Reference-derived reporting/final_hfl. See README.md."""

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'hfl': 26}
    def write_final_report(self):
        if self.STATE["current_round"] >= self.CONFIG["num_rounds"]:
            generic_weights = self.load_weights_from_file(
                self.DIRS["config"] / "generic_seed_v12_2.weights.h5"
            )
            final_general_rows = self.evaluate_general_stage_rows(
                "Final",
                "Final_L1_After_100_FedAvg_Rounds",
                4,
                [
                    {
                        "level": "L1",
                        "model_id": cid,
                        "context": self.CLIENT_TO_CONTEXT[cid],
                        "weights": self.STATE["l1_base_weights"][cid],
                        "temperature": 1.0,
                    }
                    for cid in self.CLIENT_IDS
                ],
            )
            final_general_rows += self.evaluate_general_stage_rows(
                "Final",
                "Final_L2_After_100_FedAvg_Rounds",
                5,
                [
                    {
                        "level": "L2",
                        "model_id": context_name,
                        "context": context_name,
                        "weights": self.STATE["l2_base_weights"][context_name],
                        "temperature": 1.0,
                    }
                    for context_name in self.CONFIG["context_names"]
                ],
            )
            final_general_rows += self.evaluate_general_stage_rows(
                "Final",
                "Final_L3_After_100_FedAvg_Rounds",
                6,
                [
                    {
                        "level": "L3",
                        "model_id": "Campus",
                        "context": "Campus",
                        "weights": self.STATE["l3_base_weights"],
                        "temperature": 1.0,
                    }
                ],
            )
            final_owned_rows = self.evaluate_owned_stage_rows(
                "Final",
                "Final_L1_After_100_FedAvg_Rounds",
                4,
                [
                    {
                        "client_id": cid,
                        "model_id": cid,
                        "evaluated_model_level": "L1",
                        "weights": self.STATE["l1_base_weights"][cid],
                        "temperature": 1.0,
                    }
                    for cid in self.CLIENT_IDS
                ],
            )
            generic_general_rows = self.evaluate_general_stage_rows(
                "Generic",
                "Generic_L3_Small_Generic_Model",
                0,
                [
                    {
                        "level": "L3",
                        "model_id": "Generic",
                        "context": "Campus",
                        "weights": generic_weights,
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
                        "weights": generic_weights,
                        "temperature": 1.0,
                    }
                    for cid in self.CLIENT_IDS
                ],
            )
            self.upsert_stage_metric_rows(
                self.STATE, "general_accuracy_rows", generic_general_rows + final_general_rows
            )
            self.upsert_stage_metric_rows(
                self.STATE, "owned_accuracy_rows", generic_owned_rows + final_owned_rows
            )
            general_acc_df = (
                pd.DataFrame(self.STATE["general_accuracy_rows"])
                .sort_values(["stage_order", "level", "model_id"])
                .reset_index(drop=True)
            )
            owned_acc_df = (
                pd.DataFrame(self.STATE["owned_accuracy_rows"])
                .sort_values(["stage_order", "client_id"])
                .reset_index(drop=True)
            )
            general_stage_summary_df = general_acc_df.groupby(
                ["stage", "stage_detail", "level"], as_index=False
            ).agg(
                models=("model_id", "count"),
                test_samples=("test_samples", "first"),
                mean_accuracy=("accuracy", "mean"),
                mean_macro_f1=("macro_f1", "mean"),
            )
            owned_stage_summary_df = owned_acc_df.groupby(
                ["stage", "stage_detail"], as_index=False
            ).agg(
                client_count=("client_id", "nunique"),
                mean_owned_accuracy=("owned_accuracy", "mean"),
                mean_owned_macro_f1=("owned_macro_f1", "mean"),
                min_owned_accuracy=("owned_accuracy", "min"),
                min_owned_macro_f1=("owned_macro_f1", "min"),
            )
            owned_test_support_df = pd.DataFrame(
                [
                    {
                        "client_id": cid,
                        "context": self.CLIENT_TO_CONTEXT[cid],
                        "planned_classes": ",".join(self.CLIENT_PLANNED_CLASSES[cid]),
                        "owned_class_test_samples": len(self.owned_class_test_set[cid]),
                        "planned_class_test_samples": int(
                            self.owned_class_test_set[cid]["label"]
                            .isin(self.CLIENT_PLANNED_CLASSES[cid])
                            .sum()
                        ),
                    }
                    for cid in self.CLIENT_IDS
                ]
            )
            test_set_definition_df = pd.DataFrame(
                [
                    {
                        "test_set": "general_test_set",
                        "source": "FinalTest_15pct_UNTOUCHED",
                        "scope": "One common all-class set for Generic, L1, L2, and L3",
                        "total_samples": int(len(self.general_test_set)),
                        "used_for_training_or_aggregation": False,
                    },
                    {
                        "test_set": "owned_class_test_set",
                        "source": "Disjoint client partition of FinalTest_15pct_UNTOUCHED",
                        "scope": "One different partition per client; planned classes are primary",
                        "total_samples": int(
                            sum((len(self.owned_class_test_set[cid]) for cid in self.CLIENT_IDS))
                        ),
                        "used_for_training_or_aggregation": False,
                    },
                ]
            )
            percent_columns = {
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "ece",
                "mean_accuracy",
                "mean_macro_f1",
                "owned_accuracy",
                "owned_macro_f1",
                "owned_macro_recall",
                "owned_min_class_recall",
                "mean_owned_accuracy",
                "mean_owned_macro_f1",
                "min_owned_accuracy",
                "min_owned_macro_f1",
                "mean_local_accuracy",
                "validation_accuracy",
                "validation_macro_f1",
            }
            general_acc_path = self.DIRS["master_metrics"] / "general_acc.xlsx"
            owned_acc_path = self.DIRS["master_metrics"] / "owned_acc.xlsx"
            self.write_v12_2_metric_workbook(
                general_acc_path,
                {
                    "General_Accuracy": general_acc_df,
                    "Stage_Level_Summary": general_stage_summary_df,
                    "Test_Set_Definition": test_set_definition_df,
                },
                percent_columns,
            )
            self.write_v12_2_metric_workbook(
                owned_acc_path,
                {
                    "Owned_Accuracy": owned_acc_df,
                    "Client_Stage_Summary": owned_stage_summary_df,
                    "Owned_Test_Support": owned_test_support_df,
                    "Test_Set_Definition": test_set_definition_df,
                },
                percent_columns,
            )
            self.log("\nFEDAVG V12.2 GENERAL ACCURACY/MACRO-F1")
            self.display(
                general_acc_df[
                    [
                        "stage",
                        "stage_detail",
                        "level",
                        "model_id",
                        "test_set",
                        "test_samples",
                        "accuracy",
                        "macro_f1",
                    ]
                ]
            )
            self.log("\nFEDAVG V12.2 OWNED ACCURACY/MACRO-F1 — L1 CLIENTS")
            self.display(
                owned_acc_df[
                    [
                        "stage",
                        "stage_detail",
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
            self.log("\nSeparate FedAvg workbooks saved:")
            self.log("  general_acc:", general_acc_path)
            self.log("  owned_acc:", owned_acc_path)
            round_history_df = pd.DataFrame(self.STATE["round_history"])
            communication_df = pd.DataFrame(self.STATE["communication_history"])
            fedavg_history_path = (
                self.DIRS["master_metrics"] / "TEMPO_FL_V12_2_FedAvg_ROUND_HISTORY.xlsx"
            )
            with pd.ExcelWriter(fedavg_history_path, engine="openpyxl") as writer:
                round_history_df.to_excel(writer, sheet_name="Round_History", index=False)
                communication_df.to_excel(writer, sheet_name="Communication", index=False)
            self.log("  round_history:", fedavg_history_path)
            self.report_tables(
                {
                    "general_stage_summary_df": general_stage_summary_df,
                    "owned_stage_summary_df": owned_stage_summary_df,
                    "owned_test_support_df": owned_test_support_df,
                }
            )
        else:
            self.log(
                f"Final General/Owned test evaluation is deferred until round 100. Current round={self.STATE['current_round']:03d}."
            )
