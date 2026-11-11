"""Reference-derived reporting/final_ditto. See README.md."""

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'ditto': 27}
    def write_final_report(self):
        if self.STATE["current_round"] >= self.CONFIG["num_rounds"]:
            generic_weights = self.load_weights_from_file(
                self.DIRS["config"] / "generic_seed_v12_2.weights.h5"
            )
            global_final_spec = {
                "level": "Global",
                "model_type": "Global",
                "model_id": "Global",
                "context": "AllClients",
                "weights": self.STATE["global_weights"],
                "temperature": 1.0,
            }
            personalized_final_specs = [
                {
                    "level": "Personalized",
                    "model_type": "Personalized",
                    "model_id": cid,
                    "client_id": cid,
                    "context": self.CLIENT_TO_CONTEXT[cid],
                    "weights": self.STATE["personalized_weights"][cid],
                    "temperature": 1.0,
                }
                for cid in self.CLIENT_IDS
            ]
            final_general_rows = self.evaluate_general_stage_rows(
                "Final",
                "Final_After_100_Ditto_Rounds",
                2,
                [global_final_spec] + personalized_final_specs,
            )
            final_owned_specs = [
                {
                    "client_id": cid,
                    "model_id": "Global",
                    "model_type": "Global",
                    "level": "Global",
                    "weights": self.STATE["global_weights"],
                    "temperature": 1.0,
                }
                for cid in self.CLIENT_IDS
            ] + [
                {
                    "client_id": cid,
                    "model_id": cid,
                    "model_type": "Personalized",
                    "level": "Personalized",
                    "weights": self.STATE["personalized_weights"][cid],
                    "temperature": 1.0,
                }
                for cid in self.CLIENT_IDS
            ]
            final_owned_rows = self.evaluate_owned_stage_rows(
                "Final", "Final_After_100_Ditto_Rounds", 2, final_owned_specs
            )
            generic_general_rows = self.evaluate_general_stage_rows(
                "Generic",
                "Generic_Global_Model",
                0,
                [
                    {
                        "level": "Global",
                        "model_type": "Global",
                        "model_id": "Global",
                        "context": "AllClients",
                        "weights": generic_weights,
                        "temperature": 1.0,
                    }
                ],
            )
            generic_owned_rows = self.evaluate_owned_stage_rows(
                "Generic",
                "Generic_Global_Model_At_Each_Client",
                0,
                [
                    {
                        "client_id": cid,
                        "model_id": "Global",
                        "model_type": "Global",
                        "level": "Global",
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
            general_acc_df = pd.DataFrame(self.STATE["general_accuracy_rows"]).copy()
            owned_acc_df = pd.DataFrame(self.STATE["owned_accuracy_rows"]).copy()
            model_order = {"Global": 0, "Personalized": 1}
            general_acc_df["_model_order"] = (
                general_acc_df["model_type"].map(model_order).fillna(99)
            )
            owned_acc_df["_model_order"] = owned_acc_df["model_type"].map(model_order).fillna(99)
            general_acc_df = (
                general_acc_df.sort_values(["stage_order", "_model_order", "model_id", "client_id"])
                .drop(columns=["_model_order"])
                .reset_index(drop=True)
            )
            owned_acc_df = (
                owned_acc_df.sort_values(["stage_order", "client_id", "_model_order", "model_id"])
                .drop(columns=["_model_order"])
                .reset_index(drop=True)
            )
            general_stage_summary_df = general_acc_df.groupby(
                ["stage", "stage_detail", "model_type"], as_index=False
            ).agg(
                models=("model_id", "count"),
                test_samples=("test_samples", "first"),
                mean_accuracy=("accuracy", "mean"),
                mean_macro_f1=("macro_f1", "mean"),
                min_accuracy=("accuracy", "min"),
                min_macro_f1=("macro_f1", "min"),
            )
            owned_stage_summary_df = owned_acc_df.groupby(
                ["stage", "stage_detail", "model_type"], as_index=False
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
                        "scope": "One common all-class set for Global and Personalized models",
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
                "min_accuracy",
                "min_macro_f1",
                "owned_accuracy",
                "owned_macro_f1",
                "owned_macro_recall",
                "owned_min_class_recall",
                "mean_owned_accuracy",
                "mean_owned_macro_f1",
                "min_owned_accuracy",
                "min_owned_macro_f1",
                "mean_global_local_accuracy",
                "mean_personal_local_accuracy",
                "global_validation_accuracy",
                "global_validation_macro_f1",
                "personalized_validation_mean_accuracy",
                "personalized_validation_mean_macro_f1",
                "personalized_validation_min_accuracy",
                "personalized_validation_min_macro_f1",
            }
            general_acc_path = self.DIRS["master_metrics"] / "general_acc.xlsx"
            owned_acc_path = self.DIRS["master_metrics"] / "owned_acc.xlsx"
            self.write_v12_2_metric_workbook(
                general_acc_path,
                {
                    "General_Accuracy": general_acc_df,
                    "Stage_Model_Summary": general_stage_summary_df,
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
            self.log("\nDITTO V12.2 GENERAL TEST — ACCURACY / MACRO-F1")
            self.display(
                general_acc_df[
                    [
                        "stage",
                        "stage_detail",
                        "model_type",
                        "model_id",
                        "client_id",
                        "test_set",
                        "test_samples",
                        "accuracy",
                        "macro_f1",
                    ]
                ]
            )
            self.log("\nDITTO V12.2 OWNED TEST — ACCURACY / MACRO-F1")
            self.display(
                owned_acc_df[
                    [
                        "stage",
                        "stage_detail",
                        "model_type",
                        "model_id",
                        "client_id",
                        "context",
                        "planned_classes",
                        "test_samples",
                        "owned_test_samples",
                        "accuracy",
                        "macro_f1",
                    ]
                ]
            )
            self.log("\nSeparate Ditto workbooks saved:")
            self.log("  general_acc:", general_acc_path)
            self.log("  owned_acc:", owned_acc_path)
            round_history_df = pd.DataFrame(self.STATE["round_history"])
            communication_df = pd.DataFrame(self.STATE["communication_history"])
            ditto_history_path = (
                self.DIRS["master_metrics"] / "TEMPO_FL_V12_2_Ditto_ROUND_HISTORY.xlsx"
            )
            with pd.ExcelWriter(ditto_history_path, engine="openpyxl") as writer:
                round_history_df.to_excel(writer, sheet_name="Round_History", index=False)
                communication_df.to_excel(writer, sheet_name="Communication", index=False)
            self.log("  round_history:", ditto_history_path)
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
