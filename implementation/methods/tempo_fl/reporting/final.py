"""Reference-derived reporting/final_tempo. See README.md."""

import math

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score

class Operations:
    # Reference cells (zero-based): {'tempo': 51}
    def write_final_report(self):
        if self.STATE["current_round"] >= self.CONFIG["num_rounds"]:
            self.log("\n" + "=" * 120)
            self.log("V12 FINAL EVALUATION AFTER 100 FEDERATED ROUNDS")
            self.log("=" * 120)
            generic_weights = self.load_weights_from_file(
                self.DIRS["config"] / "generic_seed_v12_2.weights.h5"
            )
            initial_l3_weights = self.load_weights_from_file(
                self.DIRS["config"] / "initial_l3_aggregated_v12_2.weights.h5"
            )
            final_general_rows = []
            final_general_rows.extend(
                self.evaluate_general_stage_rows(
                    "Final",
                    "Final_L1_After_100_HFL_Rounds",
                    4,
                    [
                        {
                            "level": "L1",
                            "model_id": cid,
                            "context": self.CLIENT_TO_CONTEXT[cid],
                            "weights": self.STATE["l1_base_weights"][cid],
                            "temperature": self.STATE["l1_temperature"][cid],
                        }
                        for cid in self.CLIENT_IDS
                    ],
                )
            )
            final_general_rows.extend(
                self.evaluate_general_stage_rows(
                    "Final",
                    "Final_L2_After_100_HFL_Rounds",
                    5,
                    [
                        {
                            "level": "L2",
                            "model_id": context_name,
                            "context": context_name,
                            "weights": self.STATE["l2_base_weights"][context_name],
                            "temperature": self.STATE["l2_temperature"][context_name],
                        }
                        for context_name in self.CONFIG["context_names"]
                    ],
                )
            )
            final_general_rows.extend(
                self.evaluate_general_stage_rows(
                    "Final",
                    "Final_L3_After_100_HFL_Rounds",
                    6,
                    [
                        {
                            "level": "L3",
                            "model_id": "Campus",
                            "context": "Campus",
                            "weights": self.STATE["l3_base_weights"],
                            "temperature": self.STATE["l3_temperature"],
                        }
                    ],
                )
            )
            final_owned_rows = self.evaluate_owned_stage_rows(
                "Final",
                "Final_L1_After_100_HFL_Rounds",
                4,
                [
                    {
                        "client_id": cid,
                        "model_id": cid,
                        "evaluated_model_level": "L1",
                        "weights": self.STATE["l1_base_weights"][cid],
                        "temperature": self.STATE["l1_temperature"][cid],
                    }
                    for cid in self.CLIENT_IDS
                ],
            )
            self.upsert_stage_metric_rows(self.STATE, "general_accuracy_rows", final_general_rows)
            self.upsert_stage_metric_rows(self.STATE, "owned_accuracy_rows", final_owned_rows)
            generic_l3_test = self.evaluate_weights(
                generic_weights, self.general_test_set, temperature=1.0
            )
            initial_l3_test = self.evaluate_weights(
                initial_l3_weights, self.general_test_set, temperature=1.0
            )
            final_l3_test = self.evaluate_weights(
                self.STATE["l3_base_weights"], self.general_test_set, self.STATE["l3_temperature"]
            )
            l3_progress_df = pd.DataFrame(
                [
                    {
                        "stage": "Small_Generic",
                        "accuracy": generic_l3_test["accuracy"],
                        "macro_f1": generic_l3_test["macro_f1"],
                        "balanced_accuracy": generic_l3_test["balanced_accuracy"],
                        "ece": generic_l3_test["ece"],
                    },
                    {
                        "stage": "Initial_L3_After_Specialist_Aggregation",
                        "accuracy": initial_l3_test["accuracy"],
                        "macro_f1": initial_l3_test["macro_f1"],
                        "balanced_accuracy": initial_l3_test["balanced_accuracy"],
                        "ece": initial_l3_test["ece"],
                    },
                    {
                        "stage": "Final_L3_After_100_HFL_Rounds",
                        "accuracy": final_l3_test["accuracy"],
                        "macro_f1": final_l3_test["macro_f1"],
                        "balanced_accuracy": final_l3_test["balanced_accuracy"],
                        "ece": final_l3_test["ece"],
                    },
                ]
            )
            l3_progress_df["accuracy_gain_vs_generic"] = (
                l3_progress_df["accuracy"] - generic_l3_test["accuracy"]
            )
            l3_progress_df["macro_f1_gain_vs_generic"] = (
                l3_progress_df["macro_f1"] - generic_l3_test["macro_f1"]
            )
            self.log("\nL3 ALL-CLASS PROGRESS — GENERIC -> INITIAL AGGREGATION -> ROUND 100")
            self.display(l3_progress_df)

            def class_recall_map(weights, df, temperature):
                rows = self.classwise_accuracy_rows(
                    weights,
                    df,
                    "final_compare",
                    "model",
                    "Global",
                    temperature=temperature,
                    train_reference_df=None,
                )
                return {
                    row["class_name"]: {
                        "support": row["evaluation_support"],
                        "recall": row["class_accuracy_recall"],
                    }
                    for row in rows
                }

            generic_class_map = class_recall_map(generic_weights, self.general_test_set, 1.0)
            initial_l3_class_map = class_recall_map(initial_l3_weights, self.general_test_set, 1.0)
            final_l3_class_map = class_recall_map(
                self.STATE["l3_base_weights"], self.general_test_set, self.STATE["l3_temperature"]
            )
            l3_per_class_rows = []
            for class_name in self.CLASS_NAMES:
                l3_per_class_rows.append(
                    {
                        "class_name": class_name,
                        "test_support": generic_class_map[class_name]["support"],
                        "generic_recall": generic_class_map[class_name]["recall"],
                        "initial_l3_recall": initial_l3_class_map[class_name]["recall"],
                        "final_l3_recall": final_l3_class_map[class_name]["recall"],
                        "initial_l3_gain_vs_generic": initial_l3_class_map[class_name]["recall"]
                        - generic_class_map[class_name]["recall"],
                        "final_l3_gain_vs_generic": final_l3_class_map[class_name]["recall"]
                        - generic_class_map[class_name]["recall"],
                    }
                )
            l3_per_class_df = pd.DataFrame(l3_per_class_rows)
            self.log("\nL3 PER-CLASS ACCURACY/RECALL — ALL SIX CLASSES")
            self.display(l3_per_class_df)
            l2_progress_rows = []
            for context_name in self.CONFIG["context_names"]:
                context_test_df = self.context_dataframe(self.client_test_local, context_name)
                initial_l2_weights = self.load_weights_from_file(
                    self.DIRS["config"] / f"initial_l2_{context_name}_v12_2.weights.h5"
                )
                generic_metrics = self.evaluate_weights(
                    generic_weights, context_test_df, temperature=1.0
                )
                initial_metrics = self.evaluate_weights(
                    initial_l2_weights, context_test_df, temperature=1.0
                )
                final_metrics = self.evaluate_weights(
                    self.STATE["l2_base_weights"][context_name],
                    context_test_df,
                    self.STATE["l2_temperature"][context_name],
                )
                for stage_name, metrics in [
                    ("Small_Generic", generic_metrics),
                    ("Initial_L2_After_Specialist_Aggregation", initial_metrics),
                    ("Final_L2_After_100_HFL_Rounds", final_metrics),
                ]:
                    l2_progress_rows.append(
                        {
                            "context": context_name,
                            "stage": stage_name,
                            "test_samples": len(context_test_df),
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "accuracy_gain_vs_generic": metrics["accuracy"]
                            - generic_metrics["accuracy"],
                            "macro_f1_gain_vs_generic": metrics["macro_f1"]
                            - generic_metrics["macro_f1"],
                        }
                    )
            l2_progress_df = pd.DataFrame(l2_progress_rows)
            self.log("\nL2 CONTEXT PROGRESS — GENERIC -> INITIAL AGGREGATION -> ROUND 100")
            self.display(l2_progress_df)
            promotion_df = pd.DataFrame(self.STATE["promotion_history"])
            client_success_rows = []
            final_planned_class_rows = []
            final_other_class_rows = []
            for cid in self.CLIENT_IDS:
                context_name = self.CLIENT_TO_CONTEXT[cid]
                test_df = self.client_test_local[cid].copy().reset_index(drop=True)
                initial_l1_weights = self.load_weights_from_file(
                    self.DIRS["config"] / f"initial_l1_{cid}_v12_2.weights.h5"
                )
                generic_owned = self.evaluate_owned_specialist(
                    generic_weights, cid, test_df, temperature=1.0
                )
                initial_owned = self.evaluate_owned_specialist(
                    initial_l1_weights, cid, test_df, temperature=1.0
                )
                final_owned = self.evaluate_owned_specialist(
                    self.STATE["l1_base_weights"][cid],
                    cid,
                    test_df,
                    temperature=self.STATE["l1_temperature"][cid],
                )
                y_true = test_df["label_id"].to_numpy(dtype=np.int32)
                cascade = self.cascade_predict_df(
                    cid, test_df, self.CONFIG["num_rounds"], self.STATE, collect_teachers=False
                )
                operational_pred = cascade["operational_pred"]
                unknown_mask = cascade["unknown_mask"]
                operational_accuracy = float(accuracy_score(y_true, operational_pred))
                operational_macro_f1 = self.fixed_label_macro_f1(y_true, operational_pred)
                counters = self.STATE["client_operational_counters"][cid]
                attempts = 0
                if len(promotion_df) > 0 and "client_id" in promotion_df.columns:
                    attempts = int(np.sum(promotion_df["client_id"] == cid))
                profile = self.STATE["shift_profiles"].get(cid) or {}
                client_success_rows.append(
                    {
                        "client_id": cid,
                        "context": context_name,
                        "planned_classes": ",".join(self.CLIENT_PLANNED_CLASSES[cid]),
                        "shift_role": "Affected" if profile.get("eligible", False) else "Control",
                        "generic_owned_test_accuracy": generic_owned["owned_accuracy"],
                        "initial_specialist_owned_test_accuracy": initial_owned["owned_accuracy"],
                        "final_l1_owned_test_accuracy": final_owned["owned_accuracy"],
                        "initial_owned_accuracy_gain_vs_generic": initial_owned["owned_accuracy"]
                        - generic_owned["owned_accuracy"],
                        "final_owned_accuracy_gain_vs_generic": final_owned["owned_accuracy"]
                        - generic_owned["owned_accuracy"],
                        "generic_owned_macro_recall": generic_owned["owned_macro_recall"],
                        "initial_specialist_owned_macro_recall": initial_owned[
                            "owned_macro_recall"
                        ],
                        "final_l1_owned_macro_recall": final_owned["owned_macro_recall"],
                        "initial_owned_macro_recall_gain": initial_owned["owned_macro_recall"]
                        - generic_owned["owned_macro_recall"],
                        "final_owned_macro_recall_gain": final_owned["owned_macro_recall"]
                        - generic_owned["owned_macro_recall"],
                        "final_owned_min_class_recall": final_owned["owned_min_class_recall"],
                        "final_other_accuracy_secondary": final_owned["other_accuracy_secondary"],
                        "final_other_macro_recall_secondary": final_owned[
                            "other_macro_recall_secondary"
                        ],
                        "final_all_local_accuracy_secondary": final_owned[
                            "all_local_accuracy_secondary"
                        ],
                        "final_all_local_macro_f1_secondary": final_owned[
                            "all_local_macro_f1_secondary"
                        ],
                        "final_operational_cascade_accuracy": operational_accuracy,
                        "final_operational_cascade_macro_f1": operational_macro_f1,
                        "final_operational_coverage": float((~unknown_mask).mean()),
                        "final_operational_unknown_rate": float(unknown_mask.mean()),
                        "total_operational_events": int(counters["total_events"]),
                        "specialist_handled_events": int(counters["specialist_accepted_events"]),
                        "shadow_invocations": int(counters["shadow_invocations"]),
                        "shadow_replacements": int(counters["shadow_replacements"]),
                        "shadow_correct_replacements_audit": int(
                            counters["shadow_correct_replacements_audit"]
                        ),
                        "l2_shadow_replacements": int(counters["l2_replacements"]),
                        "l3_shadow_replacements": int(counters["l3_replacements"]),
                        "unknown_events": int(counters["unknown_events"]),
                        "rounds_specialist_only": int(counters["rounds_specialist_only"]),
                        "rounds_shadow_invoked": int(counters["rounds_shadow_invoked"]),
                        "rounds_shadow_replaced_specialist": int(
                            counters["rounds_shadow_replaced"]
                        ),
                        "rounds_shadow_effective_audit": int(
                            counters["rounds_shadow_effective_audit"]
                        ),
                        "promotion_attempts": attempts,
                        "accepted_permanent_promotions": int(self.STATE["promotion_count"][cid]),
                    }
                )
                all_final_class_rows = self.classwise_accuracy_rows(
                    self.STATE["l1_base_weights"][cid],
                    test_df,
                    "Final_L1_Test",
                    cid,
                    context_name,
                    temperature=self.STATE["l1_temperature"][cid],
                    train_reference_df=self.client_train[cid],
                )
                for row in all_final_class_rows:
                    row = dict(row)
                    row["planned_owned_class"] = (
                        row["class_name"] in self.CLIENT_PLANNED_CLASSES[cid]
                    )
                    if row["planned_owned_class"]:
                        final_planned_class_rows.append(row)
                    else:
                        final_other_class_rows.append(row)
            client_success_df = pd.DataFrame(client_success_rows)
            final_planned_class_df = pd.DataFrame(final_planned_class_rows)
            final_other_class_df = pd.DataFrame(final_other_class_rows)
            self.log("\nPRIMARY FINAL LOW-LEVEL RESULT — PLANNED/OWNED CLASSES")
            self.display(
                client_success_df[
                    [
                        "client_id",
                        "context",
                        "planned_classes",
                        "generic_owned_test_accuracy",
                        "initial_specialist_owned_test_accuracy",
                        "final_l1_owned_test_accuracy",
                        "initial_owned_accuracy_gain_vs_generic",
                        "final_owned_accuracy_gain_vs_generic",
                        "final_owned_min_class_recall",
                    ]
                ]
            )
            self.log("\nFINAL PER-PLANNED-CLASS L1 ACCURACY")
            self.display(
                final_planned_class_df[
                    [
                        "model_id",
                        "context",
                        "class_name",
                        "evaluation_support",
                        "class_accuracy_recall",
                    ]
                ]
            )
            self.log("\nSECONDARY FINAL LOW-LEVEL DIAGNOSTICS — OTHER AND ALL LOCAL CLASSES")
            self.display(
                client_success_df[
                    [
                        "client_id",
                        "final_other_accuracy_secondary",
                        "final_other_macro_recall_secondary",
                        "final_all_local_accuracy_secondary",
                        "final_all_local_macro_f1_secondary",
                    ]
                ]
            )
            self.log("\nTEMPO-FL SPECIALIST / SHADOW USAGE BY CLIENT")
            self.display(
                client_success_df[
                    [
                        "client_id",
                        "specialist_handled_events",
                        "shadow_invocations",
                        "shadow_replacements",
                        "shadow_correct_replacements_audit",
                        "l2_shadow_replacements",
                        "l3_shadow_replacements",
                        "unknown_events",
                        "rounds_specialist_only",
                        "rounds_shadow_invoked",
                        "rounds_shadow_replaced_specialist",
                        "accepted_permanent_promotions",
                    ]
                ]
            )
            client_history_df = pd.DataFrame(self.STATE["client_round_history"])
            phase_rows = []
            for phase_name, phase_df in client_history_df.groupby("scenario"):
                eval_part = phase_df[phase_df["evaluation_round"] == True]
                phase_rows.append(
                    {
                        "scenario": phase_name,
                        "round_min": int(phase_df["round"].min()),
                        "round_max": int(phase_df["round"].max()),
                        "mean_reliance": float(phase_df["l1_shadow_reliance_rate"].mean()),
                        "mean_higher_resolution": float(phase_df["higher_resolution_rate"].mean()),
                        "mean_unknown_rate": float(phase_df["unknown_rate"].mean()),
                        "mean_l1_eval_macro_f1": float(
                            eval_part["l1_only_operational_macro_f1"].mean()
                        )
                        if len(eval_part) > 0
                        else np.nan,
                        "mean_cascade_eval_macro_f1": float(
                            eval_part["operational_macro_f1"].mean()
                        )
                        if len(eval_part) > 0
                        else np.nan,
                    }
                )
            phase_summary_df = pd.DataFrame(phase_rows)
            self.log("\nTEMPO-FL PHASE SUMMARY")
            self.display(phase_summary_df)
            initial_owned_gain_mean = float(
                client_success_df["initial_owned_macro_recall_gain"].mean()
            )
            clients_owned_improved = int(
                np.sum(client_success_df["initial_owned_macro_recall_gain"] > 0)
            )
            initial_l3_gain = float(initial_l3_test["macro_f1"] - generic_l3_test["macro_f1"])
            final_l3_gain = float(final_l3_test["macro_f1"] - generic_l3_test["macro_f1"])
            final_vs_initial_l3 = float(final_l3_test["macro_f1"] - initial_l3_test["macro_f1"])
            final_l2_rows = l2_progress_df[
                l2_progress_df["stage"] == "Final_L2_After_100_HFL_Rounds"
            ]
            initial_l2_rows = l2_progress_df[
                l2_progress_df["stage"] == "Initial_L2_After_Specialist_Aggregation"
            ]
            total_shadow_correct = int(client_success_df["shadow_correct_replacements_audit"].sum())
            temporary_promotions = 0
            persistent_promotions = 0
            if len(promotion_df) > 0 and "promotion_accepted" in promotion_df.columns:
                accepted_promotions = promotion_df[promotion_df["promotion_accepted"] == True]
                temporary_promotions = int(np.sum(accepted_promotions["round"].isin([21, 22])))
                persistent_promotions = int(np.sum(accepted_promotions["round"] >= 41))
            success_checks_df = pd.DataFrame(
                [
                    {
                        "criterion": "Generic train/validation/internal-test contain every class",
                        "value": bool(
                            all(
                                (
                                    np.sum(self.generic_train_df["label"] == class_name) > 0
                                    and np.sum(self.generic_validation_df["label"] == class_name)
                                    > 0
                                    and (
                                        np.sum(self.generic_internal_test_df["label"] == class_name)
                                        > 0
                                    )
                                    for class_name in self.CLASS_NAMES
                                )
                            )
                        ),
                        "passed": bool(
                            all(
                                (
                                    np.sum(self.generic_train_df["label"] == class_name) > 0
                                    and np.sum(self.generic_validation_df["label"] == class_name)
                                    > 0
                                    and (
                                        np.sum(self.generic_internal_test_df["label"] == class_name)
                                        > 0
                                    )
                                    for class_name in self.CLASS_NAMES
                                )
                            )
                        ),
                    },
                    {
                        "criterion": "Mean owned-class Macro-Recall improves after local specialization",
                        "value": initial_owned_gain_mean,
                        "passed": bool(initial_owned_gain_mean > 0),
                    },
                    {
                        "criterion": "Most clients improve owned-class Macro-Recall",
                        "value": f"{clients_owned_improved}/{len(self.CLIENT_IDS)}",
                        "passed": bool(
                            clients_owned_improved >= math.ceil(len(self.CLIENT_IDS) * 0.75)
                        ),
                    },
                    {
                        "criterion": "Initial L3 specialist aggregation beats small generic L3",
                        "value": initial_l3_gain,
                        "passed": bool(initial_l3_gain > 0),
                    },
                    {
                        "criterion": "Mean initial L2 Macro-F1 gain vs generic is positive",
                        "value": float(initial_l2_rows["macro_f1_gain_vs_generic"].mean()),
                        "passed": bool(initial_l2_rows["macro_f1_gain_vs_generic"].mean() > 0),
                    },
                    {
                        "criterion": "Final L3 after 100 HFL rounds beats small generic",
                        "value": final_l3_gain,
                        "passed": bool(final_l3_gain > 0),
                    },
                    {
                        "criterion": "Final L3 is not worse than initial aggregated L3",
                        "value": final_vs_initial_l3,
                        "passed": bool(final_vs_initial_l3 >= 0),
                    },
                    {
                        "criterion": "Mean final L2 Macro-F1 gain vs generic is positive",
                        "value": float(final_l2_rows["macro_f1_gain_vs_generic"].mean()),
                        "passed": bool(final_l2_rows["macro_f1_gain_vs_generic"].mean() > 0),
                    },
                    {
                        "criterion": "Shadow hierarchy provides correct fallback decisions",
                        "value": total_shadow_correct,
                        "passed": bool(total_shadow_correct > 0),
                    },
                    {
                        "criterion": "Temporary two-round shift causes no permanent promotion",
                        "value": temporary_promotions,
                        "passed": bool(temporary_promotions == 0),
                    },
                    {
                        "criterion": "Persistent/new-normal phase produces validated promotion",
                        "value": persistent_promotions,
                        "passed": bool(persistent_promotions >= 1),
                    },
                ]
            )
            self.log("\nV12 ARTICLE SUCCESS CHECKS")
            self.display(success_checks_df)
            communication_df = pd.DataFrame(self.STATE["operational_round_history"])
            communication_summary_df = pd.DataFrame(
                [
                    {
                        "rounds": len(communication_df),
                        "total_hierarchical_gb": float(
                            communication_df["total_hierarchical_bytes"].sum() / 1024**3
                        ),
                        "mean_mb_per_round": float(
                            communication_df["total_hierarchical_mb"].mean()
                        ),
                    }
                ]
            )
            general_acc_df = pd.DataFrame(self.STATE.get("general_accuracy_rows", []))
            owned_acc_df = pd.DataFrame(self.STATE.get("owned_accuracy_rows", []))
            general_acc_df = general_acc_df.sort_values(
                ["stage_order", "level", "model_id"]
            ).reset_index(drop=True)
            owned_acc_df = owned_acc_df.sort_values(["stage_order", "client_id"]).reset_index(
                drop=True
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
                        "owned_class_test_samples": int(len(self.owned_class_test_set[cid])),
                        "planned_class_test_samples": int(
                            owned_acc_df[
                                (owned_acc_df["client_id"] == cid)
                                & (owned_acc_df["stage"] == "Generic")
                            ]["owned_test_samples"].iloc[0]
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
                        "used_for_training_or_gates": False,
                    },
                    {
                        "test_set": "owned_class_test_set",
                        "source": "Disjoint client partition of FinalTest_15pct_UNTOUCHED",
                        "scope": "One different partition per client; planned classes are primary",
                        "total_samples": int(
                            sum((len(self.owned_class_test_set[cid]) for cid in self.CLIENT_IDS))
                        ),
                        "used_for_training_or_gates": False,
                    },
                ]
            )
            general_acc_path = self.DIRS["master_metrics"] / "general_acc.xlsx"
            owned_acc_path = self.DIRS["master_metrics"] / "owned_acc.xlsx"
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
            }
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
            self.log("\nV12.2 GENERAL ACCURACY/MACRO-F1 — FINAL REPORT")
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
            self.log("\nV12.2 OWNED ACCURACY/MACRO-F1 — L1 CLIENT REPORT")
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
            self.log("\nV12.2 separate workbooks saved:")
            self.log("  general_acc:", general_acc_path)
            self.log("  owned_acc:", owned_acc_path)
            final_path = self.DIRS["master_metrics"] / "TEMPO_FL_V12_2_FINAL_ARTICLE_TABLES.xlsx"
            with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
                l3_progress_df.to_excel(writer, sheet_name="L3_Progress", index=False)
                l3_per_class_df.to_excel(writer, sheet_name="L3_PerClass_All6", index=False)
                l2_progress_df.to_excel(writer, sheet_name="L2_Progress", index=False)
                client_success_df.to_excel(writer, sheet_name="TEMPO_By_Client", index=False)
                final_planned_class_df.to_excel(
                    writer, sheet_name="Final_PlannedClasses", index=False
                )
                final_other_class_df.to_excel(writer, sheet_name="Final_OtherClasses", index=False)
                phase_summary_df.to_excel(writer, sheet_name="Phase_Summary", index=False)
                success_checks_df.to_excel(writer, sheet_name="Success_Checks", index=False)
                communication_summary_df.to_excel(writer, sheet_name="Communication", index=False)
                pd.DataFrame(self.STATE["client_round_history"]).to_excel(
                    writer, sheet_name="Client_Round_History", index=False
                )
                pd.DataFrame(self.STATE["context_round_history"]).to_excel(
                    writer, sheet_name="Context_Round_History", index=False
                )
                pd.DataFrame(self.STATE["campus_round_history"]).to_excel(
                    writer, sheet_name="Campus_Round_History", index=False
                )
                if len(promotion_df) > 0:
                    promotion_df.to_excel(writer, sheet_name="Promotions", index=False)
                if self.STATE["shift_eligibility_history"]:
                    pd.DataFrame(self.STATE["shift_eligibility_history"]).to_excel(
                        writer, sheet_name="Shift_Eligibility", index=False
                    )
            self.log("\nFinal V12 article workbook saved:", final_path)
            self.report_tables(
                {
                    "client_success_df": client_success_df,
                    "phase_summary_df": phase_summary_df,
                    "final_planned_class_df": final_planned_class_df,
                    "final_other_class_df": final_other_class_df,
                    "l3_progress_df": l3_progress_df,
                    "l2_progress_df": l2_progress_df,
                    "communication_summary_df": communication_summary_df,
                }
            )
        else:
            self.log(
                "Final V12 evaluation skipped because the experiment has not reached round 100."
            )
