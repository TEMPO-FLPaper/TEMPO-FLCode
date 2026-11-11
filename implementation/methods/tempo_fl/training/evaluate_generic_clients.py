"""Reference-derived training/evaluate_generic_clients_tempo. See README.md."""

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'tempo': 29}
    def evaluate_generic_clients(self):
        pre_rows = []
        pre_all_class_rows = []
        self._init_pre_owned_metrics = {}
        pre_owned_gate_metrics = {}
        for cid in self.CLIENT_IDS:
            owned_metrics = self.evaluate_owned_specialist(
                self._init_generic_weights, cid, self.client_val_local[cid], temperature=1.0
            )
            owned_gate_metrics = self.evaluate_owned_specialist(
                self._init_generic_weights, cid, self.client_gate_local[cid], temperature=1.0
            )
            self._init_pre_owned_metrics[cid] = owned_metrics
            pre_owned_gate_metrics[cid] = owned_gate_metrics
            overall = self.overall_stage_row(
                self._init_generic_weights,
                self.client_val_local[cid],
                "Stage2_Generic_At_Client",
                cid,
                self.CLIENT_TO_CONTEXT[cid],
                temperature=1.0,
            )
            row = {
                **overall,
                "planned_classes": owned_metrics["planned_classes"],
                "owned_eval_support": owned_metrics["owned_eval_support"],
                "owned_accuracy": owned_metrics["owned_accuracy"],
                "owned_macro_recall": owned_metrics["owned_macro_recall"],
                "owned_min_class_recall": owned_metrics["owned_min_class_recall"],
                "other_eval_support": owned_metrics["other_eval_support"],
                "other_accuracy_secondary": owned_metrics["other_accuracy_secondary"],
                "other_macro_recall_secondary": owned_metrics["other_macro_recall_secondary"],
            }
            pre_rows.append(row)
            all_rows = self.classwise_accuracy_rows(
                self._init_generic_weights,
                self.client_val_local[cid],
                "Stage2_Generic_At_Client",
                cid,
                self.CLIENT_TO_CONTEXT[cid],
                temperature=1.0,
                train_reference_df=self.client_train[cid],
            )
            pre_all_class_rows.extend(self.add_planned_flag(all_rows, cid))
        self._init_pre_df = pd.DataFrame(pre_rows)
        self._init_pre_class_df = pd.DataFrame(pre_all_class_rows)
        self._init_state["client_pre_specialization_rows"] = list(pre_rows)
        self._init_state["client_class_stage_rows"].extend(pre_all_class_rows)
        self.log("\nPRIMARY LOW-LEVEL METRIC — GENERIC PERFORMANCE ON PLANNED/OWNED CLASSES")
        self.display(
            self._init_pre_df[
                [
                    "model_id",
                    "context",
                    "planned_classes",
                    "owned_eval_support",
                    "owned_accuracy",
                    "owned_macro_recall",
                    "owned_min_class_recall",
                ]
            ]
        )
        self.log("\nPER-PLANNED-CLASS GENERIC ACCURACY")
        self.display(
            self._init_pre_class_df[self._init_pre_class_df["planned_owned_class"]][
                [
                    "model_id",
                    "context",
                    "class_name",
                    "train_count_at_node",
                    "evaluation_support",
                    "class_accuracy_recall",
                ]
            ]
        )
        self.log("\nSECONDARY DIAGNOSTIC — OTHER/ALL LOCAL CLASSES")
        self.display(
            self._init_pre_df[
                [
                    "model_id",
                    "other_eval_support",
                    "other_accuracy_secondary",
                    "other_macro_recall_secondary",
                    "accuracy",
                    "macro_f1",
                ]
            ]
        )
        self.log("\n" + "=" * 118)
        self.log("STAGE 3 — MEMORY-SAFE DETAILED OWNED-CLASS SPECIALIZATION")
        self.log("=" * 118)
        self.log(
            "V12 uses ONE reusable Keras model for all eight clients. The model is reset to the generic weights before each client. Training proceeds epoch-by-epoch with adaptive learning-rate reduction and owned-class early stopping."
        )
