"""Reference-derived methods/rounds_flat. See README.md."""

import gc

import numpy as np

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'flat': 24}
    def run_round(self, round_id):
        self.set_round_seed(round_id)
        scenario = self.scenario_for_round(round_id)
        evaluation_round = self.is_evaluation_round(round_id)
        self.log("\n" + "#" * 112)
        self.log(
            f"FLAT FEDAVG V12.2 ROUND {round_id:03d}/{self.CONFIG['num_rounds']} | Scenario={scenario} | Evaluation={evaluation_round}"
        )
        self.log("#" * 112)
        client_results = {}
        client_rows = []
        global_parent_weights = self.STATE["global_weights"]
        for cid in self.CLIENT_IDS:
            result = self.train_fedavg_client(cid, round_id, global_parent_weights)
            client_results[cid] = result
            client_rows.append(
                {
                    "round": round_id,
                    "scenario": scenario,
                    "client_id": cid,
                    "data_group": self.CLIENT_TO_CONTEXT[cid],
                    "train_samples": len(self.client_train[cid]),
                    "local_steps": result["local_steps"],
                    "local_loss": result["loss"],
                    "local_accuracy": result["accuracy"],
                    "update_l2_norm": result["delta_l2_norm"],
                    "download_bytes": result["download_bytes"],
                    "upload_bytes": result["upload_bytes"],
                }
            )
        initial_round_tables = {}
        if round_id == 1:
            initial_client_general_rows = self.evaluate_general_stage_rows(
                "Initial",
                "Initial_Client_After_Local_Training",
                1,
                [
                    {
                        "level": "Client",
                        "model_id": cid,
                        "context": self.CLIENT_TO_CONTEXT[cid],
                        "weights": client_results[cid]["weights"],
                        "temperature": 1.0,
                    }
                    for cid in self.CLIENT_IDS
                ],
            )
            initial_client_owned_rows = self.evaluate_owned_stage_rows(
                "Initial",
                "Initial_Client_After_Local_Training",
                1,
                [
                    {
                        "client_id": cid,
                        "model_id": cid,
                        "evaluated_model_level": "Client",
                        "weights": client_results[cid]["weights"],
                        "temperature": 1.0,
                    }
                    for cid in self.CLIENT_IDS
                ],
            )
            self.upsert_stage_metric_rows(
                self.STATE, "general_accuracy_rows", initial_client_general_rows
            )
            self.upsert_stage_metric_rows(
                self.STATE, "owned_accuracy_rows", initial_client_owned_rows
            )
            initial_round_tables.update(
                {
                    "Initial_Client_General": pd.DataFrame(initial_client_general_rows),
                    "Initial_Client_Owned": pd.DataFrame(initial_client_owned_rows),
                }
            )
            self.log("\nFLAT FEDAVG GENERAL TEST — INITIAL CLIENT MODELS")
            self.display(pd.DataFrame(initial_client_general_rows))
            self.log("\nFLAT FEDAVG OWNED TEST — INITIAL CLIENT MODELS")
            self.display(pd.DataFrame(initial_client_owned_rows))
        total_samples = sum((len(self.client_train[cid]) for cid in self.CLIENT_IDS))
        new_global_weights = self.weighted_average_weights(
            [client_results[cid]["weights"] for cid in self.CLIENT_IDS],
            [len(self.client_train[cid]) for cid in self.CLIENT_IDS],
        )
        self.STATE["global_weights"] = [w.copy() for w in new_global_weights]
        if round_id == 1:
            initial_global_general_rows = self.evaluate_general_stage_rows(
                "Initial",
                "Initial_Global_After_FedAvg",
                2,
                [
                    {
                        "level": "Global",
                        "model_id": "Global",
                        "context": "Global",
                        "weights": self.STATE["global_weights"],
                        "temperature": 1.0,
                    }
                ],
            )
            self.upsert_stage_metric_rows(
                self.STATE, "general_accuracy_rows", initial_global_general_rows
            )
            initial_round_tables["Initial_Global_General"] = pd.DataFrame(
                initial_global_general_rows
            )
            self.log("\nFLAT FEDAVG GENERAL TEST — INITIAL GLOBAL MODEL")
            self.display(pd.DataFrame(initial_global_general_rows))
        local_loss = float(np.nanmean([row["local_loss"] for row in client_rows]))
        local_accuracy = float(np.nanmean([row["local_accuracy"] for row in client_rows]))
        round_row = {
            "round": round_id,
            "scenario": scenario,
            "evaluation_round": evaluation_round,
            "client_count": len(self.CLIENT_IDS),
            "aggregation_samples": int(total_samples),
            "mean_local_loss": local_loss,
            "mean_local_accuracy": local_accuracy,
            "global_model_bytes": self.weights_size_bytes(self.STATE["global_weights"]),
            "client_download_bytes": int(sum((row["download_bytes"] for row in client_rows))),
            "client_upload_bytes": int(sum((row["upload_bytes"] for row in client_rows))),
        }
        if evaluation_round:
            global_validation = self.evaluate_weights(
                self.STATE["global_weights"], self.global_validation_df, temperature=1.0
            )
            round_row.update(
                {
                    "validation_samples": len(self.global_validation_df),
                    "validation_accuracy": global_validation["accuracy"],
                    "validation_macro_f1": global_validation["macro_f1"],
                }
            )
        else:
            round_row.update(
                {
                    "validation_samples": 0,
                    "validation_accuracy": np.nan,
                    "validation_macro_f1": np.nan,
                }
            )
        self.STATE["round_history"].append(round_row)
        self.STATE["communication_history"].append(
            {
                "round": round_id,
                "scenario": scenario,
                "client_download_bytes": round_row["client_download_bytes"],
                "client_upload_bytes": round_row["client_upload_bytes"],
                "global_model_bytes": round_row["global_model_bytes"],
            }
        )
        self.STATE["current_round"] = round_id
        if evaluation_round:
            self.log(
                f"[Flat FedAvg] mean local loss={local_loss:.4f} | mean local accuracy={local_accuracy:.4f} | Global validation Macro-F1={round_row['validation_macro_f1']:.4f}"
            )
        else:
            self.log(
                f"[Flat FedAvg] mean local loss={local_loss:.4f} | mean local accuracy={local_accuracy:.4f}"
            )
        round_tables = {
            "Round_Summary": pd.DataFrame([round_row]),
            "Client_Updates": pd.DataFrame(client_rows),
        }
        round_tables.update(initial_round_tables)
        self.save_round_checkpoint(round_id, self.STATE, round_tables)
        del client_results
        del client_rows
        gc.collect()
