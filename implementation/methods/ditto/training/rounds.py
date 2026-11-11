"""Reference-derived methods/rounds_ditto. See README.md."""

import gc

import numpy as np

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'ditto': 25}
    def run_round(self, round_id):
        self.set_round_seed(round_id)
        scenario = self.scenario_for_round(round_id)
        evaluation_round = self.is_evaluation_round(round_id)
        self.log("\n" + "#" * 112)
        self.log(
            f"DITTO V12.2 ROUND {round_id:03d}/{self.CONFIG['num_rounds']} | Scenario={scenario} | Evaluation={evaluation_round}"
        )
        self.log("#" * 112)
        global_parent_weights = [w.copy() for w in self.STATE["global_weights"]]
        personal_parent_weights = {
            cid: [w.copy() for w in self.STATE["personalized_weights"][cid]]
            for cid in self.CLIENT_IDS
        }
        client_results = {}
        client_rows = []
        global_local_weights = []
        global_aggregation_weights = []
        next_personalized_weights = {}
        for cid in self.CLIENT_IDS:
            result = self.train_ditto_client(
                cid, round_id, global_parent_weights, personal_parent_weights[cid]
            )
            client_results[cid] = result
            global_local_weights.append(result["global_weights"])
            global_aggregation_weights.append(result["aggregation_weight"])
            next_personalized_weights[cid] = [w.copy() for w in result["personalized_weights"]]
            client_rows.append(
                {
                    "round": round_id,
                    "scenario": scenario,
                    "client_id": cid,
                    "context": result["context"],
                    "train_samples": len(self.client_train[cid]),
                    "local_steps": result["local_steps"],
                    "global_local_loss": result["global_loss"],
                    "global_local_accuracy": result["global_accuracy"],
                    "personal_local_loss": result["personal_loss"],
                    "personal_local_accuracy": result["personal_accuracy"],
                    "ditto_lambda": self.CONFIG["ditto_lambda"],
                    "personal_mean_proximal_norm": result["mean_proximal_norm"],
                    "global_update_l2_norm": result["global_delta_l2_norm"],
                    "personal_update_l2_norm": result["personal_delta_l2_norm"],
                    "download_bytes": result["download_bytes"],
                    "global_upload_bytes": result["global_upload_bytes"],
                    "personal_state_bytes": result["personal_state_bytes"],
                    "personalized_model_uploaded": False,
                }
            )
        new_global_weights = self.weighted_average_weights(
            global_local_weights, global_aggregation_weights
        )
        self.STATE["global_weights"] = [w.copy() for w in new_global_weights]
        self.STATE["personalized_weights"] = next_personalized_weights
        mean_global_loss = float(np.nanmean([row["global_local_loss"] for row in client_rows]))
        mean_global_accuracy = float(
            np.nanmean([row["global_local_accuracy"] for row in client_rows])
        )
        mean_personal_loss = float(np.nanmean([row["personal_local_loss"] for row in client_rows]))
        mean_personal_accuracy = float(
            np.nanmean([row["personal_local_accuracy"] for row in client_rows])
        )
        round_row = {
            "round": round_id,
            "scenario": scenario,
            "evaluation_round": evaluation_round,
            "client_count": len(self.CLIENT_IDS),
            "ditto_lambda": self.CONFIG["ditto_lambda"],
            "mean_global_local_loss": mean_global_loss,
            "mean_global_local_accuracy": mean_global_accuracy,
            "mean_personal_local_loss": mean_personal_loss,
            "mean_personal_local_accuracy": mean_personal_accuracy,
            "global_model_bytes": self.weights_size_bytes(self.STATE["global_weights"]),
            "client_download_bytes": int(sum((row["download_bytes"] for row in client_rows))),
            "global_upload_bytes": int(sum((row["global_upload_bytes"] for row in client_rows))),
            "personalized_upload_bytes": 0,
            "personalized_state_bytes_total": int(
                sum((row["personal_state_bytes"] for row in client_rows))
            ),
        }
        if evaluation_round:
            global_validation = self.evaluate_weights(
                self.STATE["global_weights"], self.global_validation_df, temperature=1.0
            )
            personal_validation = [
                self.evaluate_weights(
                    self.STATE["personalized_weights"][cid],
                    self.global_validation_df,
                    temperature=1.0,
                )
                for cid in self.CLIENT_IDS
            ]
            round_row.update(
                {
                    "validation_samples": len(self.global_validation_df),
                    "global_validation_accuracy": global_validation["accuracy"],
                    "global_validation_macro_f1": global_validation["macro_f1"],
                    "personalized_validation_mean_accuracy": float(
                        np.nanmean([item["accuracy"] for item in personal_validation])
                    ),
                    "personalized_validation_mean_macro_f1": float(
                        np.nanmean([item["macro_f1"] for item in personal_validation])
                    ),
                    "personalized_validation_min_accuracy": float(
                        np.nanmin([item["accuracy"] for item in personal_validation])
                    ),
                    "personalized_validation_min_macro_f1": float(
                        np.nanmin([item["macro_f1"] for item in personal_validation])
                    ),
                }
            )
        else:
            round_row.update(
                {
                    "validation_samples": 0,
                    "global_validation_accuracy": np.nan,
                    "global_validation_macro_f1": np.nan,
                    "personalized_validation_mean_accuracy": np.nan,
                    "personalized_validation_mean_macro_f1": np.nan,
                    "personalized_validation_min_accuracy": np.nan,
                    "personalized_validation_min_macro_f1": np.nan,
                }
            )
        self.STATE["round_history"].append(round_row)
        self.STATE["communication_history"].append(
            {
                "round": round_id,
                "scenario": scenario,
                "client_download_bytes": round_row["client_download_bytes"],
                "global_upload_bytes": round_row["global_upload_bytes"],
                "personalized_upload_bytes": 0,
                "personalized_state_bytes_total": round_row["personalized_state_bytes_total"],
            }
        )
        self.STATE["current_round"] = round_id
        if evaluation_round:
            self.log(
                f"[Ditto] global validation Acc={round_row['global_validation_accuracy']:.4f} | Macro-F1={round_row['global_validation_macro_f1']:.4f} | personalized mean Acc={round_row['personalized_validation_mean_accuracy']:.4f} | Macro-F1={round_row['personalized_validation_mean_macro_f1']:.4f}"
            )
        self.log(
            f"[Ditto] global local loss={mean_global_loss:.4f} | global local accuracy={mean_global_accuracy:.4f} | personal local loss={mean_personal_loss:.4f} | personal local accuracy={mean_personal_accuracy:.4f}"
        )
        self.save_round_checkpoint(
            round_id,
            self.STATE,
            {
                "Round_Summary": pd.DataFrame([round_row]),
                "Client_Updates": pd.DataFrame(client_rows),
            },
        )
        del client_results
        del client_rows
        del global_local_weights
        del global_aggregation_weights
        del next_personalized_weights
        gc.collect()
