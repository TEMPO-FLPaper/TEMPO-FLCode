"""Reference-derived methods/rounds_hfl. See README.md."""

import gc

import numpy as np

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'hfl': 24}
    def run_round(self, round_id):
        self.set_round_seed(round_id)
        scenario = self.scenario_for_round(round_id)
        evaluation_round = self.is_evaluation_round(round_id)
        self.log("\n" + "#" * 112)
        self.log(
            f"FEDAVG V12.2 ROUND {round_id:03d}/{self.CONFIG['num_rounds']} | Scenario={scenario} | Evaluation={evaluation_round}"
        )
        self.log("#" * 112)
        client_results = {}
        client_rows = []
        context_updates = {context_name: [] for context_name in self.CONFIG["context_names"]}
        context_counts = {context_name: [] for context_name in self.CONFIG["context_names"]}
        for cid in self.CLIENT_IDS:
            context_name = self.CLIENT_TO_CONTEXT[cid]
            parent_weights = self.STATE["l3_base_weights"]
            result = self.train_fedavg_client(cid, round_id, parent_weights)
            client_results[cid] = result
            context_updates[context_name].append(result["weights"])
            context_counts[context_name].append(result["aggregation_weight"])
            client_rows.append(
                {
                    "round": round_id,
                    "scenario": scenario,
                    "client_id": cid,
                    "context": context_name,
                    "train_samples": len(self.client_train[cid]),
                    "local_steps": result["local_steps"],
                    "local_loss": result["loss"],
                    "local_accuracy": result["accuracy"],
                    "update_l2_norm": result["delta_l2_norm"],
                    "download_bytes": result["download_bytes"],
                    "upload_bytes": result["upload_bytes"],
                }
            )
        new_l2_weights = {}
        context_rows = []
        for context_name in self.CONFIG["context_names"]:
            new_l2_weights[context_name] = self.weighted_average_weights(
                context_updates[context_name], context_counts[context_name]
            )
            total_context_samples = int(sum(context_counts[context_name]))
            context_rows.append(
                {
                    "round": round_id,
                    "scenario": scenario,
                    "context": context_name,
                    "client_count": len(context_updates[context_name]),
                    "aggregation_samples": total_context_samples,
                    "l2_to_l3_upload_bytes": self.weights_size_bytes(new_l2_weights[context_name]),
                }
            )
        new_l3_weights = self.weighted_average_weights(
            [new_l2_weights[context_name] for context_name in self.CONFIG["context_names"]],
            [
                sum((len(self.client_train[cid]) for cid in self.CONTEXT_TO_CLIENTS[context_name]))
                for context_name in self.CONFIG["context_names"]
            ],
        )
        self.STATE["l1_base_weights"] = {
            cid: [w.copy() for w in client_results[cid]["weights"]] for cid in self.CLIENT_IDS
        }
        self.STATE["l2_base_weights"] = new_l2_weights
        self.STATE["l3_base_weights"] = [w.copy() for w in new_l3_weights]
        local_loss = float(np.nanmean([row["local_loss"] for row in client_rows]))
        local_accuracy = float(np.nanmean([row["local_accuracy"] for row in client_rows]))
        round_row = {
            "round": round_id,
            "scenario": scenario,
            "evaluation_round": evaluation_round,
            "client_count": len(self.CLIENT_IDS),
            "mean_local_loss": local_loss,
            "mean_local_accuracy": local_accuracy,
            "l3_model_bytes": self.weights_size_bytes(self.STATE["l3_base_weights"]),
            "client_download_bytes": int(sum((row["download_bytes"] for row in client_rows))),
            "client_upload_bytes": int(sum((row["upload_bytes"] for row in client_rows))),
            "l2_to_l3_upload_bytes": int(
                sum((row["l2_to_l3_upload_bytes"] for row in context_rows))
            ),
        }
        if evaluation_round:
            l3_validation = self.evaluate_weights(
                self.STATE["l3_base_weights"], self.global_validation_df, temperature=1.0
            )
            round_row.update(
                {
                    "validation_samples": len(self.global_validation_df),
                    "validation_accuracy": l3_validation["accuracy"],
                    "validation_macro_f1": l3_validation["macro_f1"],
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
                "l2_to_l3_upload_bytes": round_row["l2_to_l3_upload_bytes"],
            }
        )
        self.STATE["current_round"] = round_id
        self.log(
            f"[FedAvg] mean local loss={local_loss:.4f} | mean local accuracy={local_accuracy:.4f} | L3 validation Macro-F1={round_row['validation_macro_f1']:.4f}"
            if evaluation_round
            else f"[FedAvg] mean local loss={local_loss:.4f} | mean local accuracy={local_accuracy:.4f}"
        )
        self.save_round_checkpoint(
            round_id,
            self.STATE,
            {
                "Round_Summary": pd.DataFrame([round_row]),
                "Client_Updates": pd.DataFrame(client_rows),
                "Context_Aggregates": pd.DataFrame(context_rows),
            },
        )
        del client_results
        del client_rows
        del context_updates
        del context_counts
        del context_rows
        gc.collect()
