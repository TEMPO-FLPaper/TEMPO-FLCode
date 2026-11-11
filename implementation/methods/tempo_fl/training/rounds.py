"""Reference-derived methods/rounds_tempo. See README.md."""

import gc, time

from collections import defaultdict

import numpy as np

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'tempo': 49}
    def run_round(self, round_id):
        round_start_time = time.time()
        self.set_round_seed(round_id)
        scenario = self.scenario_for_round(round_id)
        evaluation_round = self.is_evaluation_round(round_id)
        self.log("\n" + "#" * 126)
        self.log(
            f"TEMPO-FL V12.2 ROUND {round_id:03d}/{self.CONFIG['num_rounds']} | Scenario={scenario} | Evaluation={evaluation_round}"
        )
        self.log("#" * 126)
        operational_summaries = {}
        operational_eval_results = {}
        round_counter_rows = {}
        all_event_rows = []
        for cid in self.CLIENT_IDS:
            summary, event_rows = self.operational_cascade_for_client(cid, round_id, self.STATE)
            operational_summaries[cid] = summary
            all_event_rows.extend(event_rows)
            round_counter_rows[cid] = self.update_v12_2_operational_counters(
                cid, event_rows, self.STATE
            )
            if evaluation_round:
                operational_eval_results[cid] = self.evaluate_operational_cascade(
                    cid, round_id, self.STATE
                )
            else:
                operational_eval_results[cid] = self.empty_operational_eval_row()
            shift_status, shift_eligible, shift_classes = self.client_shift_label(cid, self.STATE)
            eval_row = operational_eval_results[cid]
            self.log(
                f"[Operation {cid} | {self.CLIENT_TO_CONTEXT[cid]:11s}] Status={shift_status:8s} | Reliance={summary['l1_shadow_reliance_rate']:.3f} | L2->L3={summary['l2_to_l3_count']:2d} | Resolution={summary['higher_resolution_rate']:.3f} | Unknown={summary['unknown_rate']:.3f} | L1F1={eval_row['l1_only_operational_macro_f1']:.3f} | CascadeF1={eval_row['operational_macro_f1']:.3f}"
                if evaluation_round
                else f"[Operation {cid} | {self.CLIENT_TO_CONTEXT[cid]:11s}] Status={shift_status:8s} | Reliance={summary['l1_shadow_reliance_rate']:.3f} | L2->L3={summary['l2_to_l3_count']:2d} | Resolution={summary['higher_resolution_rate']:.3f} | Unknown={summary['unknown_rate']:.3f}"
            )
        promotion_results = {}
        for cid in self.CLIENT_IDS:
            op = operational_summaries[cid]
            promotion_results[cid] = self.maybe_promote_l1(
                cid,
                round_id,
                op["l1_shadow_reliance_rate"],
                op["higher_resolution_rate"],
                op["events"],
                op["l1_to_l2_count"],
                op["higher_resolved_count"],
                self.STATE,
            )
        l2_updates_by_context = defaultdict(list)
        l2_counts_by_context = defaultdict(list)
        l2_local_results = {}
        client_upload_bytes = 0
        for cid in self.CLIENT_IDS:
            result = self.train_l2_shadow(cid, round_id, self.STATE)
            l2_local_results[cid] = result
            context_name = result["context"]
            l2_updates_by_context[context_name].append(result["weights"])
            l2_counts_by_context[context_name].append(result["aggregation_weight"])
            client_upload_bytes += int(result["upload_bytes"])
        old_l2_weights = {
            context_name: [w.copy() for w in self.STATE["l2_base_weights"][context_name]]
            for context_name in self.CONFIG["context_names"]
        }
        context_pre_topdown_weights = {}
        context_pre_topdown_scores = {}
        context_rows = []
        for context_name in self.CONFIG["context_names"]:
            raw_context_aggregate = self.weighted_average_weights(
                l2_updates_by_context[context_name], l2_counts_by_context[context_name]
            )
            context_gate_df = self.context_dataframe(self.client_gate_local, context_name)
            trust = self.select_trust_region_blend(
                old_l2_weights[context_name],
                raw_context_aggregate,
                context_gate_df,
                self.STATE["l2_temperature"][context_name],
                self.CONFIG["l2_blend_candidates"],
                self.CONFIG["l2_exploration_score_drop"],
                self.CONFIG["l2_hard_score_drop"],
                self.CONFIG["l2_min_improvement"],
            )
            selected = [w.copy() for w in trust["weights"]]
            selected_score = float(trust["score"])
            if (
                selected_score
                < self.STATE["l2_best_score"][context_name] - self.CONFIG["l2_rollback_score_drop"]
            ):
                selected = [w.copy() for w in self.STATE["l2_best_weights"][context_name]]
                selected_score = float(self.STATE["l2_best_score"][context_name])
                aggregate_mode = "rollback_to_best"
            else:
                aggregate_mode = trust["mode"]
            if (
                selected_score
                > self.STATE["l2_best_score"][context_name] + self.CONFIG["l2_min_improvement"]
            ):
                self.STATE["l2_best_score"][context_name] = selected_score
                self.STATE["l2_best_weights"][context_name] = [w.copy() for w in selected]
            context_pre_topdown_weights[context_name] = selected
            context_pre_topdown_scores[context_name] = max(selected_score, 0.0001)
            self.log(
                f"[L2 Upward {context_name:11s}] Mode={aggregate_mode:16s} | alpha={trust['alpha']:.2f} | GateΔ={selected_score - trust['old_score']:+.5f}"
            )
        context_sample_counts = [
            sum((len(self.client_train[cid]) for cid in self.CONTEXT_TO_CLIENTS[context_name]))
            for context_name in self.CONFIG["context_names"]
        ]
        l2_to_l3_bytes = (
            int(
                sum(
                    (
                        self.weights_size_bytes(context_pre_topdown_weights[context_name])
                        for context_name in self.CONFIG["context_names"]
                    )
                )
            )
            if self.CONFIG["count_l2_to_l3_communication"]
            else 0
        )
        raw_l3, l3_combined_weights = self.quality_weighted_average_weights(
            [
                context_pre_topdown_weights[context_name]
                for context_name in self.CONFIG["context_names"]
            ],
            context_sample_counts,
            [
                context_pre_topdown_scores[context_name]
                for context_name in self.CONFIG["context_names"]
            ],
            power=self.CONFIG["l3_context_quality_power"],
        )
        old_l3 = [w.copy() for w in self.STATE["l3_base_weights"]]
        l3_trust = self.select_trust_region_blend(
            old_l3,
            raw_l3,
            self.global_model_gate_df,
            self.STATE["l3_temperature"],
            self.CONFIG["l3_blend_candidates"],
            self.CONFIG["l3_exploration_score_drop"],
            self.CONFIG["l3_hard_score_drop"],
            self.CONFIG["l3_min_improvement"],
        )
        new_l3 = [w.copy() for w in l3_trust["weights"]]
        new_l3_score = float(l3_trust["score"])
        if new_l3_score < self.STATE["l3_best_score"] - self.CONFIG["l3_rollback_score_drop"]:
            new_l3 = [w.copy() for w in self.STATE["l3_best_weights"]]
            new_l3_score = float(self.STATE["l3_best_score"])
            l3_mode = "rollback_to_best"
        else:
            l3_mode = l3_trust["mode"]
        if new_l3_score > self.STATE["l3_best_score"] + self.CONFIG["l3_min_improvement"]:
            self.STATE["l3_best_score"] = new_l3_score
            self.STATE["l3_best_weights"] = [w.copy() for w in new_l3]
        self.STATE["l3_base_weights"] = [w.copy() for w in new_l3]
        self.STATE["l3_version"] += 1
        self.STATE["l3_total_accepted_updates"] += int(self.weight_delta_l2(new_l3, old_l3) > 1e-10)
        self.log(
            f"[L3 Campus Upward] Mode={l3_mode:16s} | alpha={l3_trust['alpha']:.2f} | GateΔ={new_l3_score - l3_trust['old_score']:+.5f} | BestGain={self.STATE['l3_best_score'] - self.STATE['initial_l3_best_score']:+.5f}"
        )
        final_l2_weights = {}
        topdown_rows = []
        l3_to_l2_bytes = 0
        for context_name in self.CONFIG["context_names"]:
            if self.CONFIG["count_l3_to_l2_communication"]:
                l3_to_l2_bytes += int(self.weights_size_bytes(self.STATE["l3_base_weights"]))
            topdown = self.safe_topdown_l3_to_l2(
                context_name,
                context_pre_topdown_weights[context_name],
                self.STATE["l3_base_weights"],
                self.STATE,
            )
            final_context = [w.copy() for w in topdown["weights"]]
            final_score = float(topdown["score"])
            final_l2_weights[context_name] = final_context
            if (
                final_score
                > self.STATE["l2_best_score"][context_name] + self.CONFIG["l2_min_improvement"]
            ):
                self.STATE["l2_best_score"][context_name] = final_score
                self.STATE["l2_best_weights"][context_name] = [w.copy() for w in final_context]
            self.STATE["l2_total_accepted_updates"][context_name] += int(
                self.weight_delta_l2(final_context, old_l2_weights[context_name]) > 1e-10
            )
            self.STATE["l2_version"][context_name] += 1
            topdown_rows.append(
                {
                    "round": round_id,
                    "context": context_name,
                    "topdown_alpha": topdown["alpha"],
                    "topdown_score_delta": topdown["score_delta"],
                    "final_context_gate_score": final_score,
                }
            )
            self.log(
                f"[L3 -> L2 {context_name:11s}] alpha={topdown['alpha']:.2f} | GateΔ={topdown['score_delta']:+.5f}"
            )
        self.STATE["l2_base_weights"] = final_l2_weights
        l2_to_client_bytes = 0
        l3_relay_to_client_bytes = 0
        for cid in self.CLIENT_IDS:
            context_name = self.CLIENT_TO_CONTEXT[cid]
            self.STATE["shadow_l2_weights"][cid] = [
                w.copy() for w in self.STATE["l2_base_weights"][context_name]
            ]
            self.STATE["shadow_l3_weights"][cid] = [w.copy() for w in self.STATE["l3_base_weights"]]
            self.STATE["shadow_l2_version"][cid] = self.STATE["l2_version"][context_name]
            self.STATE["shadow_l3_version"][cid] = self.STATE["l3_version"]
            l2_to_client_bytes += int(self.weights_size_bytes(self.STATE["shadow_l2_weights"][cid]))
            l3_relay_to_client_bytes += int(
                self.weights_size_bytes(self.STATE["shadow_l3_weights"][cid])
            )
        if evaluation_round or round_id in {
            self.CONFIG["commissioning_rounds"],
            self.CONFIG["persistent_profile_refresh_round"],
        }:
            self.refresh_all_gates(self.STATE, round_id)
        context_rows = []
        for context_name in self.CONFIG["context_names"]:
            if evaluation_round:
                context_eval_df = self.context_dataframe(self.client_val_local, context_name)
                context_metrics = self.evaluate_weights(
                    self.STATE["l2_base_weights"][context_name],
                    context_eval_df,
                    self.STATE["l2_temperature"][context_name],
                )
            else:
                context_metrics = {
                    "loss": np.nan,
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_precision": np.nan,
                    "macro_recall": np.nan,
                    "ece": np.nan,
                }
            td = next((row for row in topdown_rows if row["context"] == context_name))
            context_rows.append(
                {
                    "round": round_id,
                    "scenario": scenario,
                    "evaluation_round": evaluation_round,
                    "context": context_name,
                    "clients": ",".join(self.CONTEXT_TO_CLIENTS[context_name]),
                    "topdown_alpha": td["topdown_alpha"],
                    "topdown_score_delta": td["topdown_score_delta"],
                    "best_gate_score": self.STATE["l2_best_score"][context_name],
                    "best_gate_score_gain_from_round0": self.STATE["l2_best_score"][context_name]
                    - self.STATE["initial_l2_best_score"][context_name],
                    **{f"context_{key}": value for key, value in context_metrics.items()},
                }
            )
        if evaluation_round:
            campus_metrics = self.evaluate_weights(
                self.STATE["l3_base_weights"],
                self.global_validation_df,
                self.STATE["l3_temperature"],
            )
        else:
            campus_metrics = {
                "loss": np.nan,
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "balanced_accuracy": np.nan,
                "macro_precision": np.nan,
                "macro_recall": np.nan,
                "ece": np.nan,
            }
        campus_row = {
            "round": round_id,
            "scenario": scenario,
            "evaluation_round": evaluation_round,
            "campus_update_mode": l3_mode,
            "campus_selected_alpha": l3_trust["alpha"],
            "campus_best_gate_score": self.STATE["l3_best_score"],
            "campus_best_gate_score_gain_from_round0": self.STATE["l3_best_score"]
            - self.STATE["initial_l3_best_score"],
            "campus_update_norm": self.weight_delta_l2(self.STATE["l3_base_weights"], old_l3),
            **{f"campus_{key}": value for key, value in campus_metrics.items()},
        }
        shift_eligibility_df = pd.DataFrame()
        if round_id in {
            self.CONFIG["commissioning_rounds"],
            self.CONFIG["persistent_profile_refresh_round"],
        }:
            shift_eligibility_df = self.build_shift_profiles(self.STATE, round_id)
            self.log(
                f"\n[V12 Shift Discovery] Round {round_id}: {int(shift_eligibility_df['eligible'].sum())}/{len(self.CLIENT_IDS)} clients eligible."
            )
        client_rows = []
        for cid in self.CLIENT_IDS:
            context_name = self.CLIENT_TO_CONTEXT[cid]
            op = operational_summaries[cid]
            eval_row = operational_eval_results[cid]
            promo = promotion_results[cid]
            audit = round_counter_rows[cid]
            status, eligible, classes = self.client_shift_label(cid, self.STATE)
            client_row = {
                "round": round_id,
                "scenario": scenario,
                "evaluation_round": evaluation_round,
                "client_id": cid,
                "context": context_name,
                "shift_status": status,
                "shift_eligible": eligible,
                "shift_classes": classes,
                **op,
                **eval_row,
                **audit,
                "shadow_train_loss": l2_local_results[cid]["loss"],
                "shadow_train_accuracy": l2_local_results[cid]["accuracy"],
                "shadow_train_delta_norm": l2_local_results[cid]["delta_l2_norm"],
                "promotion_triggered": promo.get("promotion_triggered", False),
                "promotion_accepted": promo.get("promotion_accepted", False),
                "promotion_count_total": self.STATE["promotion_count"][cid],
                "baseline_reliance": self.STATE["baseline_reliance"][cid],
            }
            client_rows.append(client_row)
            self.STATE["client_round_history"].append(dict(client_row))
        self.STATE["context_round_history"].extend([dict(row) for row in context_rows])
        self.STATE["campus_round_history"].append(dict(campus_row))
        round_time = float(time.time() - round_start_time)
        total_comm_bytes = int(
            client_upload_bytes
            + l2_to_l3_bytes
            + l3_to_l2_bytes
            + l2_to_client_bytes
            + l3_relay_to_client_bytes
        )
        communication_row = {
            "round": round_id,
            "scenario": scenario,
            "client_to_l2_upload_bytes": int(client_upload_bytes),
            "l2_to_l3_bytes": int(l2_to_l3_bytes),
            "l3_to_l2_bytes": int(l3_to_l2_bytes),
            "l2_to_client_bytes": int(l2_to_client_bytes),
            "l3_relay_to_client_bytes": int(l3_relay_to_client_bytes),
            "total_hierarchical_bytes": total_comm_bytes,
            "total_hierarchical_mb": float(total_comm_bytes / 1024**2),
            "round_time_sec": round_time,
        }
        self.STATE["operational_round_history"].append(dict(communication_row))
        self.log(
            f"[Round {round_id:03d}] Communication={communication_row['total_hierarchical_mb']:.2f} MB | Time={round_time:.1f}s"
        )
        round_tables = {
            "Clients": pd.DataFrame(client_rows),
            "Contexts": pd.DataFrame(context_rows),
            "Campus": pd.DataFrame([campus_row]),
            "Events": pd.DataFrame(all_event_rows),
            "Communication": pd.DataFrame([communication_row]),
            "TopDown": pd.DataFrame(topdown_rows),
        }
        if len(shift_eligibility_df) > 0:
            round_tables["ShiftEligibility"] = shift_eligibility_df
        if promotion_results:
            round_tables["Promotion"] = pd.DataFrame(
                [{"client_id": cid, **result} for cid, result in promotion_results.items()]
            )
        self.STATE["current_round"] = round_id
        pd.DataFrame(self.STATE["client_round_history"]).to_excel(
            self.DIRS["master_metrics"] / "client_round_history_v12_2.xlsx", index=False
        )
        pd.DataFrame(self.STATE["context_round_history"]).to_excel(
            self.DIRS["master_metrics"] / "context_round_history_v12_2.xlsx", index=False
        )
        pd.DataFrame(self.STATE["campus_round_history"]).to_excel(
            self.DIRS["master_metrics"] / "campus_round_history_v12_2.xlsx", index=False
        )
        pd.DataFrame(self.STATE["operational_round_history"]).to_excel(
            self.DIRS["master_metrics"] / "communication_round_history_v12_2.xlsx", index=False
        )
        if self.STATE["promotion_history"]:
            pd.DataFrame(self.STATE["promotion_history"]).to_excel(
                self.DIRS["master_metrics"] / "promotion_history_v12_2.xlsx", index=False
            )
        if self.STATE["shift_eligibility_history"]:
            pd.DataFrame(self.STATE["shift_eligibility_history"]).to_excel(
                self.DIRS["master_metrics"] / "shift_eligibility_history_v12_2.xlsx", index=False
            )
        self.save_round_checkpoint(round_id, self.STATE, round_tables)
        gc.collect()
