"""Reference-derived methods/state. See README.md."""

from collections import deque

import numpy as np

class Operations:
    # Reference cells (zero-based): {'tempo': 24}
    def fresh_runtime_state__tempo(self):
        return {
            "current_round": 0,
            "generic_seed_weights": None,
            "generic_stage_summary": {},
            "generic_global_class_rows": [],
            "general_accuracy_rows": [],
            "owned_accuracy_rows": [],
            "client_pre_specialization_rows": [],
            "client_post_specialization_rows": [],
            "client_class_stage_rows": [],
            "hierarchy_initial_rows": [],
            "client_operational_counters": {
                cid: {
                    "total_events": 0,
                    "specialist_accepted_events": 0,
                    "specialist_correct_events_audit": 0,
                    "shadow_invocations": 0,
                    "shadow_replacements": 0,
                    "shadow_correct_replacements_audit": 0,
                    "l2_replacements": 0,
                    "l2_correct_replacements_audit": 0,
                    "l3_replacements": 0,
                    "l3_correct_replacements_audit": 0,
                    "unknown_events": 0,
                    "rounds_specialist_only": 0,
                    "rounds_specialist_dominant": 0,
                    "rounds_shadow_invoked": 0,
                    "rounds_shadow_replaced": 0,
                    "rounds_shadow_effective_audit": 0,
                }
                for cid in self.CLIENT_IDS
            },
            "l1_base_weights": {},
            "l2_base_weights": {},
            "l3_base_weights": None,
            "shadow_l2_weights": {},
            "shadow_l3_weights": {},
            "l2_best_weights": {},
            "l3_best_weights": None,
            "l2_best_score": {
                context_name: -np.inf for context_name in self.CONFIG["context_names"]
            },
            "l3_best_score": -np.inf,
            "initial_l2_best_score": {},
            "initial_l3_best_score": -np.inf,
            "reliability_models": {"L1": {}, "L2": {}, "L3": None},
            "reliability_history": [],
            "l2_exploration_streak": {
                context_name: 0 for context_name in self.CONFIG["context_names"]
            },
            "l3_exploration_streak": 0,
            "l2_total_accepted_updates": {
                context_name: 0 for context_name in self.CONFIG["context_names"]
            },
            "l3_total_accepted_updates": 0,
            "l3_total_proxy_accepted_updates": 0,
            "l2_version": {context_name: 0 for context_name in self.CONFIG["context_names"]},
            "l3_version": 0,
            "shadow_l2_version": {cid: 0 for cid in self.CLIENT_IDS},
            "shadow_l3_version": {cid: 0 for cid in self.CLIENT_IDS},
            "l1_temperature": {},
            "l2_temperature": {},
            "l3_temperature": 1.0,
            "l1_threshold": {},
            "l2_threshold": {},
            "l3_threshold": 0.5,
            "threshold_diagnostics": [],
            "shift_profiles": {cid: None for cid in self.CLIENT_IDS},
            "shift_profile_rows": [],
            "shift_profile_history": [],
            "shift_pool_history": [],
            "shift_eligibility_history": [],
            "teacher_buffers": {
                cid: deque(maxlen=self.CONFIG["max_teacher_buffer"]) for cid in self.CLIENT_IDS
            },
            "baseline_reliance_samples": {cid: [] for cid in self.CLIENT_IDS},
            "baseline_reliance": {cid: np.nan for cid in self.CLIENT_IDS},
            "rebaseline_active": {cid: False for cid in self.CLIENT_IDS},
            "rebaseline_samples": {cid: [] for cid in self.CLIENT_IDS},
            "rebaseline_start_round": {cid: None for cid in self.CLIENT_IDS},
            "reliance_history": {
                cid: deque(maxlen=self.CONFIG["persistence_rounds"]) for cid in self.CLIENT_IDS
            },
            "resolution_history": {
                cid: deque(maxlen=self.CONFIG["persistence_rounds"]) for cid in self.CLIENT_IDS
            },
            "escalation_count_history": {
                cid: deque(maxlen=self.CONFIG["persistence_rounds"]) for cid in self.CLIENT_IDS
            },
            "resolved_count_history": {
                cid: deque(maxlen=self.CONFIG["persistence_rounds"]) for cid in self.CLIENT_IDS
            },
            "total_event_count_history": {
                cid: deque(maxlen=self.CONFIG["persistence_rounds"]) for cid in self.CLIENT_IDS
            },
            "promotion_count": {cid: 0 for cid in self.CLIENT_IDS},
            "last_promotion_round": {cid: -(10**9) for cid in self.CLIENT_IDS},
            "last_promotion_attempt_round": {cid: -(10**9) for cid in self.CLIENT_IDS},
            "l2_rejected_server_updates": {
                context_name: 0 for context_name in self.CONFIG["context_names"]
            },
            "l2_rollback_count": {context_name: 0 for context_name in self.CONFIG["context_names"]},
            "l3_rejected_server_updates": 0,
            "l3_rollback_count": 0,
            "client_round_history": [],
            "context_round_history": [],
            "campus_round_history": [],
            "operational_round_history": [],
            "promotion_history": [],
        }

    # Reference cells (zero-based): {'tempo': 49}
    def empty_operational_eval_row(self):
        return {
            "operational_eval_samples": 0,
            "l1_only_operational_accuracy": np.nan,
            "l1_only_operational_macro_f1": np.nan,
            "l1_only_operational_balanced_accuracy": np.nan,
            "operational_accuracy": np.nan,
            "operational_macro_f1": np.nan,
            "operational_balanced_accuracy": np.nan,
            "operational_coverage": np.nan,
            "operational_selective_accuracy": np.nan,
            "operational_selective_macro_f1": np.nan,
            "forced_cascade_accuracy": np.nan,
            "forced_cascade_macro_f1": np.nan,
            "forced_cascade_balanced_accuracy": np.nan,
            "cascade_accuracy_gain_over_l1": np.nan,
            "cascade_macro_f1_gain_over_l1": np.nan,
            "forced_cascade_accuracy_gain_over_l1": np.nan,
            "forced_cascade_macro_f1_gain_over_l1": np.nan,
            "operational_eval_reliance_rate": np.nan,
            "operational_eval_higher_resolution_rate": np.nan,
            "operational_eval_true_higher_resolution_rate": np.nan,
            "operational_eval_higher_accept_correctness": np.nan,
            "operational_eval_higher_override_count": np.nan,
            "operational_eval_beneficial_override_count": np.nan,
            "operational_eval_harmful_override_count": np.nan,
            "operational_eval_harmful_override_rate": np.nan,
            "operational_eval_override_net_correctness_gain": np.nan,
            "operational_eval_unknown_rate": np.nan,
            "operational_eval_mean_depth": np.nan,
            "operational_eval_mean_resolved_expected_gain": np.nan,
        }

    # Reference cells (zero-based): {'tempo': 49}
    def client_shift_label(self, cid, state):
        profile = state["shift_profiles"].get(cid)
        if not profile:
            return ("PENDING", np.nan, "")
        eligible = bool(profile.get("eligible", False))
        classes = ",".join(profile.get("classes", []))
        return ("ELIGIBLE" if eligible else "CONTROL", eligible, classes)
