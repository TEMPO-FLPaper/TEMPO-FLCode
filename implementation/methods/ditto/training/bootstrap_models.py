"""Reference-derived training/bootstrap_models_ditto. See README.md."""

import gc

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'ditto': 23}
    def bootstrap_models(self):
        self._init_state["global_weights"] = [w.copy() for w in self._init_generic_weights]
        self._init_state["personalized_weights"] = {
            cid: [w.copy() for w in self._init_generic_weights] for cid in self.CLIENT_IDS
        }
        global_initial_spec = {
            "level": "Global",
            "model_type": "Global",
            "model_id": "Global",
            "context": "AllClients",
            "weights": self._init_state["global_weights"],
            "temperature": 1.0,
        }
        personalized_initial_specs = [
            {
                "level": "Personalized",
                "model_type": "Personalized",
                "model_id": cid,
                "client_id": cid,
                "context": self.CLIENT_TO_CONTEXT[cid],
                "weights": self._init_state["personalized_weights"][cid],
                "temperature": 1.0,
            }
            for cid in self.CLIENT_IDS
        ]
        initial_general_rows = self.evaluate_general_stage_rows(
            "Initial",
            "Initial_Global_and_Personalized",
            1,
            [global_initial_spec] + personalized_initial_specs,
        )
        initial_owned_specs = [
            {
                "client_id": cid,
                "model_id": "Global",
                "model_type": "Global",
                "level": "Global",
                "weights": self._init_state["global_weights"],
                "temperature": 1.0,
            }
            for cid in self.CLIENT_IDS
        ] + [
            {
                "client_id": cid,
                "model_id": cid,
                "model_type": "Personalized",
                "level": "Personalized",
                "weights": self._init_state["personalized_weights"][cid],
                "temperature": 1.0,
            }
            for cid in self.CLIENT_IDS
        ]
        initial_owned_rows = self.evaluate_owned_stage_rows(
            "Initial", "Initial_Global_and_Personalized", 1, initial_owned_specs
        )
        self.upsert_stage_metric_rows(
            self._init_state, "general_accuracy_rows", initial_general_rows
        )
        self.upsert_stage_metric_rows(self._init_state, "owned_accuracy_rows", initial_owned_rows)
        self.log("\nDITTO GENERAL TEST — INITIAL GLOBAL + PERSONALIZED")
        self.display(pd.DataFrame(initial_general_rows))
        self.log("\nDITTO OWNED TEST — INITIAL GLOBAL + PERSONALIZED")
        self.display(pd.DataFrame(initial_owned_rows))
        self.save_weights_to_file(
            self._init_state["global_weights"],
            self.DIRS["initial"] / "initial_global_v12_2_ditto.weights.h5",
        )
        for cid in self.CLIENT_IDS:
            self.save_weights_to_file(
                self._init_state["personalized_weights"][cid],
                self.DIRS["initial"] / f"initial_personalized_{cid}_v12_2_ditto.weights.h5",
            )
        self._init_state["current_round"] = 0
        self.save_round_checkpoint(
            0,
            self._init_state,
            {
                "General_Initial": pd.DataFrame(initial_general_rows),
                "Owned_Initial": pd.DataFrame(initial_owned_rows),
            },
        )
        gc.collect()
        self.print_memory_snapshot("[After Ditto initialization]")
        self.log("\nDitto initialization is complete. Round 1 follows.")
        return self._init_state
