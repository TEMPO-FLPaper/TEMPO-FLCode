"""Reference-derived training/bootstrap_models_hfl. See README.md."""

import gc

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'hfl': 22}
    def bootstrap_models(self):
        self._init_state["l1_base_weights"] = {
            cid: [w.copy() for w in self._init_generic_weights] for cid in self.CLIENT_IDS
        }
        self._init_state["l2_base_weights"] = {
            context_name: self.weighted_average_weights(
                [
                    self._init_state["l1_base_weights"][cid]
                    for cid in self.CONTEXT_TO_CLIENTS[context_name]
                ],
                [len(self.client_train[cid]) for cid in self.CONTEXT_TO_CLIENTS[context_name]],
            )
            for context_name in self.CONFIG["context_names"]
        }
        self._init_state["l3_base_weights"] = self.weighted_average_weights(
            [
                self._init_state["l2_base_weights"][context_name]
                for context_name in self.CONFIG["context_names"]
            ],
            [
                sum((len(self.client_train[cid]) for cid in self.CONTEXT_TO_CLIENTS[context_name]))
                for context_name in self.CONFIG["context_names"]
            ],
        )
        initial_l1_specs = [
            {
                "level": "L1",
                "model_id": cid,
                "context": self.CLIENT_TO_CONTEXT[cid],
                "weights": self._init_state["l1_base_weights"][cid],
                "temperature": 1.0,
            }
            for cid in self.CLIENT_IDS
        ]
        initial_l1_general_rows = self.evaluate_general_stage_rows(
            "Initial", "Initial_L1_Generic_Initialization", 1, initial_l1_specs
        )
        initial_l1_owned_rows = self.evaluate_owned_stage_rows(
            "Initial",
            "Initial_L1_Generic_Initialization",
            1,
            [
                {
                    "client_id": cid,
                    "model_id": cid,
                    "evaluated_model_level": "L1",
                    "weights": self._init_state["l1_base_weights"][cid],
                    "temperature": 1.0,
                }
                for cid in self.CLIENT_IDS
            ],
        )
        initial_l2_general_rows = self.evaluate_general_stage_rows(
            "Initial",
            "Initial_L2_After_FedAvg",
            2,
            [
                {
                    "level": "L2",
                    "model_id": context_name,
                    "context": context_name,
                    "weights": self._init_state["l2_base_weights"][context_name],
                    "temperature": 1.0,
                }
                for context_name in self.CONFIG["context_names"]
            ],
        )
        initial_l3_general_rows = self.evaluate_general_stage_rows(
            "Initial",
            "Initial_L3_After_FedAvg",
            3,
            [
                {
                    "level": "L3",
                    "model_id": "Campus",
                    "context": "Campus",
                    "weights": self._init_state["l3_base_weights"],
                    "temperature": 1.0,
                }
            ],
        )
        self.upsert_stage_metric_rows(
            self._init_state,
            "general_accuracy_rows",
            initial_l1_general_rows + initial_l2_general_rows + initial_l3_general_rows,
        )
        self.upsert_stage_metric_rows(
            self._init_state, "owned_accuracy_rows", initial_l1_owned_rows
        )
        self.log("\nFedAvg GENERAL TEST — INITIAL L1")
        self.display(pd.DataFrame(initial_l1_general_rows))
        self.log("\nFedAvg GENERAL TEST — INITIAL L2")
        self.display(pd.DataFrame(initial_l2_general_rows))
        self.log("\nFedAvg GENERAL TEST — INITIAL L3")
        self.display(pd.DataFrame(initial_l3_general_rows))
        self.log("\nFedAvg OWNED TEST — INITIAL L1")
        self.display(pd.DataFrame(initial_l1_owned_rows))
        for cid in self.CLIENT_IDS:
            self.save_weights_to_file(
                self._init_state["l1_base_weights"][cid],
                self.DIRS["initial"] / f"initial_l1_{cid}_v12_2_fedavg.weights.h5",
            )
        for context_name in self.CONFIG["context_names"]:
            self.save_weights_to_file(
                self._init_state["l2_base_weights"][context_name],
                self.DIRS["initial"] / f"initial_l2_{context_name}_v12_2_fedavg.weights.h5",
            )
        self.save_weights_to_file(
            self._init_state["l3_base_weights"],
            self.DIRS["initial"] / "initial_l3_v12_2_fedavg.weights.h5",
        )
        self._init_state["current_round"] = 0
        self.save_round_checkpoint(
            0,
            self._init_state,
            {
                "General_Initial": pd.DataFrame(
                    initial_l1_general_rows + initial_l2_general_rows + initial_l3_general_rows
                ),
                "Owned_Initial": pd.DataFrame(initial_l1_owned_rows),
            },
        )
        gc.collect()
        self.print_memory_snapshot("[After FedAvg initialization]")
        self.log("\nFedAvg initialization is complete. Round 1 follows.")
        return self._init_state
