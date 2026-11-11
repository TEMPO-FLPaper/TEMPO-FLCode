"""Reference-derived training/bootstrap_models_flat. See README.md."""

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'flat': 22}
    def bootstrap_models(self):
        self._init_state["global_weights"] = [w.copy() for w in self._init_generic_weights]
        self._init_state["current_round"] = 0
        self.save_round_checkpoint(
            0,
            self._init_state,
            {
                "Generic_General": pd.DataFrame(self._init_generic_general_rows),
                "Generic_Owned": pd.DataFrame(self._init_generic_owned_rows),
            },
        )
        self.print_memory_snapshot("[After flat FedAvg Generic initialization]")
        self.log("\nFlat FedAvg initialization is complete. Round 1 follows.")
        return self._init_state
