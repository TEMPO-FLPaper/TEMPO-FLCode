"""Reference-derived methods/state. See README.md."""

from collections import deque

import numpy as np

class Operations:
    # Reference cells (zero-based): {'hfl': 20}
    def fresh_runtime_state__hfl(self):
        return {
            "current_round": 0,
            "l1_base_weights": {},
            "l2_base_weights": {},
            "l3_base_weights": None,
            "general_accuracy_rows": [],
            "owned_accuracy_rows": [],
            "round_history": [],
            "communication_history": [],
        }
