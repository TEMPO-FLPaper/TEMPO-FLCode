"""Reference-derived methods/state. See README.md."""

from collections import deque

import numpy as np

class Operations:
    # Reference cells (zero-based): {'ditto': 21}
    def fresh_runtime_state__ditto(self):
        return {
            "current_round": 0,
            "global_weights": None,
            "personalized_weights": {},
            "general_accuracy_rows": [],
            "owned_accuracy_rows": [],
            "round_history": [],
            "communication_history": [],
        }
