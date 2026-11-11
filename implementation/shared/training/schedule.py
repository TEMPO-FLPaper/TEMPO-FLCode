"""Reference-derived training/schedule. See README.md."""

import os, random

import numpy as np

import tensorflow as tf

class Operations:
    # Reference cells (zero-based): {'tempo': 5, 'ditto': 5, 'hfl': 5, 'flat': 5}
    def scenario_for_round(self, round_id):
        if 1 <= round_id <= 20:
            return "stable_commissioning"
        if 21 <= round_id <= 22:
            return "temporary_shift"
        if 23 <= round_id <= 40:
            return "recovery"
        if 41 <= round_id <= 65:
            return "persistent_shift"
        return "new_normal"

    # Reference cells (zero-based): {'tempo': 5, 'ditto': 5, 'hfl': 5, 'flat': 5}
    def is_evaluation_round(self, round_id):
        return (
            round_id in self.FORCED_EVAL_ROUNDS or round_id % self.CONFIG["evaluation_every"] == 0
        )

    # Reference cells (zero-based): {'tempo': 8, 'ditto': 7, 'hfl': 7, 'flat': 7}
    def set_round_seed(self, round_id=0):
        seed = self.CONFIG["seed"] + 10007 * int(round_id)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        return seed
