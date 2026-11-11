"""Reference-derived data/partitions. See README.md."""

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from implementation.shared.data.partitions import Operations as SharedOperations


class Operations(SharedOperations):
    # Reference cells (zero-based): {'tempo': 47}
    def context_dataframe(self, parts_dict, context_name):
        frames = [
            parts_dict[cid]
            for cid in self.CONTEXT_TO_CLIENTS[context_name]
            if len(parts_dict[cid]) > 0
        ]
        if not frames:
            return pd.DataFrame(columns=self.global_validation_df.columns)
        return pd.concat(frames, ignore_index=True)
