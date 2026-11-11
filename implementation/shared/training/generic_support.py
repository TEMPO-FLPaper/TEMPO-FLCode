"""Reference-derived training/generic_support. See README.md."""

import numpy as np

class Operations:
    # Reference cells (zero-based): {'tempo': 29, 'ditto': 23, 'hfl': 22, 'flat': 22}
    def generic_class_weights(self):
        if not self.CONFIG["generic_use_class_weights"]:
            return None
        counts = self.generic_train_df["label_id"].value_counts().to_dict()
        total = float(len(self.generic_train_df))
        result = {}
        for class_id in range(self.NUM_CLASSES):
            count = max(int(counts.get(class_id, 0)), 1)
            raw = total / (self.NUM_CLASSES * count)
            result[class_id] = float(
                np.clip(
                    raw,
                    self.CONFIG["generic_class_weight_min"],
                    self.CONFIG["generic_class_weight_max"],
                )
            )
        return result

    # Reference cells (zero-based): {'tempo': 29, 'ditto': 23, 'hfl': 22, 'flat': 22}
    def add_planned_flag(self, class_rows, cid):
        output = []
        for row in class_rows:
            row = dict(row)
            row["planned_owned_class"] = row["class_name"] in self.CLIENT_PLANNED_CLASSES[cid]
            output.append(row)
        return output
