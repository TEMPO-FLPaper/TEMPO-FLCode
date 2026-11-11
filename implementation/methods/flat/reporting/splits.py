"""Reference-derived reporting/splits_flat. See README.md."""

import numpy as np

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'flat': 11}
    def report_splits(self):
        split_summary = pd.DataFrame(
            [
                {"split": "FullDataset", "count": len(self.full_df), "percent_of_full": 100.0},
                {
                    "split": "InitialTrain_60pct",
                    "count": len(self.train_initial_df),
                    "percent_of_full": 100.0 * len(self.train_initial_df) / len(self.full_df),
                },
                {
                    "split": "GenericTrain_BALANCED",
                    "count": len(self.generic_train_df),
                    "percent_of_full": 100.0 * len(self.generic_train_df) / len(self.full_df),
                },
                {
                    "split": "GenericValidation_BALANCED",
                    "count": len(self.generic_validation_df),
                    "percent_of_full": 100.0 * len(self.generic_validation_df) / len(self.full_df),
                },
                {
                    "split": "GenericInternalTest_BALANCED",
                    "count": len(self.generic_internal_test_df),
                    "percent_of_full": 100.0
                    * len(self.generic_internal_test_df)
                    / len(self.full_df),
                },
                {
                    "split": "PrivateFederatedTrain",
                    "count": len(self.private_train_df),
                    "percent_of_full": 100.0 * len(self.private_train_df) / len(self.full_df),
                },
                {
                    "split": "GlobalValidation_10pct",
                    "count": len(self.global_validation_df),
                    "percent_of_full": 100.0 * len(self.global_validation_df) / len(self.full_df),
                },
                {
                    "split": "OnlinePool_15pct",
                    "count": len(self.online_df),
                    "percent_of_full": 100.0 * len(self.online_df) / len(self.full_df),
                },
                {
                    "split": "FinalTest_15pct_UNTOUCHED",
                    "count": len(self.global_test_df),
                    "percent_of_full": 100.0 * len(self.global_test_df) / len(self.full_df),
                },
            ]
        )
        self.log("\nV12 GLOBAL DATA SPLITS")
        self.display(split_summary)
        generic_support_rows = []
        for class_id, class_name in enumerate(self.CLASS_NAMES):
            generic_support_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "generic_train": int(np.sum(self.generic_train_df["label_id"] == class_id)),
                    "generic_validation": int(
                        np.sum(self.generic_validation_df["label_id"] == class_id)
                    ),
                    "generic_internal_test": int(
                        np.sum(self.generic_internal_test_df["label_id"] == class_id)
                    ),
                }
            )
        generic_support_df = pd.DataFrame(generic_support_rows)
        self.log("\nSMALL GENERIC MODEL SUPPORT — EVERY CLASS IS PRESENT")
        self.display(generic_support_df)
        with pd.ExcelWriter(
            self.DIRS["dataset"] / "global_splits_v12_2.xlsx", engine="openpyxl"
        ) as writer:
            split_summary.to_excel(writer, sheet_name="Summary", index=False)
            generic_support_df.to_excel(writer, sheet_name="GenericSupport", index=False)
            self.generic_train_df.to_excel(writer, sheet_name="GenericTrain", index=False)
            self.generic_validation_df.to_excel(writer, sheet_name="GenericValidation", index=False)
            self.generic_internal_test_df.to_excel(
                writer, sheet_name="GenericInternalTest", index=False
            )
            self.private_train_df.to_excel(writer, sheet_name="PrivateFLTrain", index=False)
            self.global_validation_part_a_df.to_excel(
                writer, sheet_name="ValidationPartA", index=False
            )
            self.global_validation_part_b_df.to_excel(
                writer, sheet_name="ValidationPartB", index=False
            )
            self.online_df.to_excel(writer, sheet_name="OnlinePool", index=False)
            self.global_test_df.to_excel(writer, sheet_name="FinalTest", index=False)
