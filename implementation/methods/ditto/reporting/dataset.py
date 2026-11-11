"""Reference-derived reporting/dataset_ditto. See README.md."""

import math

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

class Operations:
    # Reference cells (zero-based): {'ditto': 9}
    def report_dataset(self):
        class_stats = (
            self.full_df.groupby("label")
            .size()
            .reindex(self.CLASS_NAMES, fill_value=0)
            .reset_index(name="count")
        )
        class_stats["percent"] = 100.0 * class_stats["count"] / len(self.full_df)
        class_stats["class_id"] = class_stats["label"].map(self.CLASS_TO_ID)
        image_stats = pd.DataFrame(
            [
                {
                    "dataset": self.CONFIG["dataset_name"],
                    "total_images": len(self.full_df),
                    "num_classes": self.NUM_CLASSES,
                    "min_class_count": int(class_stats["count"].min()),
                    "max_class_count": int(class_stats["count"].max()),
                    "class_imbalance_ratio_max_over_min": float(
                        class_stats["count"].max() / max(class_stats["count"].min(), 1)
                    ),
                    "min_width": int(self.full_df["width"].min()),
                    "max_width": int(self.full_df["width"].max()),
                    "mean_width": float(self.full_df["width"].mean()),
                    "min_height": int(self.full_df["height"].min()),
                    "max_height": int(self.full_df["height"].max()),
                    "mean_height": float(self.full_df["height"].mean()),
                }
            ]
        )
        self.log("=" * 90)
        self.log("Ditto V12.2 — DATASET SUMMARY BEFORE TRAINING")
        self.log("=" * 90)
        self.log("Dataset root:", self.dataset_root)
        self.display(image_stats)
        self.log("\nCLASS DISTRIBUTION")
        self.display(class_stats[["class_id", "label", "count", "percent"]])
        plt.figure(figsize=(9, 4))
        plt.bar(class_stats["label"], class_stats["count"])
        plt.xlabel("Class")
        plt.ylabel("Images")
        plt.title("TrashNet Class Distribution — Before Training")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(self.DIRS["figures"] / "dataset_class_distribution_v12_2.png", dpi=160)
        plt.close()
        with pd.ExcelWriter(
            self.DIRS["dataset"] / "dataset_statistics_v12_2.xlsx", engine="openpyxl"
        ) as writer:
            image_stats.to_excel(writer, sheet_name="DatasetSummary", index=False)
            class_stats.to_excel(writer, sheet_name="ClassDistribution", index=False)
            self.full_df.to_excel(writer, sheet_name="AllImages", index=False)
        plt.figure(figsize=(9, 4))
        plt.bar(class_stats["label"], class_stats["count"])
        plt.title("TrashNet Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Images")
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.savefig(self.DIRS["figures"] / "dataset_class_distribution.png", dpi=160)
        plt.close()
        sample_count = min(12, len(self.full_df))
        samples = self.full_df.sample(sample_count, random_state=self.CONFIG["seed"]).reset_index(
            drop=True
        )
        cols = 4
        rows = math.ceil(sample_count / cols)
        plt.figure(figsize=(12, 3 * rows))
        for i, row in samples.iterrows():
            plt.subplot(rows, cols, i + 1)
            image = Image.open(row["path"]).convert("RGB")
            plt.imshow(image)
            plt.title(row["label"])
            plt.axis("off")
        plt.suptitle("Example TrashNet Images Before Training", y=1.02)
        plt.tight_layout()
        plt.savefig(self.DIRS["figures"] / "dataset_examples.png", dpi=160, bbox_inches="tight")
        plt.close()
