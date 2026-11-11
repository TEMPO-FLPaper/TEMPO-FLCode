"""Reference-derived reporting/partitions_tempo. See README.md."""

import pandas as pd

import matplotlib.pyplot as plt

class Operations:
    # Reference cells (zero-based): {'tempo': 17}
    def report_client_partitions(self):
        partition_rows = []
        class_rows = []
        for cid in self.CLIENT_IDS:
            train_df = self.client_train[cid]
            val_df = self.client_val_local[cid]
            train_counts = train_df["label"].value_counts()
            val_counts = val_df["label"].value_counts()
            planned_classes = self.CLIENT_PLANNED_CLASSES[cid]
            planned_train_count = int(train_df["label"].isin(planned_classes).sum())
            row = {
                "client_id": cid,
                "context": self.CLIENT_TO_CONTEXT[cid],
                "planned_classes": ",".join(planned_classes),
                "train_samples": len(train_df),
                "planned_train_samples": planned_train_count,
                "planned_train_fraction": float(planned_train_count / max(len(train_df), 1)),
                "local_val_samples": len(val_df),
                "planned_val_samples": int(val_df["label"].isin(planned_classes).sum()),
                "online_pool_samples": len(self.client_online_pool[cid]),
                "local_test_samples": len(self.client_test_local[cid]),
            }
            for class_name in self.CLASS_NAMES:
                train_count = int(train_counts.get(class_name, 0))
                val_count = int(val_counts.get(class_name, 0))
                is_planned = class_name in planned_classes
                row[f"train_{class_name}"] = train_count
                row[f"val_{class_name}"] = val_count
                class_rows.append(
                    {
                        "client_id": cid,
                        "context": self.CLIENT_TO_CONTEXT[cid],
                        "class_name": class_name,
                        "planned_owned_class": bool(is_planned),
                        "train_count": train_count,
                        "validation_count": val_count,
                    }
                )
            partition_rows.append(row)
        partition_stats = pd.DataFrame(partition_rows)
        client_class_availability = pd.DataFrame(class_rows)
        self.log("\nV12 CLIENT RESPONSIBILITY PLAN AND DATA DISTRIBUTION")
        self.display(
            partition_stats[
                [
                    "client_id",
                    "context",
                    "planned_classes",
                    "train_samples",
                    "planned_train_samples",
                    "planned_train_fraction",
                    "local_val_samples",
                    "planned_val_samples",
                    "online_pool_samples",
                    "local_test_samples",
                ]
            ]
        )
        self.log(
            "\nPER-CLASS SUPPORT AT EACH CLIENT (planned classes are the primary specialist target)"
        )
        self.display(client_class_availability)
        with pd.ExcelWriter(
            self.DIRS["dataset"] / "partition_statistics_v12_2.xlsx", engine="openpyxl"
        ) as writer:
            partition_stats.to_excel(writer, sheet_name="Clients", index=False)
            client_class_availability.to_excel(writer, sheet_name="ClientClassLong", index=False)
        matrix = partition_stats[
            [f"train_{class_name}" for class_name in self.CLASS_NAMES]
        ].to_numpy()
        plt.figure(figsize=(10, 5))
        plt.imshow(matrix, aspect="auto")
        plt.xticks(range(self.NUM_CLASSES), self.CLASS_NAMES, rotation=30)
        plt.yticks(range(self.CONFIG["num_clients"]), partition_stats["client_id"])
        plt.colorbar(label="Private training images")
        plt.title("V12 Client-Class Distribution Before Training")
        plt.tight_layout()
        plt.savefig(self.DIRS["figures"] / "client_training_distribution_v12_2.png", dpi=160)
        plt.close()
