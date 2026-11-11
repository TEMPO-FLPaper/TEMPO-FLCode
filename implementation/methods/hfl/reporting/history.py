"""Reference-derived reporting/history_hfl. See README.md."""

import pandas as pd

import matplotlib.pyplot as plt

class Operations:
    # Reference cells (zero-based): {'hfl': 25}
    def write_history_figures(self):
        if self.STATE["round_history"]:
            round_history_df = pd.DataFrame(self.STATE["round_history"])
            plt.figure(figsize=(10, 4))
            plt.plot(
                round_history_df["round"],
                round_history_df["mean_local_accuracy"],
                marker=".",
                label="Mean local accuracy",
            )
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.title("Hierarchical FedAvg V12.2 — Mean Local Accuracy")
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.DIRS["figures"] / "fedavg_mean_local_accuracy_v12_2.png", dpi=160)
            plt.close()
            validation_df = round_history_df[round_history_df["evaluation_round"]].copy()
            if not validation_df.empty:
                plt.figure(figsize=(10, 4))
                plt.plot(
                    validation_df["round"],
                    validation_df["validation_macro_f1"],
                    marker="o",
                    label="L3 validation Macro-F1",
                )
                plt.xlabel("Round")
                plt.ylabel("Macro-F1")
                plt.title("Hierarchical FedAvg V12.2 — Validation Macro-F1")
                plt.grid(alpha=0.25)
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    self.DIRS["figures"] / "fedavg_l3_validation_macro_f1_v12_2.png", dpi=160
                )
                plt.close()
