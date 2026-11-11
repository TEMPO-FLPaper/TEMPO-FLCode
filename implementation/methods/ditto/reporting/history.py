"""Reference-derived reporting/history_ditto. See README.md."""

import pandas as pd

import matplotlib.pyplot as plt

class Operations:
    # Reference cells (zero-based): {'ditto': 26}
    def write_history_figures(self):
        if self.STATE["round_history"]:
            round_history_df = pd.DataFrame(self.STATE["round_history"])
            plt.figure(figsize=(10, 4))
            plt.plot(
                round_history_df["round"],
                round_history_df["mean_global_local_accuracy"],
                marker=".",
                label="Mean global local accuracy",
            )
            plt.plot(
                round_history_df["round"],
                round_history_df["mean_personal_local_accuracy"],
                marker=".",
                label="Mean personalized local accuracy",
            )
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.title("Ditto V12.2 — Mean Local Accuracies")
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.DIRS["figures"] / "ditto_mean_local_accuracy_v12_2.png", dpi=160)
            plt.close()
            validation_df = round_history_df[round_history_df["evaluation_round"]].copy()
            if not validation_df.empty:
                plt.figure(figsize=(10, 4))
                plt.plot(
                    validation_df["round"],
                    validation_df["global_validation_macro_f1"],
                    marker="o",
                    label="Global validation Macro-F1",
                )
                plt.plot(
                    validation_df["round"],
                    validation_df["personalized_validation_mean_macro_f1"],
                    marker="o",
                    label="Personalized mean validation Macro-F1",
                )
                plt.xlabel("Round")
                plt.ylabel("Macro-F1")
                plt.title("Ditto V12.2 — Validation Macro-F1")
                plt.grid(alpha=0.25)
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.DIRS["figures"] / "ditto_validation_macro_f1_v12_2.png", dpi=160)
                plt.close()
