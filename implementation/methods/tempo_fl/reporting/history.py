"""Reference-derived reporting/history_tempo. See README.md."""

import pandas as pd

import matplotlib.pyplot as plt

class Operations:
    # Reference cells (zero-based): {'tempo': 50}
    def write_history_figures(self):
        client_history_df = pd.DataFrame(self.STATE["client_round_history"])
        context_history_df = pd.DataFrame(self.STATE["context_round_history"])
        campus_history_df = pd.DataFrame(self.STATE["campus_round_history"])
        if not campus_history_df.empty:
            eval_campus = campus_history_df[campus_history_df["evaluation_round"] == True]
            if len(eval_campus) > 0:
                plt.figure(figsize=(10, 4))
                plt.plot(
                    eval_campus["round"],
                    eval_campus["campus_macro_f1"],
                    marker="o",
                    label="L3 Macro-F1",
                )
                plt.plot(
                    eval_campus["round"],
                    eval_campus["campus_accuracy"],
                    marker="s",
                    label="L3 Accuracy",
                )
                self.add_phase_boundaries()
                plt.xlabel("Federated round")
                plt.ylabel("Score")
                plt.title("V12 L3 Campus Progress Across 100 HFL Rounds")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.DIRS["figures"] / "v9_l3_progress.png", dpi=160)
                plt.close()
        if not context_history_df.empty:
            eval_context = context_history_df[context_history_df["evaluation_round"] == True]
            if len(eval_context) > 0:
                plt.figure(figsize=(10, 4))
                for context_name in self.CONFIG["context_names"]:
                    part = eval_context[eval_context["context"] == context_name]
                    plt.plot(
                        part["round"], part["context_macro_f1"], marker="o", label=context_name
                    )
                self.add_phase_boundaries()
                plt.xlabel("Federated round")
                plt.ylabel("L2 Macro-F1")
                plt.title("V12 Context L2 Progress")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.DIRS["figures"] / "v9_l2_context_progress.png", dpi=160)
                plt.close()
        if not client_history_df.empty:
            mean_round = (
                client_history_df.groupby("round")[["l1_shadow_reliance_rate", "unknown_rate"]]
                .mean()
                .reset_index()
            )
            plt.figure(figsize=(10, 4))
            plt.plot(
                mean_round["round"],
                mean_round["l1_shadow_reliance_rate"],
                label="Mean shadow reliance",
            )
            plt.plot(mean_round["round"], mean_round["unknown_rate"], label="Mean Unknown rate")
            self.add_phase_boundaries()
            plt.xlabel("Round")
            plt.ylabel("Rate")
            plt.title("V12 Specialist Reliance and Unknown Across Phases")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.DIRS["figures"] / "v9_reliance_unknown.png", dpi=160)
            plt.close()
            eval_clients = client_history_df[client_history_df["evaluation_round"] == True]
            if len(eval_clients) > 0:
                comparison = (
                    eval_clients.groupby("round")[
                        [
                            "l1_only_operational_macro_f1",
                            "operational_macro_f1",
                            "forced_cascade_macro_f1",
                        ]
                    ]
                    .mean()
                    .reset_index()
                )
                plt.figure(figsize=(10, 4))
                plt.plot(
                    comparison["round"],
                    comparison["l1_only_operational_macro_f1"],
                    marker="o",
                    label="L1 specialist only",
                )
                plt.plot(
                    comparison["round"],
                    comparison["operational_macro_f1"],
                    marker="s",
                    label="Operational TEMPO cascade",
                )
                plt.plot(
                    comparison["round"],
                    comparison["forced_cascade_macro_f1"],
                    marker="^",
                    label="Forced cascade audit",
                )
                self.add_phase_boundaries()
                plt.xlabel("Round")
                plt.ylabel("Mean Macro-F1")
                plt.title("V12 Specialist vs Hierarchical Shadow Cascade")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.DIRS["figures"] / "v9_specialist_vs_cascade.png", dpi=160)
                plt.close()
