"""Reference-derived reporting/deployment_ditto. See README.md."""

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'ditto': 28}
    def write_deployment_exports(self):
        if (
            self.CONFIG["export_tflite_at_end"]
            and self.STATE["current_round"] >= self.CONFIG["num_rounds"]
        ):
            export_rows = []
            size_mb, quantization_mode = self.export_tflite_model(
                self.STATE["global_weights"], self.DIRS["exports"] / "Global_V12_2_Ditto.tflite"
            )
            export_rows.append(
                {
                    "model_type": "Global",
                    "model_id": "Global",
                    "size_mb": size_mb,
                    "quantization": quantization_mode,
                }
            )
            for cid in self.CLIENT_IDS:
                size_mb, quantization_mode = self.export_tflite_model(
                    self.STATE["personalized_weights"][cid],
                    self.DIRS["exports"] / f"Personalized_{cid}_V12_2_Ditto.tflite",
                )
                export_rows.append(
                    {
                        "model_type": "Personalized",
                        "model_id": cid,
                        "size_mb": size_mb,
                        "quantization": quantization_mode,
                    }
                )
            export_df = pd.DataFrame(export_rows)
            export_df.to_excel(
                self.DIRS["exports"] / "tflite_model_sizes_v12_2_ditto.xlsx", index=False
            )
            self.display(export_df)
