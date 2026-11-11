"""Reference-derived reporting/deployment_hfl. See README.md."""

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'hfl': 27}
    def write_deployment_exports(self):
        if (
            self.CONFIG["export_tflite_at_end"]
            and self.STATE["current_round"] >= self.CONFIG["num_rounds"]
        ):
            export_rows = []
            for cid in self.CLIENT_IDS:
                size_mb, quantization_mode = self.export_tflite_model(
                    self.STATE["l1_base_weights"][cid],
                    self.DIRS["exports"] / f"L1_{cid}_V12_2_FedAvg.tflite",
                )
                export_rows.append(
                    {
                        "level": "L1",
                        "id": cid,
                        "size_mb": size_mb,
                        "quantization": quantization_mode,
                    }
                )
            for context_name in self.CONFIG["context_names"]:
                size_mb, quantization_mode = self.export_tflite_model(
                    self.STATE["l2_base_weights"][context_name],
                    self.DIRS["exports"] / f"L2_{context_name}_V12_2_FedAvg.tflite",
                )
                export_rows.append(
                    {
                        "level": "L2",
                        "id": context_name,
                        "size_mb": size_mb,
                        "quantization": quantization_mode,
                    }
                )
            size_mb, quantization_mode = self.export_tflite_model(
                self.STATE["l3_base_weights"],
                self.DIRS["exports"] / "L3_Campus_V12_2_FedAvg.tflite",
            )
            export_rows.append(
                {
                    "level": "L3",
                    "id": "Campus",
                    "size_mb": size_mb,
                    "quantization": quantization_mode,
                }
            )
            export_df = pd.DataFrame(export_rows)
            export_df.to_excel(
                self.DIRS["exports"] / "tflite_model_sizes_v12_2_fedavg.xlsx", index=False
            )
            self.display(export_df)
