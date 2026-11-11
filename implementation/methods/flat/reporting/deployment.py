"""Reference-derived reporting/deployment_flat. See README.md."""

import pandas as pd

class Operations:
    # Reference cells (zero-based): {'flat': 27}
    def write_deployment_exports(self):
        if (
            self.CONFIG["export_tflite_at_end"]
            and self.STATE["current_round"] >= self.CONFIG["num_rounds"]
        ):
            generic_weights = self.load_weights_from_file(
                self.DIRS["config"] / "generic_seed_v12_2.weights.h5"
            )
            export_rows = []
            size_mb, quantization_mode = self.export_tflite_model(
                generic_weights, self.DIRS["exports"] / "Generic_V12_2_Flat_FedAvg.tflite"
            )
            export_rows.append(
                {"model": "Generic", "size_mb": size_mb, "quantization": quantization_mode}
            )
            size_mb, quantization_mode = self.export_tflite_model(
                self.STATE["global_weights"],
                self.DIRS["exports"] / "Global_V12_2_Flat_FedAvg.tflite",
            )
            export_rows.append(
                {"model": "Global", "size_mb": size_mb, "quantization": quantization_mode}
            )
            export_df = pd.DataFrame(export_rows)
            export_df.to_excel(
                self.DIRS["exports"] / "tflite_model_sizes_v12_2_flat_fedavg.xlsx", index=False
            )
            self.display(export_df)
