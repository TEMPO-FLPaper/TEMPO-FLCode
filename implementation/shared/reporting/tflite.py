"""Reference-derived reporting/tflite. See README.md."""

import gc

from pathlib import Path

import numpy as np

import pandas as pd

import tensorflow as tf

class Operations:
    # Reference cells (zero-based): {'tempo': 52, 'ditto': 28, 'hfl': 27, 'flat': 27}
    def representative_dataset_generator(self):
        rep_df = self.server_proxy_df.head(
            min(self.CONFIG["tflite_representative_samples"], len(self.server_proxy_df))
        ).copy()
        for _, row in rep_df.iterrows():
            mini_df = pd.DataFrame([{"path": row["path"], "label_id": int(row["label_id"])}])
            x_batch, _ = self.load_batch_tensor(mini_df, training=False)
            yield [x_batch.numpy().astype(np.float32)]

    # Reference cells (zero-based): {'tempo': 52, 'ditto': 28, 'hfl': 27, 'flat': 27}
    def export_tflite_model(self, weights, output_path):
        model = self.build_tempo_model(self.NUM_CLASSES)
        model.set_weights(weights)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.tflite_converter_factory(model) as make_converter:
            quantization_mode = "dynamic_range"
            try:
                if self.CONFIG["prefer_full_int8_tflite"]:
                    converter = make_converter()
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = self.representative_dataset_generator
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                    tflite_model = converter.convert()
                    quantization_mode = "full_int8"
                else:
                    raise RuntimeError("Full INT8 disabled by config.")
            except Exception as exc:
                self.log("Full-int8 conversion failed; using dynamic-range fallback.")
                self.log("Reason:", type(exc).__name__, str(exc)[:300])
                converter = make_converter()
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                quantization_mode = "dynamic_range_fallback"
            output_path.write_bytes(tflite_model)
            size_mb = output_path.stat().st_size / 1024**2
            del model
            gc.collect()
            return (size_mb, quantization_mode)
