"""Isolated export-only check using a trusted completed verification snapshot.

This is not experiment resume. It reuses measured weights solely to exercise the
source export path and validate the resulting TensorFlow Lite tensors.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["MPLBACKEND"] = "Agg"
os.environ["KERAS_HOME"] = str(ROOT / ".cache/keras")
from implementation.config.schema import RunConfig
from implementation.execution.provenance import environment
from implementation.methods.flat import Experiment
from implementation.shared.data.preparation import prepare_data, prepare_models
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
args = parser.parse_args()
source = Path(args.run_dir)
config = RunConfig.load(source / "resolved_config.json")
assert config.method == "flat"
root = ROOT / ".verification" / ("export_" + time.strftime("%Y%m%d_%H%M%S"))
root.mkdir(parents=True)
e = Experiment(
    config,
    root,
    lambda event: print(json.dumps(event), flush=True) if event["type"] != "log" else None,
    environment(),
)
prepare_data(e)
prepare_models(e)
pointer = json.loads((source / "03_checkpoints/latest.json").read_text())
checkpoint = source / "03_checkpoints" / f"round_{pointer['latest_round']:03d}"
with (checkpoint / "state.pkl").open("rb") as handle:
    state = pickle.load(handle)["state"]
weights = state["global_weights"]
path = e.DIRS["exports"] / "verification_global.tflite"
size, mode = e.export_tflite_model(weights, path)
interpreter = tf.lite.Interpreter(model_path=str(path))
interpreter.allocate_tensors()
input_ = interpreter.get_input_details()[0]
output = interpreter.get_output_details()[0]
x, _ = e.load_batch_tensor(e.generic_train_df.head(1), training=False)
values = x.numpy()
if input_["dtype"] == np.int8:
    scale, zero = input_["quantization"]
    values = np.clip(np.rint(values / scale + zero), -128, 127).astype(np.int8)
interpreter.set_tensor(input_["index"], values)
interpreter.invoke()
prediction = interpreter.get_tensor(output["index"])
assert prediction.shape == (1, 6)
assert np.isfinite(prediction).all()
record = {
    "source_run": str(source),
    "export": str(path),
    "size_mb": size,
    "mode": mode,
    "input_dtype": str(input_["dtype"]),
    "output_dtype": str(output["dtype"]),
    "output_shape": list(prediction.shape),
    "interpreter_invocation_passed": True,
}
(ROOT / ".verification/export_evidence.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
e.log_file.close()
print(json.dumps(record, indent=2), flush=True)
