"""Experiment lifecycle; method modules contain explicit reference calculations."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import time
from implementation.persistence.checkpoints import Checkpoints, atomic_json
from implementation.shared.reporting.results import publish, json_safe


class ExperimentBase(Checkpoints):
    def __init__(self, run_config, run_dir, emit, environment):
        self.run_config = run_config
        self.CONFIG = deepcopy(run_config.settings)
        self.ROOT = Path(run_dir).resolve()
        self.CONFIG["project_root"] = str(self.ROOT)
        self.dataset_cache = Path(run_config.dataset_cache).resolve()
        self.emit = emit
        self.environment = environment
        self.implementation_identity = environment["implementation_sha256"]
        self.DIRS = {
            k: self.ROOT / v
            for k, v in {
                "config": "00_config",
                "dataset": "01_dataset",
                "initial": "02_initial_models",
                "checkpoints": "03_checkpoints",
                "master_metrics": "04_master_metrics",
                "figures": "05_figures",
                "exports": "06_exports",
            }.items()
        }
        self.DIRS["root"] = self.ROOT
        for p in self.DIRS.values():
            p.mkdir(parents=True, exist_ok=True)
        self.GPU_DEVICES = []
        self.FORCED_EVAL_ROUNDS = {
            1,
            5,
            10,
            15,
            20,
            21,
            22,
            23,
            40,
            41,
            45,
            50,
            55,
            60,
            65,
            66,
            70,
            80,
            90,
            100,
        }
        self.log_file = (self.ROOT / "experiment.log").open("a", encoding="utf-8", buffering=1)
        self.current_stage = "Starting"
        self.extra_result_tables = {}

    def stopped(self):
        return (self.ROOT / "stop.request").exists()

    def check_stop(self):
        if self.stopped():
            raise InterruptedError(
                "Cooperative stop requested. Last committed checkpoint is preserved."
            )

    def stage(self, name):
        self.check_stop()
        self.current_stage = name
        self.emit({"type": "stage", "stage": name, "total": None})
        self.log(name)

    def log(self, *parts, **kwargs):
        message = kwargs.get("sep", " ").join(str(p) for p in parts)
        self.log_file.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + message + "\n")
        self.emit({"type": "log", "message": message[-5000:]})

    def display(self, value):
        self.log(
            f"Report table: {value.shape[0]} rows; see run reports."
            if hasattr(value, "shape")
            else value
        )

    def progress_iter(self, iterable, total=None, desc="Working"):
        for index, item in enumerate(iterable):
            yield item
            self.emit({"type": "progress", "stage": desc, "current": index + 1, "total": total})

    def training_callback(self):
        from tensorflow import keras

        e = self

        class Progress(keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                e.check_stop()

            def on_epoch_end(self, epoch, logs=None):
                e.emit(
                    {
                        "type": "epoch",
                        "stage": e.current_stage,
                        "epoch": epoch + 1,
                        "metrics": json_safe(logs or {}),
                    }
                )
                e.check_stop()

        return Progress()

    def publish_results(self, state):
        publish(self, state)

    def report_tables(self, tables):
        self.extra_result_tables.update(
            {key: frame.to_dict("records") for key, frame in tables.items()}
        )
        atomic_json(self.ROOT / "report_tables.json", json_safe(self.extra_result_tables))

    def tflite_converter_factory(self, model):
        from implementation.shared.reporting.converter_compat import converter_factory

        return converter_factory(model, self.DIRS["exports"])

    def run(self, resume=False):
        import tensorflow as tf
        from implementation.shared.data.preparation import prepare_data, prepare_models

        self.GPU_DEVICES = tf.config.list_physical_devices("GPU")
        for gpu in self.GPU_DEVICES:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                self.log("GPU memory growth:", exc)
        prepare_data(self)
        prepare_models(self)
        if resume:
            self.stage("Restoring compatible round checkpoint")
            if not self.checkpoint_exists():
                raise ValueError("No committed checkpoint to resume. Start a new experiment.")
            self.STATE = self.load_latest_checkpoint()
        else:
            if self.checkpoint_exists():
                raise FileExistsError("Checkpoint already exists. Use Resume or choose a new run.")
            self.STATE = self.initialize_from_scratch()
        for round_id in range(int(self.STATE["current_round"]) + 1, self.CONFIG["num_rounds"] + 1):
            self.stage(f"Operational round {round_id}")
            self.emit(
                {
                    "type": "round",
                    "current": round_id,
                    "total": self.CONFIG["num_rounds"],
                    "completed": self.STATE["current_round"],
                }
            )
            self.run_round(round_id)
            self.check_stop()
        self.stage("Reference history figures")
        self.write_history_figures()
        self.stage("Final evaluation and reference-compatible reports")
        self.write_final_report()
        self.publish_results(self.STATE)
        self.stage("Model deployment exports")
        self.write_deployment_exports()
        atomic_json(
            self.ROOT / "complete.json",
            {"round": self.STATE["current_round"], "requested_rounds": self.CONFIG["num_rounds"]},
        )
        from implementation.persistence.archive import archive_experiment

        archive_experiment(self)
        return self.STATE["current_round"]
