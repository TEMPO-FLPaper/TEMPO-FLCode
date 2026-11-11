"""Atomic complete-round checkpoints, including method state and random generators."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Checkpoints:
    def latest_pointer_path(self):
        return self.DIRS["checkpoints"] / "latest.json"

    def round_checkpoint_dir(self, round_id):
        return self.DIRS["checkpoints"] / f"round_{round_id:03d}"

    def checkpoint_exists(self):
        return self.latest_pointer_path().is_file()

    def rng_snapshot(self):
        import numpy as np
        import tensorflow as tf

        result = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "generator": tf.random.get_global_generator().state.numpy(),
        }
        for name in ("augmenter", "worker_model", "eval_model"):
            model = getattr(self, name)
            result[name] = [
                v.numpy()
                for v in model.variables
                if "seed_generator" in (v.path if hasattr(v, "path") else v.name)
            ]
        return result

    def save_round_checkpoint(self, round_id, state, round_tables=None):
        target = self.round_checkpoint_dir(round_id)
        staging = target.with_name(target.name + ".incomplete")
        if staging.exists():
            if staging.resolve().parent != self.DIRS["checkpoints"].resolve():
                raise ValueError("Invalid checkpoint staging path.")
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        with (staging / "state.pkl").open("wb") as handle:
            pickle.dump(
                {"state": state, "rng": self.rng_snapshot()},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        if round_tables:
            for name, frame in round_tables.items():
                frame.to_csv(staging / (name + ".csv"), index=False)
        manifest = {
            "schema_version": 1,
            "method": self.run_config.method,
            "configuration_sha256": self.run_config.fingerprint(),
            "dataset_sha256": self.dataset_identity,
            "implementation_sha256": self.implementation_identity,
            "round": int(round_id),
            "checkpoint_complete": True,
            "state_sha256": file_sha256(staging / "state.pkl"),
            "tensorflow": self.environment["packages"].get("tensorflow"),
            "keras": self.environment["packages"].get("keras"),
        }
        atomic_json(staging / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(f"Committed checkpoint already exists: {target}")
        os.replace(staging, target)
        atomic_json(self.latest_pointer_path(), {"latest_round": int(round_id)})
        self.publish_results(state)
        self.emit({"type": "checkpoint", "round": int(round_id), "path": str(target)})
        self.log(f"Checkpoint committed: round {round_id:03d}")

    def load_latest_checkpoint(self):
        import numpy as np
        import tensorflow as tf

        pointer = json.loads(self.latest_pointer_path().read_text(encoding="utf-8"))
        directory = self.round_checkpoint_dir(int(pointer["latest_round"]))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "method": self.run_config.method,
            "configuration_sha256": self.run_config.fingerprint(),
            "dataset_sha256": self.dataset_identity,
            "implementation_sha256": self.implementation_identity,
            "tensorflow": self.environment["packages"].get("tensorflow"),
            "keras": self.environment["packages"].get("keras"),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Checkpoint incompatible: {key}. Use the original method, data, settings and software."
                )
        if (
            not manifest.get("checkpoint_complete")
            or file_sha256(directory / "state.pkl") != manifest["state_sha256"]
        ):
            raise ValueError("Checkpoint incomplete or corrupt.")
        with (directory / "state.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        tf.random.get_global_generator().state.assign(rng["generator"])
        for name in ("augmenter", "worker_model", "eval_model"):
            variables = [
                v
                for v in getattr(self, name).variables
                if "seed_generator" in (v.path if hasattr(v, "path") else v.name)
            ]
            if len(variables) != len(rng[name]):
                raise ValueError(f"Checkpoint incompatible: {name} random-generator structure.")
            for variable, value in zip(variables, rng[name]):
                if tuple(variable.shape) != tuple(value.shape):
                    raise ValueError(f"Checkpoint incompatible: {name} random-generator shape.")
                variable.assign(value)
        state = payload["state"]
        if state["current_round"] != manifest["round"]:
            raise ValueError("Checkpoint round does not match manifest.")
        self.emit({"type": "restored", "round": state["current_round"]})
        return state
