"""Measured results with explicit protocol and metric identities."""

from __future__ import annotations
import json
import math
from pathlib import Path
from implementation.persistence.checkpoints import atomic_json


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def publish(e, state):
    keys = [
        "general_accuracy_rows",
        "owned_accuracy_rows",
        "round_history",
        "communication_history",
        "client_round_history",
        "context_round_history",
        "campus_round_history",
        "operational_round_history",
        "promotion_history",
        "shift_eligibility_history",
        "client_operational_counters",
        "threshold_diagnostics",
    ]
    result = {
        "schema_version": 1,
        "method": e.run_config.method,
        "preset": e.run_config.preset,
        "protocol_label": e.run_config.protocol_label,
        "configuration_sha256": e.run_config.fingerprint(),
        "dataset_sha256": e.dataset_identity,
        "current_round": state["current_round"],
        "requested_rounds": e.CONFIG["num_rounds"],
        "departures": e.run_config.departures(),
        "metric_warning": "Legacy TEMPO override-audit counts are defective; do not interpret them as beneficial/harmful counts.",
        "tables": {k: state[k] for k in keys if k in state},
    }
    result["tables"].update(getattr(e, "extra_result_tables", {}))
    atomic_json(e.ROOT / "results.json", json_safe(result))
    metrics = []
    for key in ("round_history", "campus_round_history", "client_round_history"):
        for row in state.get(key, []):
            if row.get("round") == state["current_round"]:
                metrics.append({"table": key, **json_safe(row)})
    e.emit({"type": "metrics", "round": state["current_round"], "rows": metrics})


def load_results(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported results schema.")
    return result


def protocol_compatible(a, b):
    return all(
        a.get(k) == b.get(k)
        for k in ("method", "configuration_sha256", "dataset_sha256", "protocol_label")
    )
