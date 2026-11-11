"""Reference order for data preparation, ownership, and reusable model workers."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path


def prepare_data(e):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from implementation.shared.data.acquisition import acquire

    s = e.CONFIG
    e.set_round_seed(0)
    e.stage("Dataset discovery and validation")
    root = (
        Path(s["custom_dataset_dir"])
        if s["dataset_mode"] == "local_folder"
        else acquire(e.dataset_cache, s["trashnet_zip_url"], e.emit, e.stopped)
    )
    e.dataset_root = root
    e.full_df, e.CLASS_NAMES, e.CLASS_TO_ID = e.scan_classification_folder(root)
    if e.CLASS_NAMES != ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
        raise ValueError(f"Expected six TrashNet classes; found {e.CLASS_NAMES}.")
    e.NUM_CLASSES = len(e.CLASS_NAMES)
    e.stage("Reference splits and owned-class allocation")
    e.train_initial_df, e.global_validation_df, e.online_df, e.global_test_df = (
        e.stratified_global_split(e.full_df, s["seed"])
    )
    e.generic_train_df, e.generic_validation_df, e.generic_internal_test_df, e.private_train_df = (
        e.build_balanced_generic_sets(e.train_initial_df, s["seed"] + 10)
    )
    fraction = s.get("calibration_fraction_of_validation", s.get("validation_subsplit_fraction"))
    a, b = train_test_split(
        e.global_validation_df,
        train_size=fraction,
        stratify=e.global_validation_df["label_id"],
        random_state=s["seed"] + 12,
    )
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    e.global_calibration_df, e.global_model_gate_df = a, b
    e.global_validation_part_a_df, e.global_validation_part_b_df = a, b
    e.generic_seed_df = pd.concat(
        [e.generic_train_df, e.generic_validation_df, e.generic_internal_test_df], ignore_index=True
    )
    e.generic_fit_df = e.generic_train_df.copy()
    e.generic_earlystop_df = e.generic_validation_df.copy()
    e.l3_initial_train_df = e.generic_train_df.copy()
    e.l3_fit_df = e.generic_train_df.copy()
    e.l3_earlystop_df = e.generic_validation_df.copy()
    e.server_proxy_df = e.generic_train_df.copy()
    e.CLIENT_IDS = [f"C{i:02d}" for i in range(s["num_clients"])]
    e.CONTEXT_TO_CLIENTS = {
        name: e.CLIENT_IDS[i * s["clients_per_context"] : (i + 1) * s["clients_per_context"]]
        for i, name in enumerate(s["context_names"])
    }
    e.CLIENT_TO_CONTEXT = {cid: name for name, cids in e.CONTEXT_TO_CLIENTS.items() for cid in cids}
    e.CLIENT_PLANNED_CLASSES = {
        cid: [name for name in s["planned_classes"][cid] if name in e.CLASS_NAMES]
        for cid in e.CLIENT_IDS
    }
    e.client_train = e.partition_by_planned_classes(
        e.private_train_df, s["seed"] + 20, s["min_client_train_samples"]
    )
    e.client_calibration_local = e.partition_by_planned_classes(a, s["seed"] + 21)
    e.client_gate_local = e.partition_by_planned_classes(b, s["seed"] + 22)
    e.client_validation_part_a = e.client_calibration_local
    e.client_validation_part_b = e.client_gate_local
    e.client_online_pool = e.partition_by_planned_classes(e.online_df, s["seed"] + 23)
    e.client_test_local = e.partition_by_planned_classes(e.global_test_df, s["seed"] + 24)
    e.general_test_set = e.global_test_df.copy().reset_index(drop=True)
    e.owned_class_test_set = {
        cid: e.client_test_local[cid].copy().reset_index(drop=True) for cid in e.CLIENT_IDS
    }
    e.client_val_local = {
        cid: pd.concat(
            [e.client_calibration_local[cid], e.client_gate_local[cid]], ignore_index=True
        )
        for cid in e.CLIENT_IDS
    }
    for cid in e.CLIENT_IDS:
        e.client_val_local[cid]["is_planned_class"] = e.client_val_local[cid]["label"].isin(
            e.CLIENT_PLANNED_CLASSES[cid]
        )
    for name in (
        "full_df",
        "train_initial_df",
        "global_validation_df",
        "online_df",
        "global_test_df",
        "generic_train_df",
        "generic_validation_df",
        "generic_internal_test_df",
        "private_train_df",
        "global_calibration_df",
        "global_model_gate_df",
    ):
        getattr(e, name).to_csv(e.DIRS["dataset"] / f"{name}.csv", index=False)
    parts = []
    for role, allocation in [
        ("train", e.client_train),
        ("calibration", e.client_calibration_local),
        ("gate", e.client_gate_local),
        ("online", e.client_online_pool),
        ("test", e.client_test_local),
    ]:
        for cid, frame in allocation.items():
            piece = frame.copy()
            piece["role"] = role
            piece["client_id"] = cid
            parts.append(piece)
    pd.concat(parts, ignore_index=True).to_csv(
        e.DIRS["dataset"] / "partition_manifest.csv", index=False
    )
    records = []
    for row in e.full_df.itertuples():
        e.check_stop()
        p = Path(row.path)
        records.append(
            {
                "path": str(p.resolve()),
                "label_id": int(row.label_id),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
    e.dataset_identity = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    (e.DIRS["dataset"] / "identity.json").write_text(
        json.dumps(
            {
                "name": "TrashNet",
                "classes": e.CLASS_TO_ID,
                "sha256": e.dataset_identity,
                "images": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (e.DIRS["dataset"] / "ownership.json").write_text(
        json.dumps(e.CLIENT_PLANNED_CLASSES, indent=2), encoding="utf-8"
    )
    e.log(f"TrashNet: {len(e.full_df)} images; labels={e.CLASS_TO_ID}; partition manifest saved.")
    e.stage("Dataset statistics and reference partition reports")
    e.report_dataset()
    e.report_splits()
    e.report_client_partitions()


def prepare_models(e):
    import gc
    from tensorflow import keras
    from tensorflow.keras import layers
    from implementation.shared.models.weights_cache import ensure_imagenet_weights

    s = e.CONFIG
    e.stage("MobileNetV2 construction and ImageNet weights")
    initialization = ensure_imagenet_weights(s, e.emit, e.stopped)
    (e.ROOT / "model_initialization.json").write_text(
        json.dumps(initialization, indent=2) + "\n", encoding="utf-8"
    )
    e.IMG_SIZE = s["image_size"]
    e.DATA_PARALLEL_CALLS = s["tf_data_num_parallel_calls"]
    e.DATA_PREFETCH = s["tf_data_prefetch_batches"]
    e.augmenter = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="light_augmentation",
    )
    # Preserve the disposable prototype's RNG consumption and notebook execution order.
    prototype = e.build_tempo_model(e.NUM_CLASSES)
    e.log(f"Model parameters: {prototype.count_params():,}")
    del prototype
    gc.collect()
    e.eval_model = e.build_tempo_model(e.NUM_CLASSES)
    e.worker_model = e.build_tempo_model(e.NUM_CLASSES)
    e.classification_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
