"""Reference-derived data/partitions. See README.md."""

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

class Operations:
    # Reference cells (zero-based): {'tempo': 12, 'ditto': 11, 'hfl': 11, 'flat': 11}
    def stratified_global_split(self, df, seed):
        train_df, temp_df = train_test_split(
            df, test_size=0.4, stratify=df["label_id"], random_state=seed
        )
        validation_df, remainder_df = train_test_split(
            temp_df, test_size=0.75, stratify=temp_df["label_id"], random_state=seed + 1
        )
        online_df, test_df = train_test_split(
            remainder_df, test_size=0.5, stratify=remainder_df["label_id"], random_state=seed + 2
        )
        return (
            train_df.reset_index(drop=True),
            validation_df.reset_index(drop=True),
            online_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    # Reference cells (zero-based): {'tempo': 12, 'ditto': 11, 'hfl': 11, 'flat': 11}
    def build_balanced_generic_sets(self, train_df, seed):
        rng = np.random.default_rng(seed)
        train_rows = []
        val_rows = []
        test_rows = []
        selected_paths = set()
        n_train = int(self.CONFIG["generic_train_per_class"])
        n_val = int(self.CONFIG["generic_validation_per_class"])
        n_test = int(self.CONFIG["generic_internal_test_per_class"])
        total_needed = n_train + n_val + n_test
        for class_id, class_name in enumerate(self.CLASS_NAMES):
            class_df = train_df[train_df["label_id"] == class_id].copy().reset_index(drop=True)
            if len(class_df) < total_needed:
                raise RuntimeError(
                    f"V12 generic seed requires {total_needed} examples of class {class_name}, but only {len(class_df)} are available."
                )
            order = rng.permutation(len(class_df))
            chosen = class_df.iloc[order[:total_needed]].copy().reset_index(drop=True)
            train_part = chosen.iloc[:n_train].copy()
            val_part = chosen.iloc[n_train : n_train + n_val].copy()
            test_part = chosen.iloc[n_train + n_val :].copy()
            train_rows.append(train_part)
            val_rows.append(val_part)
            test_rows.append(test_part)
            selected_paths.update(chosen["path"].astype(str))
        generic_train_df = (
            pd.concat(train_rows, ignore_index=True)
            .sample(frac=1.0, random_state=seed + 101)
            .reset_index(drop=True)
        )
        generic_validation_df = (
            pd.concat(val_rows, ignore_index=True)
            .sample(frac=1.0, random_state=seed + 102)
            .reset_index(drop=True)
        )
        generic_internal_test_df = (
            pd.concat(test_rows, ignore_index=True)
            .sample(frac=1.0, random_state=seed + 103)
            .reset_index(drop=True)
        )
        private_train_df = (
            train_df[~train_df["path"].astype(str).isin(selected_paths)]
            .copy()
            .reset_index(drop=True)
        )
        return (generic_train_df, generic_validation_df, generic_internal_test_df, private_train_df)

    # Reference cells (zero-based): {'tempo': 14, 'ditto': 12, 'hfl': 12, 'flat': 12}
    def client_class_assignment_weight(self, cid, class_name):
        return (
            self.CONFIG["planned_class_assignment_weight"]
            if class_name in self.CLIENT_PLANNED_CLASSES[cid]
            else self.CONFIG["background_class_assignment_weight"]
        )

    # Reference cells (zero-based): {'tempo': 15, 'ditto': 13, 'hfl': 13, 'flat': 13}
    def allocate_class_indices_to_clients(self, class_indices, class_name, seed):
        rng = np.random.default_rng(seed)
        class_indices = np.asarray(class_indices, dtype=np.int64).copy()
        rng.shuffle(class_indices)
        owners = [cid for cid in self.CLIENT_IDS if class_name in self.CLIENT_PLANNED_CLASSES[cid]]
        allocations = {cid: [] for cid in self.CLIENT_IDS}
        cursor = 0
        min_owned = int(self.CONFIG["min_owned_samples_per_split_when_possible"])
        if min_owned > 0 and len(class_indices) >= min_owned * len(owners):
            for cid in owners:
                take = class_indices[cursor : cursor + min_owned].tolist()
                allocations[cid].extend(take)
                cursor += min_owned
        remaining = class_indices[cursor:]
        if len(remaining) > 0:
            weights = np.asarray(
                [self.client_class_assignment_weight(cid, class_name) for cid in self.CLIENT_IDS],
                dtype=np.float64,
            )
            weights = weights / weights.sum()
            counts = rng.multinomial(len(remaining), weights)
            local_cursor = 0
            for cid, count in zip(self.CLIENT_IDS, counts):
                allocations[cid].extend(remaining[local_cursor : local_cursor + count].tolist())
                local_cursor += count
        return allocations

    # Reference cells (zero-based): {'tempo': 15, 'ditto': 13, 'hfl': 13, 'flat': 13}
    def partition_by_planned_classes(self, df, seed, min_client_samples=0):
        allocations = {cid: [] for cid in self.CLIENT_IDS}
        for class_id, class_name in enumerate(self.CLASS_NAMES):
            class_indices = df.index[df["label_id"] == class_id].to_numpy()
            local_allocations = self.allocate_class_indices_to_clients(
                class_indices, class_name, seed + 1000 * class_id
            )
            for cid in self.CLIENT_IDS:
                allocations[cid].extend(local_allocations[cid])
        sizes = {cid: len(allocations[cid]) for cid in self.CLIENT_IDS}
        if min_client_samples > 0 and min(sizes.values()) < min_client_samples:
            raise RuntimeError(
                f"Planned-class partition produced a client with too few samples. Sizes={sizes}. Adjust assignment weights or minimum size."
            )
        parts = {}
        for cid in self.CLIENT_IDS:
            part = (
                df.loc[allocations[cid]]
                .copy()
                .sample(frac=1.0, random_state=seed + int(cid[1:]))
                .reset_index(drop=True)
            )
            part["client_id"] = cid
            part["context"] = self.CLIENT_TO_CONTEXT[cid]
            part["is_planned_class"] = part["label"].isin(self.CLIENT_PLANNED_CLASSES[cid])
            parts[cid] = part
        self.log(
            "\nPrivate federated train sizes:", {cid: len(parts[cid]) for cid in self.CLIENT_IDS}
        )
        return parts
