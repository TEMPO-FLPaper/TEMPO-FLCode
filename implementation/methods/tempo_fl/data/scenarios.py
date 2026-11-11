"""Reference-derived data/scenarios. See README.md."""

import math, hashlib

from pathlib import Path

from collections import defaultdict

import numpy as np

import pandas as pd

from PIL import Image, ImageEnhance, ImageFilter

from sklearn.model_selection import train_test_split

class Operations:
    # Reference cells (zero-based): {'tempo': 28}
    def predict_routing_diagnostics(self, cid, df, state):
        context_name = self.CLIENT_TO_CONTEXT[cid]
        logits1, p1 = self.predict_logits_and_probabilities(
            state["l1_base_weights"][cid], df, state["l1_temperature"][cid]
        )
        logits2, p2 = self.predict_logits_and_probabilities(
            state["l2_base_weights"][context_name], df, state["l2_temperature"][context_name]
        )
        logits3, p3 = self.predict_logits_and_probabilities(
            state["l3_base_weights"], df, state["l3_temperature"]
        )
        pred1 = p1.argmax(axis=1)
        pred2 = p2.argmax(axis=1)
        pred3 = p3.argmax(axis=1)
        conf1 = p1.max(axis=1)
        conf2 = p2.max(axis=1)
        conf3 = p3.max(axis=1)
        rel1 = self.estimate_correctness_probability(conf1, state["reliability_models"]["L1"][cid])
        rel2 = self.estimate_correctness_probability(
            conf2, state["reliability_models"]["L2"][context_name]
        )
        rel3 = self.estimate_correctness_probability(conf3, state["reliability_models"]["L3"])
        return {
            "pred1": pred1,
            "pred2": pred2,
            "pred3": pred3,
            "conf1": conf1,
            "conf2": conf2,
            "conf3": conf3,
            "rel1": rel1,
            "rel2": rel2,
            "rel3": rel3,
            "gain2_audit": rel2 - rel1,
            "gain3_audit": rel3 - np.maximum(rel1, rel2),
            "l1_accept": conf1 >= state["l1_threshold"][cid],
            "l2_accept": conf2 >= state["l2_threshold"][context_name],
            "l3_accept": conf3 >= state["l3_threshold"],
        }

    # Reference cells (zero-based): {'tempo': 28}
    def source_identifier(self, source_path):
        return hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:14]

    # Reference cells (zero-based): {'tempo': 28}
    def apply_shift_transform(self, source_path, output_path, transform_name):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(source_path).convert("RGB")
        if transform_name == "brightness_down":
            image = ImageEnhance.Brightness(image).enhance(0.72)
        elif transform_name == "brightness_up":
            image = ImageEnhance.Brightness(image).enhance(1.28)
        elif transform_name == "contrast_down":
            image = ImageEnhance.Contrast(image).enhance(0.7)
        elif transform_name == "gaussian_blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
        elif transform_name == "zoom_crop":
            width, height = image.size
            margin_x = max(1, int(width * 0.08))
            margin_y = max(1, int(height * 0.08))
            image = image.crop((margin_x, margin_y, width - margin_x, height - margin_y)).resize(
                (width, height), Image.Resampling.BILINEAR
            )
        elif transform_name == "jpeg_degrade":
            image.save(output_path, format="JPEG", quality=45, optimize=True)
            return
        else:
            raise ValueError(f"Unknown shift transform: {transform_name}")
        image.save(output_path, format="JPEG", quality=90, optimize=True)

    # Reference cells (zero-based): {'tempo': 28}
    def ensure_shift_transform_bank(self, round_id):
        root = self.DIRS["config"] / "shift_transform_bank" / f"round_{round_id:03d}"
        manifest_path = root / "manifest.xlsx"
        if manifest_path.exists():
            return pd.read_excel(manifest_path)
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for _, row in self.progress_iter(
            self.online_df.iterrows(),
            total=len(self.online_df),
            desc=f"V12 transform bank R{round_id:03d}",
        ):
            source_path = str(row["path"])
            source_id = self.source_identifier(source_path)
            for transform_name in self.CONFIG["shift_transform_names"]:
                output_path = root / transform_name / str(row["label"]) / (source_id + ".jpg")
                if not output_path.exists():
                    self.apply_shift_transform(source_path, output_path, transform_name)
                rows.append(
                    {
                        "path": str(output_path),
                        "source_path": source_path,
                        "transform_name": transform_name,
                        "label": str(row["label"]),
                        "label_id": int(row["label_id"]),
                    }
                )
        manifest_df = pd.DataFrame(rows)
        manifest_df.to_excel(manifest_path, index=False)
        return manifest_df

    # Reference cells (zero-based): {'tempo': 28}
    def make_original_variant_df(self, source_df):
        result = source_df[["path", "label", "label_id"]].copy()
        result["source_path"] = result["path"].astype(str)
        result["transform_name"] = "original"
        return result

    # Reference cells (zero-based): {'tempo': 28}
    def strict_resolvable_candidates(self, cid, variant_df, state, round_id, role):
        if len(variant_df) == 0:
            return pd.DataFrame()
        diagnostics = self.predict_routing_diagnostics(cid, variant_df, state)
        y_true = variant_df["label_id"].to_numpy(dtype=np.int32)
        l1_rejected = ~diagnostics["l1_accept"]
        resolved_by_l2 = l1_rejected & diagnostics["l2_accept"] & (diagnostics["pred2"] == y_true)
        resolved_by_l3 = (
            l1_rejected
            & ~diagnostics["l2_accept"]
            & diagnostics["l3_accept"]
            & (diagnostics["pred3"] == y_true)
        )
        strict_mask = resolved_by_l2 | resolved_by_l3
        indices = np.where(strict_mask)[0]
        rows = []
        for idx in indices:
            if resolved_by_l2[idx]:
                teacher_level = 2
                higher_conf = float(diagnostics["conf2"][idx])
                higher_rel = float(diagnostics["rel2"][idx])
                expected_gain_audit = float(diagnostics["gain2_audit"][idx])
            else:
                teacher_level = 3
                higher_conf = float(diagnostics["conf3"][idx])
                higher_rel = float(diagnostics["rel3"][idx])
                expected_gain_audit = float(diagnostics["gain3_audit"][idx])
            l1_hardness = float(max(0.0, state["l1_threshold"][cid] - diagnostics["conf1"][idx]))
            rows.append(
                {
                    "round_discovered": int(round_id),
                    "client_id": cid,
                    "context": self.CLIENT_TO_CONTEXT[cid],
                    "pool_role": role,
                    "path": str(variant_df.iloc[idx]["path"]),
                    "source_path": str(variant_df.iloc[idx]["source_path"]),
                    "transform_name": str(variant_df.iloc[idx]["transform_name"]),
                    "label": str(variant_df.iloc[idx]["label"]),
                    "label_id": int(y_true[idx]),
                    "teacher_level": int(teacher_level),
                    "l1_confidence": float(diagnostics["conf1"][idx]),
                    "higher_confidence": higher_conf,
                    "l1_reliability_audit": float(diagnostics["rel1"][idx]),
                    "higher_reliability_audit": higher_rel,
                    "expected_gain_audit": expected_gain_audit,
                    "shift_score": float(
                        l1_hardness
                        + 0.35 * higher_conf
                        + 0.15 * higher_rel
                        + 0.1 * max(expected_gain_audit, -0.25)
                    ),
                }
            )
        return pd.DataFrame(rows)

    # Reference cells (zero-based): {'tempo': 28}
    def select_source_unique_candidates(self, candidate_df, target_count):
        if len(candidate_df) == 0:
            return candidate_df.copy()
        ranked = (
            candidate_df.sort_values(["shift_score", "higher_confidence"], ascending=[False, False])
            .drop_duplicates(subset=["source_path"], keep="first")
            .reset_index(drop=True)
        )
        target_count = min(int(target_count), len(ranked))
        if target_count <= 0:
            return ranked.iloc[[]].copy()
        max_per_class = max(
            1, int(math.ceil(target_count * self.CONFIG["shift_max_fraction_per_class"]))
        )
        chosen = []
        class_counts = defaultdict(int)
        for idx, row in ranked.iterrows():
            class_name = str(row["label"])
            if class_counts[class_name] >= max_per_class:
                continue
            chosen.append(idx)
            class_counts[class_name] += 1
            if len(chosen) >= target_count:
                break
        if len(chosen) < target_count:
            remaining = [idx for idx in ranked.index if idx not in set(chosen)]
            chosen.extend(remaining[: target_count - len(chosen)])
        return ranked.loc[chosen].copy().reset_index(drop=True)

    # Reference cells (zero-based): {'tempo': 28}
    def split_shift_sources(self, cid):
        source_df = (
            self.online_df[["path", "label", "label_id"]]
            .drop_duplicates(subset=["path"])
            .reset_index(drop=True)
        )
        try:
            stream_df, eval_df = train_test_split(
                source_df,
                test_size=self.CONFIG["shift_source_eval_fraction"],
                stratify=source_df["label_id"],
                random_state=self.CONFIG["seed"] + 6000 + int(cid[1:]),
            )
        except ValueError:
            stream_df, eval_df = train_test_split(
                source_df,
                test_size=self.CONFIG["shift_source_eval_fraction"],
                random_state=self.CONFIG["seed"] + 6000 + int(cid[1:]),
            )
        return (stream_df.reset_index(drop=True), eval_df.reset_index(drop=True))

    # Reference cells (zero-based): {'tempo': 28}
    def collect_role_candidates(
        self, cid, source_df, role, state, round_id, target_count, transform_bank_cache
    ):
        original_df = self.make_original_variant_df(source_df)
        candidates = self.strict_resolvable_candidates(cid, original_df, state, round_id, role)
        unique_originals = candidates["source_path"].nunique() if len(candidates) > 0 else 0
        if unique_originals < target_count:
            if transform_bank_cache["df"] is None:
                transform_bank_cache["df"] = self.ensure_shift_transform_bank(round_id)
            source_paths = set(source_df["path"].astype(str))
            transformed_df = (
                transform_bank_cache["df"][
                    transform_bank_cache["df"]["source_path"].astype(str).isin(source_paths)
                ]
                .copy()
                .reset_index(drop=True)
            )
            transformed_candidates = self.strict_resolvable_candidates(
                cid, transformed_df, state, round_id, role
            )
            candidates = pd.concat([candidates, transformed_candidates], ignore_index=True)
        selected = self.select_source_unique_candidates(candidates, target_count)
        return (candidates, selected)

    # Reference cells (zero-based): {'tempo': 28}
    def build_shift_profiles(self, state, round_id):
        self.log("\n" + "=" * 118)
        self.log("V12 ELIGIBILITY-BASED HARD-AND-RESOLVABLE SHIFT DISCOVERY")
        self.log("=" * 118)
        target_total = int(self.CONFIG["shift_target_pool_size"])
        target_eval = max(
            self.CONFIG["shift_min_eval_pool_size"],
            int(round(target_total * self.CONFIG["shift_source_eval_fraction"])),
        )
        target_stream = max(self.CONFIG["shift_min_stream_pool_size"], target_total - target_eval)
        transform_bank_cache = {"df": None}
        pool_rows = []
        summary_rows = []
        for cid in self.CLIENT_IDS:
            context_name = self.CLIENT_TO_CONTEXT[cid]
            stream_sources, eval_sources = self.split_shift_sources(cid)
            stream_candidates, stream_selected = self.collect_role_candidates(
                cid,
                stream_sources,
                "operational_adaptation",
                state,
                round_id,
                target_stream,
                transform_bank_cache,
            )
            eval_candidates, eval_selected = self.collect_role_candidates(
                cid,
                eval_sources,
                "scenario_evaluation_holdout",
                state,
                round_id,
                target_eval,
                transform_bank_cache,
            )
            eligible = (
                len(stream_selected) >= self.CONFIG["shift_min_stream_pool_size"]
                and len(eval_selected) >= self.CONFIG["shift_min_eval_pool_size"]
            )
            baseline = float(state["baseline_reliance"][cid])
            if np.isnan(baseline):
                baseline = 0.0
            required_reliance = self.required_promotion_reliance(baseline)
            minimum_fraction = self.required_hard_fraction_for_reliance(baseline, required_reliance)
            persistent_fraction = float(
                np.clip(
                    max(self.CONFIG["persistent_shift_hard_fraction"], minimum_fraction + 0.05),
                    0.0,
                    0.95,
                )
            )
            if eligible:
                stream_paths = stream_selected["path"].astype(str).tolist()
                eval_paths = eval_selected["path"].astype(str).tolist()
                classes = sorted(
                    set(stream_selected["label"].astype(str))
                    | set(eval_selected["label"].astype(str))
                )
                reason = "strict_original_or_transformed_samples_available"
            else:
                stream_paths = []
                eval_paths = []
                classes = []
                persistent_fraction = 0.0
                reason = "control_no_sufficient_strict_hierarchy_advantage"
            state["shift_profiles"][cid] = {
                "eligible": bool(eligible),
                "eligibility_reason": reason,
                "round_discovered": int(round_id),
                "stream_sample_paths": stream_paths,
                "evaluation_sample_paths": eval_paths,
                "classes": classes,
                "stream_pool_size": int(len(stream_paths)),
                "evaluation_pool_size": int(len(eval_paths)),
                "stream_candidate_count": int(len(stream_candidates)),
                "evaluation_candidate_count": int(len(eval_candidates)),
                "persistent_hard_fraction": persistent_fraction,
            }
            for selected_df in [stream_selected, eval_selected]:
                if len(selected_df) > 0:
                    selected_export = selected_df.copy()
                    selected_export["eligible_client"] = bool(eligible)
                    pool_rows.extend(selected_export.to_dict("records"))
            summary = {
                "round_discovered": int(round_id),
                "client_id": cid,
                "context": context_name,
                "eligible": bool(eligible),
                "eligibility_reason": reason,
                "baseline_reliance": baseline,
                "required_promotion_reliance": required_reliance,
                "stream_strict_candidates": int(len(stream_candidates)),
                "eval_strict_candidates": int(len(eval_candidates)),
                "stream_selected": int(len(stream_paths)),
                "eval_selected": int(len(eval_paths)),
                "persistent_hard_fraction": persistent_fraction,
                "selected_classes": ",".join(classes),
                "transformed_stream_selected": int(
                    np.sum(stream_selected["transform_name"] != "original")
                )
                if len(stream_selected) > 0
                else 0,
                "transformed_eval_selected": int(
                    np.sum(eval_selected["transform_name"] != "original")
                )
                if len(eval_selected) > 0
                else 0,
            }
            summary_rows.append(summary)
            self.log(
                f"{cid} | {context_name:11s} | Eligible={eligible} | stream={len(stream_paths):2d}/{target_stream:2d} | eval={len(eval_paths):2d}/{target_eval:2d} | classes={(classes if eligible else ['CONTROL'])}"
            )
        state["shift_profile_rows"] = list(pool_rows)
        state["shift_pool_history"].extend(pool_rows)
        state["shift_eligibility_history"].extend(summary_rows)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(
            self.DIRS["config"] / f"shift_eligibility_round_{round_id:03d}_v12_2.xlsx", index=False
        )
        if pool_rows:
            pd.DataFrame(pool_rows).to_excel(
                self.DIRS["config"] / f"strict_shift_pool_round_{round_id:03d}_v12_2.xlsx",
                index=False,
            )
        eligible_count = int(summary_df["eligible"].sum())
        self.log(
            f"\n[V12 Shift Eligibility] {eligible_count}/{len(self.CLIENT_IDS)} clients are shift eligible; the rest remain controls."
        )
        return summary_df

    # Reference cells (zero-based): {'tempo': 28}
    def ensure_shift_profiles(self, state):
        missing = [cid for cid in self.CLIENT_IDS if state["shift_profiles"].get(cid) is None]
        if missing:
            raise RuntimeError(
                "V12 shift eligibility profiles are missing for: " + ", ".join(missing)
            )

    # Reference cells (zero-based): {'tempo': 40}
    def hard_fraction_for_scenario(self, scenario, cid=None, state=None):
        if cid is None or state is None:
            return 0.0
        profile = state["shift_profiles"].get(cid)
        if not profile or not profile.get("eligible", False):
            return 0.0
        if scenario == "temporary_shift":
            return self.CONFIG["temporary_shift_hard_fraction"]
        if scenario == "persistent_shift":
            return float(
                profile.get(
                    "persistent_hard_fraction", self.CONFIG["persistent_shift_hard_fraction"]
                )
            )
        if scenario == "new_normal":
            return self.CONFIG["new_normal_hard_fraction"]
        return 0.0

    # Reference cells (zero-based): {'tempo': 40}
    def sample_rows_from_pool(self, pool, count, rng):
        if count <= 0:
            return pool.iloc[[]].copy()
        if len(pool) == 0:
            raise RuntimeError("Cannot sample from an empty pool.")
        indices = rng.choice(np.arange(len(pool)), size=int(count), replace=True)
        return pool.iloc[indices].copy().reset_index(drop=True)

    # Reference cells (zero-based): {'tempo': 40}
    def sample_scenario_batch(self, cid, round_id, sample_count, salt=0):
        context_name = self.CLIENT_TO_CONTEXT[cid]
        scenario = self.scenario_for_round(round_id)
        normal_pool = self.client_online_pool[cid]
        if len(normal_pool) == 0:
            raise RuntimeError(f"Online pool is empty for {cid}.")
        numeric_client = int(cid[1:])
        rng = np.random.default_rng(
            self.CONFIG["seed"] + 500000 + 1000 * round_id + 17 * numeric_client + int(salt)
        )
        hard_fraction = self.hard_fraction_for_scenario(scenario, cid=cid, state=self.STATE)
        if hard_fraction <= 0:
            sampled = self.sample_rows_from_pool(normal_pool, sample_count, rng)
            sampled["stream_source"] = "normal_client_pool"
        else:
            self.ensure_shift_profiles(self.STATE)
            profile = self.STATE["shift_profiles"][cid]
            is_evaluation = int(salt) == 9001
            pool_key = "evaluation_sample_paths" if is_evaluation else "stream_sample_paths"
            hard_paths = profile[pool_key]
            if not hard_paths:
                raise RuntimeError(f"Eligible V12 shift pool is empty for {cid}, role={pool_key}.")
            source_lookup = {
                str(row["path"]): {"label": str(row["label"]), "label_id": int(row["label_id"])}
                for _, row in self.online_df.iterrows()
            }
            pool_rows = []
            shift_pool_table = pd.DataFrame(self.STATE["shift_profile_rows"])
            for shift_path in hard_paths:
                match = shift_pool_table[shift_pool_table["path"].astype(str) == str(shift_path)]
                if len(match) == 0:
                    raise RuntimeError(f"Missing V12 shift metadata for {shift_path}")
                row = match.iloc[0]
                pool_rows.append(
                    {
                        "path": str(shift_path),
                        "label": str(row["label"]),
                        "label_id": int(row["label_id"]),
                    }
                )
            hard_pool = pd.DataFrame(pool_rows)
            hard_count = int(round(sample_count * hard_fraction))
            hard_count = min(max(hard_count, 1), sample_count)
            normal_count = sample_count - hard_count
            hard_samples = self.sample_rows_from_pool(hard_pool, hard_count, rng)
            hard_samples["stream_source"] = (
                "eligible_shift_eval_holdout"
                if is_evaluation
                else "eligible_shift_operational_adaptation"
            )
            normal_samples = self.sample_rows_from_pool(normal_pool, normal_count, rng)
            normal_samples["stream_source"] = "normal_client_pool"
            sampled = pd.concat([hard_samples, normal_samples], ignore_index=True)
            sampled = sampled.sample(
                frac=1.0, random_state=self.CONFIG["seed"] + round_id + numeric_client + int(salt)
            ).reset_index(drop=True)
        sampled["round"] = round_id
        sampled["scenario"] = scenario
        sampled["client_id"] = cid
        sampled["context"] = context_name
        return sampled

    # Reference cells (zero-based): {'tempo': 40}
    def sample_operational_events(self, cid, round_id):
        return self.sample_scenario_batch(
            cid, round_id, self.CONFIG["events_per_client_per_round"], salt=0
        )

    # Reference cells (zero-based): {'tempo': 40}
    def sample_operational_evaluation(self, cid, round_id):
        return self.sample_scenario_batch(
            cid, round_id, self.CONFIG["evaluation_samples_per_client"], salt=9001
        )
