from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .aggregators import (
    FoolsGoldState,
    clipped_mean,
    fltrust_aggregate,
    krum_select,
    multi_krum,
    resolve_defense_params,
    trimmed_mean,
)
from .data_loader import DataRepository, DeviceSplits
from .evaluation import attach_confusion_matrix, attribution_metrics, compute_metrics
from .model import SoftmaxRegressor
from .security import (
    AuditChain,
    ClientSecurityRecord,
    TrustState,
    clip_delta,
    compute_trust_score,
    coordinate_wise_median,
    reference_statistics,
    robust_norm_zscores,
    zero_trust_admission,
)


TRUST_SCENARIOS = {
    "S3": {
        "reference": "mean",
        "use_self_reported_loss": True,
        "use_memory": False,
    },
    "S4": {
        "reference": "median",
        "use_self_reported_loss": False,
        "use_memory": True,
    },
    "S4_minus_CT1": {
        "reference": "mean",
        "use_self_reported_loss": False,
        "use_memory": True,
    },
    "S4_minus_verif": {
        "reference": "median",
        "use_self_reported_loss": True,
        "use_memory": True,
    },
    "S4_minus_CT3": {
        "reference": "median",
        "use_self_reported_loss": False,
        "use_memory": False,
    },
}

# Published robust-aggregation baselines used in the paper comparison.
BASELINE_SCENARIOS = {
    "B_krum": "krum",
    "B_mkrum": "multikrum",
    "B_tmean": "trimmed_mean",
    "B_cmean": "clipped_mean",
    "B_fltrust": "fltrust",
    "B_fgold": "foolsgold",
}

FEDERATED_MODES = {
    "fedavg", "median", "trust",
    "krum", "multikrum", "trimmed_mean", "clipped_mean", "fltrust", "foolsgold",
}


class ExperimentRunner:
    def __init__(self, config: dict, repo: DataRepository, seed: int | None = None) -> None:
        self.config = config
        self.repo = repo
        self.model_cfg = config["model"]
        self.seed = int(config["random_seed"] if seed is None else seed)

    def run_all(self) -> dict:
        """Run the clean S1 modes for each enabled task."""
        results = {"tasks": {}, "summary": {}}
        for task_name, task_cfg in self.config["tasks"].items():
            if not task_cfg["enabled"]:
                continue
            task_output = self.run_task(task_name)
            results["tasks"][task_name] = task_output
            results["summary"][task_name] = task_output["comparison_table"].to_dict(orient="records")
        return results

    def run_task(self, task_name: str) -> dict:
        splits = self.repo.build_device_splits(task_name, seed=self.seed)
        mean, std = self.repo.pooled_scaler(splits)
        class_names = self.repo.class_names(task_name)
        n_classes = len(class_names)
        clean_attack = {**self.config.get("attack", {}), "enabled": False, "malicious_clients": []}

        centralized = self.run_centralized(task_name, splits, mean, std, n_classes)
        local_rows = self.run_local_only(task_name, splits, mean, std, n_classes)
        cross_rows = self.run_cross_device(task_name, splits, mean, std, n_classes)
        fedavg_result = self.run_federated(
            task_name, splits, mean, std, n_classes, mode="fedavg",
            attack_cfg=clean_attack, scenario_name="S1_fedavg",
        )
        trust_result = self.run_federated(
            task_name, splits, mean, std, n_classes, mode="trust",
            attack_cfg=clean_attack, security_cfg=self.scenario_security_config("S4"),
            scenario_name="S1_trust",
        )

        comparison_rows = [
            {"method": "centralized", **centralized["metrics"]},
            {"method": "fedavg", **fedavg_result["metrics"]},
            {"method": "trust", **trust_result["metrics"]},
        ]
        comparison_rows.extend({"method": row["scenario"], **row["metrics"]} for row in local_rows)
        comparison_rows.extend({"method": row["scenario"], **row["metrics"]} for row in cross_rows)

        return {
            "class_names": class_names,
            "centralized": centralized,
            "local_rows": local_rows,
            "cross_device_rows": cross_rows,
            "fedavg": fedavg_result,
            "trust_fed": trust_result,
            "federated_rounds": fedavg_result["rounds"] + trust_result["rounds"],
            "comparison_table": pd.DataFrame(comparison_rows),
        }

    def _task_data(
        self,
        frame: pd.DataFrame,
        task_name: str,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.repo.transform_features(frame, mean, std), self.repo.encode_labels(frame, task_name)

    def _evaluate_model(
        self,
        model: SoftmaxRegressor,
        frame: pd.DataFrame,
        task_name: str,
        mean: np.ndarray,
        std: np.ndarray,
        n_classes: int,
    ) -> dict:
        x, y = self._task_data(frame, task_name, mean, std)
        predictions = model.predict(x)
        return {
            "metrics": compute_metrics(y, predictions, task_name),
            "predictions": predictions,
            "truth": y,
            "confusion_matrix": attach_confusion_matrix(y, predictions, n_classes),
        }

    def _save_model(self, model: SoftmaxRegressor, name: str) -> None:
        if not self.config["export"].get("write_models", True):
            return
        model_dir = Path(self.config["paths"]["models_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        np.savez(model_dir / f"{safe_name}.npz", weights=model.weights, bias=model.bias)

    def run_centralized(
        self,
        task_name: str,
        splits: Dict[str, DeviceSplits],
        mean: np.ndarray,
        std: np.ndarray,
        n_classes: int,
    ) -> dict:
        train_df = pd.concat([splits[device].train for device in self.repo.devices], ignore_index=True)
        test_df = pd.concat([splits[device].test for device in self.repo.devices], ignore_index=True)
        x_train, y_train = self._task_data(train_df, task_name, mean, std)
        model = SoftmaxRegressor(len(self.repo.feature_columns), n_classes, seed=self.seed)
        model.fit(
            x_train, y_train,
            epochs=self.model_cfg["centralized_epochs"],
            learning_rate=self.model_cfg["learning_rate"],
            batch_size=self.model_cfg["batch_size"],
            l2=self.model_cfg["l2"],
            seed=self.seed,
        )
        result = self._evaluate_model(model, test_df, task_name, mean, std, n_classes)
        self._save_model(model, f"S1_centralized_{task_name}_seed{self.seed}")
        return result

    def run_local_only(
        self,
        task_name: str,
        splits: Dict[str, DeviceSplits],
        mean: np.ndarray,
        std: np.ndarray,
        n_classes: int,
    ) -> List[dict]:
        rows = []
        for client_index, device in enumerate(self.repo.devices):
            device_splits = splits[device]
            x_train, y_train = self._task_data(device_splits.train, task_name, mean, std)
            model = SoftmaxRegressor(len(self.repo.feature_columns), n_classes, seed=self.seed)
            model.fit(
                x_train, y_train,
                epochs=self.model_cfg["local_epochs"],
                learning_rate=self.model_cfg["learning_rate"],
                batch_size=self.model_cfg["batch_size"],
                l2=self.model_cfg["l2"],
                seed=self.seed + client_index,
            )
            evaluation = self._evaluate_model(model, device_splits.test, task_name, mean, std, n_classes)
            rows.append({"scenario": f"local_only_{device}", "device": device, **evaluation})
        return rows

    def run_cross_device(
        self,
        task_name: str,
        splits: Dict[str, DeviceSplits],
        mean: np.ndarray,
        std: np.ndarray,
        n_classes: int,
    ) -> List[dict]:
        rows = []
        for client_index, source in enumerate(self.repo.devices):
            x_train, y_train = self._task_data(splits[source].train, task_name, mean, std)
            model = SoftmaxRegressor(len(self.repo.feature_columns), n_classes, seed=self.seed)
            model.fit(
                x_train, y_train,
                epochs=self.model_cfg["local_epochs"],
                learning_rate=self.model_cfg["learning_rate"],
                batch_size=self.model_cfg["batch_size"],
                l2=self.model_cfg["l2"],
                seed=self.seed + client_index,
            )
            for target in self.repo.devices:
                if target == source:
                    continue
                evaluation = self._evaluate_model(model, splits[target].test, task_name, mean, std, n_classes)
                rows.append(
                    {
                        "scenario": f"cross_device_{source}_to_{target}",
                        "source_device": source,
                        "target_device": target,
                        **evaluation,
                    }
                )
        return rows

    @staticmethod
    def _apply_flattened_delta(model: SoftmaxRegressor, delta: np.ndarray) -> None:
        weight_size = model.weights.size
        expected_size = weight_size + model.bias.size
        if delta.size != expected_size:
            raise ValueError(f"Delta has {delta.size} coordinates; expected {expected_size}")
        model.weights += delta[:weight_size].reshape(model.weights.shape)
        model.bias += delta[weight_size:]

    def run_federated(
        self,
        task_name: str,
        splits: Dict[str, DeviceSplits],
        mean: np.ndarray,
        std: np.ndarray,
        n_classes: int,
        mode: str,
        security_cfg: dict | None = None,
        attack_cfg: dict | None = None,
        scenario_name: str | None = None,
    ) -> dict:
        """Run one FL experiment using trust aggregation or a published baseline."""
        if mode == "trust_fed":
            mode = "trust"
        if mode not in FEDERATED_MODES:
            raise ValueError(f"aggregation mode must be one of: {sorted(FEDERATED_MODES)}")

        security = copy.deepcopy(security_cfg or self.config["security"])
        attack = copy.deepcopy(attack_cfg or self.config.get("attack", {}))
        malicious_devices = set(self.repo.resolve_malicious_clients(attack)) if attack.get("enabled", False) else set()
        scenario = scenario_name or mode
        print(
            f"[FL] scenario={scenario} seed={self.seed} mode={mode} "
            f"rho={attack.get('rho') if attack.get('enabled', False) else '-'} "
            f"malicious={len(malicious_devices)}"
        )

        global_model = SoftmaxRegressor(len(self.repo.feature_columns), n_classes, seed=self.seed)
        if task_name == "binary" and len(self.repo.feature_columns) == 115:
            delta_size = global_model.weights.size + global_model.bias.size
            if delta_size != 232:
                raise AssertionError(f"Binary softmax update must be 232-dimensional, got {delta_size}")

        audit_dir = Path(self.config["paths"]["audit_dir"])
        audit_dir.mkdir(parents=True, exist_ok=True)
        safe_scenario = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario)
        # Attack parameters belong in the filename: the same scenario runs at
        # several rho values in one process (G2) and each chain must survive.
        attack_suffix = (
            f"_rho{float(attack.get('rho', 0.0)):g}_m{len(malicious_devices)}"
            if attack.get("enabled", False) else ""
        )
        audit_path = audit_dir / f"audit_{safe_scenario}_{task_name}{attack_suffix}_seed{self.seed}.jsonl"
        audit_chain = AuditChain(audit_path)
        trust_state = TrustState(self.repo.devices, security)

        # Build the poisoned label vector once so each round sees the same local dataset.
        client_data: dict[str, dict] = {}
        for client_index, device in enumerate(self.repo.devices):
            device_splits = splits[device]
            x_train = self.repo.transform_features(device_splits.train, mean, std)
            clean_y_train = self.repo.encode_labels(device_splits.train, task_name)
            attack_seed = self.seed * 1_000_003 + client_index * 1_009 + 17
            y_train = self.repo.encode_training_labels(
                device_splits.train, task_name, device, attack, attack_seed,
            )
            x_val, y_val = self._task_data(device_splits.val, task_name, mean, std)
            client_data[device] = {
                "x_train": x_train,
                "y_train": y_train,
                "x_val": x_val,
                "y_val": y_val,
                "flipped_train_labels": int(np.sum(clean_y_train != y_train)),
            }

        test_frame = pd.concat([splits[device].test for device in self.repo.devices], ignore_index=True)
        defense_params = resolve_defense_params(
            len(self.repo.devices), len(malicious_devices), self.config.get("defense"),
        )
        foolsgold_state = (
            FoolsGoldState(self.repo.devices, global_model.weights.size + global_model.bias.size)
            if mode == "foolsgold" else None
        )
        root_x = root_y = None
        if mode == "fltrust":
            # FLTrust assumes a small clean server-side root dataset; sample it
            # stratified from the pooled clean training labels, never poisoned.
            train_frames = pd.concat(
                [splits[device].train for device in self.repo.devices], ignore_index=True
            )
            y_all = self.repo.encode_labels(train_frames, task_name)
            root_rng = np.random.default_rng(self.seed + 977)
            root_size = defense_params["fltrust_root_size"]
            chosen: list[int] = []
            for cls in np.unique(y_all):
                cls_idx = np.flatnonzero(y_all == cls)
                take = max(1, int(round(root_size * len(cls_idx) / len(y_all))))
                chosen.extend(root_rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False))
            root_frame = train_frames.iloc[sorted(chosen)]
            root_x = self.repo.transform_features(root_frame, mean, std)
            root_y = self.repo.encode_labels(root_frame, task_name)

        rounds: list[dict] = []
        detections_per_round: list[list[str]] = []
        rejections_per_round: list[list[str]] = []

        for round_index in range(1, self.model_cfg["federated_rounds"] + 1):
            global_state = global_model.get_state()
            client_updates: list[dict] = []
            for client_index, device in enumerate(self.repo.devices):
                arrays = client_data[device]
                local_model = global_model.copy()
                local_model.fit(
                    arrays["x_train"], arrays["y_train"],
                    epochs=self.model_cfg["federated_local_epochs"],
                    learning_rate=self.model_cfg["learning_rate"],
                    batch_size=self.model_cfg["batch_size"],
                    l2=self.model_cfg["l2"],
                    seed=self.seed * 100_003 + round_index * 1_003 + client_index,
                )
                delta = local_model.flattened_delta(global_state)
                client_updates.append(
                    {
                        "device": device,
                        "train_samples": len(arrays["x_train"]),
                        "val_loss": local_model.cross_entropy(arrays["x_val"], arrays["y_val"]),
                        "delta": delta,
                        "delta_norm": float(np.linalg.norm(delta)),
                    }
                )

            deltas = [item["delta"] for item in client_updates]
            reference_delta, reference_norm, norms = reference_statistics(
                deltas, security.get("reference", "median")
            )
            norm_zscores = robust_norm_zscores(norms, security)
            in_warmup = mode == "trust" and round_index <= int(security.get("warmup_rounds", 0))

            security_records: list[ClientSecurityRecord] = []
            raw_weights: list[float] = []
            flagged_clients: list[str] = []
            rejected_clients: list[str] = []
            detected_clients: list[str] = []
            trust_threshold = None

            if mode == "trust":
                # Pass 1: update every client's EMA/strike state so the round's
                # relative rejection threshold sees the full cohort.
                interim: list[dict] = []
                for item, norm_z in zip(client_updates, norm_zscores):
                    components = compute_trust_score(
                        item["delta"], reference_delta, float(norm_z), security,
                        val_loss=item["val_loss"],
                    )
                    trust_score, strike, flagged, violation_reasons = trust_state.update(
                        item["device"], components.s_k, components.cosine,
                        float(norm_z), in_warmup,
                    )
                    admitted, admission_reason = zero_trust_admission(
                        item["train_samples"], float(norm_z), security,
                    )
                    interim.append({
                        "components": components,
                        "trust_score": trust_score,
                        "strike": strike,
                        "flagged": flagged,
                        "violation_reasons": violation_reasons,
                        "admitted": admitted,
                        "admission_reason": admission_reason,
                    })
                trust_threshold = trust_state.round_trust_threshold(
                    item["device"]
                    for item, state in zip(client_updates, interim)
                    if state["admitted"] and not state["flagged"]
                )

                # Pass 2: usability, weights, and forensic events.  Detection
                # (for attribution) is the persistent strike flag only; weight
                # suppression is tracked separately as the cost of defense.
                for item, state in zip(client_updates, interim):
                    components = state["components"]
                    trust_score = state["trust_score"]
                    flagged = state["flagged"]
                    above_threshold = in_warmup or trust_score >= trust_threshold
                    usable = state["admitted"] and not flagged and above_threshold
                    reasons: list[str] = []
                    if not state["admitted"]:
                        reasons.append(state["admission_reason"])
                    if flagged:
                        reasons.append("strike_window_threshold")
                    if not above_threshold:
                        reasons.append("trust_below_relative_tau")
                    rejection_reason = ";".join(reasons) if reasons else "accepted"
                    raw_weight = (
                        item["train_samples"] * (float(security.get("eps_w", 0.1)) + trust_score)
                        if usable else 0.0
                    )
                    if flagged:
                        flagged_clients.append(item["device"])
                        detected_clients.append(item["device"])
                        audit_chain.append_flag_event(
                            round_index, item["device"], item["delta"],
                            state["violation_reasons"] or ["strike_window_threshold"],
                            scenario=scenario,
                            seed=self.seed,
                            strike=state["strike"],
                            trust_score=trust_score,
                        )
                    if not usable:
                        rejected_clients.append(item["device"])
                    raw_weights.append(float(raw_weight))
                    security_records.append(
                        ClientSecurityRecord(
                            device=item["device"],
                            train_samples=int(item["train_samples"]),
                            val_loss=float(item["val_loss"]),
                            delta_norm=float(item["delta_norm"]),
                            reference_norm=float(reference_norm),
                            cosine_to_reference=float(components.cosine),
                            a_k=float(components.a_k),
                            m_k=float(components.m_k),
                            s_k=float(components.s_k),
                            trust_score=float(trust_score),
                            strike=int(state["strike"]),
                            flagged=bool(flagged),
                            admitted=bool(state["admitted"]),
                            rejection_reason=rejection_reason,
                            detected=bool(flagged),
                            is_malicious_ground_truth=item["device"] in malicious_devices,
                            aggregation_weight=0.0,
                        )
                    )
            else:
                for item, norm_z in zip(client_updates, norm_zscores):
                    components = compute_trust_score(
                        item["delta"], reference_delta, float(norm_z), security,
                        val_loss=item["val_loss"],
                    )
                    raw_weights.append(1.0 if mode == "fedavg" else 0.0)
                    security_records.append(
                        ClientSecurityRecord(
                            device=item["device"],
                            train_samples=int(item["train_samples"]),
                            val_loss=float(item["val_loss"]),
                            delta_norm=float(item["delta_norm"]),
                            reference_norm=float(reference_norm),
                            cosine_to_reference=float(components.cosine),
                            a_k=float(components.a_k),
                            m_k=float(components.m_k),
                            s_k=float(components.s_k),
                            trust_score=float(components.s_k),
                            strike=0,
                            flagged=False,
                            admitted=True,
                            rejection_reason="not_applicable",
                            detected=False,
                            is_malicious_ground_truth=item["device"] in malicious_devices,
                            aggregation_weight=0.0,
                        )
                    )

            if mode == "fedavg":
                aggregate_delta = np.mean(np.stack(deltas), axis=0)
                for record in security_records:
                    record.aggregation_weight = 1.0 / len(security_records)
            elif mode == "median":
                aggregate_delta = coordinate_wise_median(deltas)
            elif mode == "krum":
                selected = krum_select(deltas, defense_params["byzantine_f"], m=1)
                aggregate_delta = deltas[selected[0]]
                security_records[selected[0]].aggregation_weight = 1.0
            elif mode == "multikrum":
                aggregate_delta, selected = multi_krum(
                    deltas, defense_params["byzantine_f"], defense_params["multikrum_m"],
                )
                for index in selected:
                    security_records[index].aggregation_weight = 1.0 / len(selected)
            elif mode == "trimmed_mean":
                aggregate_delta = trimmed_mean(deltas, defense_params["trim_k"])
            elif mode == "clipped_mean":
                clip_norm = defense_params["clip_to_median_multiplier"] * float(np.median(norms))
                aggregate_delta = clipped_mean(deltas, max(clip_norm, 1e-12))
                for record in security_records:
                    record.aggregation_weight = 1.0 / len(security_records)
            elif mode == "fltrust":
                server_model = global_model.copy()
                server_model.fit(
                    root_x, root_y,
                    epochs=self.model_cfg["federated_local_epochs"],
                    learning_rate=self.model_cfg["learning_rate"],
                    batch_size=self.model_cfg["batch_size"],
                    l2=self.model_cfg["l2"],
                    seed=self.seed * 100_003 + round_index * 1_003 + 999,
                )
                server_delta = server_model.flattened_delta(global_state)
                aggregate_delta, fltrust_scores = fltrust_aggregate(deltas, server_delta)
                total_trust = float(fltrust_scores.sum())
                for record, score in zip(security_records, fltrust_scores):
                    record.trust_score = float(score)
                    record.aggregation_weight = float(score / total_trust) if total_trust > 0 else 0.0
            elif mode == "foolsgold":
                weight_by_device = foolsgold_state.weights(
                    {item["device"]: item["delta"] for item in client_updates}
                )
                weight_array = np.asarray(
                    [weight_by_device[item["device"]] for item in client_updates], dtype=np.float64
                )
                total_weight = float(weight_array.sum())
                if total_weight <= 0.0:
                    aggregate_delta = np.zeros_like(deltas[0])
                else:
                    normalized_weights = weight_array / total_weight
                    aggregate_delta = sum(
                        weight * delta for weight, delta in zip(normalized_weights, deltas)
                    )
                    for record, weight in zip(security_records, normalized_weights):
                        record.aggregation_weight = float(weight)
                        record.trust_score = float(weight_by_device[record.device])
            else:  # trust
                weight_array = np.asarray(raw_weights, dtype=np.float64)
                if weight_array.sum() <= 0.0:
                    aggregate_delta = np.zeros_like(deltas[0])
                else:
                    normalized_weights = weight_array / weight_array.sum()
                    aggregate_delta = sum(
                        weight * clip_delta(item["delta"], reference_norm, security)
                        for weight, item in zip(normalized_weights, client_updates)
                    )
                    for record, weight in zip(security_records, normalized_weights):
                        record.aggregation_weight = float(weight)

            # A zero delta is the specified fallback: preserve theta from the prior round.
            self._apply_flattened_delta(global_model, np.asarray(aggregate_delta))
            evaluation = self._evaluate_model(global_model, test_frame, task_name, mean, std, n_classes)
            round_payload = {
                "event": "round_summary",
                "round": round_index,
                "task": task_name,
                "scenario": scenario,
                "mode": mode,
                "seed": self.seed,
                "rho": float(attack.get("rho", 0.0)) if attack.get("enabled", False) else None,
                "metrics": evaluation["metrics"],
                "reference_norm": float(reference_norm),
                "norms": norms.tolist(),
                "norm_zscores": [float(z) for z in norm_zscores],
                "trust_threshold": None if trust_threshold is None else float(trust_threshold),
                "in_warmup": bool(in_warmup),
                # Per-client weights are undefined for coordinate-wise aggregates.
                "malicious_weight_share": (
                    None if mode in {"median", "trimmed_mean"} else float(
                        sum(r.aggregation_weight for r in security_records if r.is_malicious_ground_truth)
                    )
                ),
                "flagged_clients": flagged_clients,
                "rejected_clients": rejected_clients,
                "detected_clients": detected_clients,
                "clients": [asdict(record) for record in security_records],
            }
            audit_chain.append(round_payload)
            rounds.append({**round_payload, "confusion_matrix": evaluation["confusion_matrix"].tolist()})
            detections_per_round.append(detected_clients)
            rejections_per_round.append(rejected_clients)
            if round_index == 1 or round_index % 10 == 0 or round_index == self.model_cfg["federated_rounds"]:
                print(
                    f"[FL] {scenario} seed={self.seed} round={round_index}/"
                    f"{self.model_cfg['federated_rounds']} accuracy={evaluation['metrics']['accuracy']:.4f} "
                    f"f1_macro={evaluation['metrics'].get('f1_macro', evaluation['metrics']['f1']):.4f} "
                    f"detected={len(detected_clients)}"
                )

        final_metrics = rounds[-1]["metrics"]
        final_confusion = np.asarray(rounds[-1]["confusion_matrix"], dtype=int)
        attribution = None
        if mode == "trust":
            attribution = attribution_metrics(
                detections_per_round,
                malicious_devices,
                all_clients=self.repo.devices,
                audit_entries=audit_chain.records(),
            )
            # Cost-of-defense diagnostics: honest clients whose weight was ever
            # suppressed after warm-up (distinct from detection/attribution).
            warmup = int(security.get("warmup_rounds", 0))
            honest = set(self.repo.devices) - malicious_devices
            suppressed_honest = {
                device
                for round_number, rejected in enumerate(rejections_per_round, start=1)
                if round_number > warmup
                for device in rejected
                if device in honest
            }
            attribution["suppression_fpr"] = (
                float(len(suppressed_honest) / len(honest)) if honest else 0.0
            )
            post_warmup_rounds = [
                rejected
                for round_number, rejected in enumerate(rejections_per_round, start=1)
                if round_number > warmup
            ]
            attribution["mean_honest_suppressed_per_round"] = (
                float(
                    np.mean([
                        len([device for device in rejected if device in honest]) / len(honest)
                        for rejected in post_warmup_rounds
                    ])
                )
                if honest and post_warmup_rounds else 0.0
            )
            attribution["suppressed_honest_clients"] = sorted(suppressed_honest)
            attribution["final_malicious_weight_share"] = float(
                rounds[-1]["malicious_weight_share"]
            )
        self._save_model(global_model, f"{safe_scenario}_{task_name}_seed{self.seed}")
        return {
            "scenario": scenario,
            "mode": mode,
            "seed": self.seed,
            "rho": float(attack.get("rho", 0.0)) if attack.get("enabled", False) else None,
            "malicious_clients": sorted(malicious_devices),
            "metrics": final_metrics,
            "rounds": rounds,
            "confusion_matrix": final_confusion,
            "attribution": attribution,
            "audit_path": str(audit_path),
            "audit_chain_valid": audit_chain.verify(),
            "flipped_train_labels": {
                device: client_data[device]["flipped_train_labels"] for device in self.repo.devices
            },
        }

    def scenario_security_config(self, scenario: str) -> dict:
        security = copy.deepcopy(self.config["security"])
        if scenario in TRUST_SCENARIOS:
            security.update(TRUST_SCENARIOS[scenario])
        return security

    def run_federated_scenario(
        self,
        scenario: str,
        task_name: str,
        splits: Dict[str, DeviceSplits],
        mean: np.ndarray,
        std: np.ndarray,
        n_classes: int,
        attack_cfg: dict,
    ) -> dict:
        if scenario == "S1":
            clean_attack = {**attack_cfg, "enabled": False, "malicious_clients": []}
            return self.run_federated(
                task_name, splits, mean, std, n_classes, "fedavg",
                attack_cfg=clean_attack, scenario_name="S1",
            )
        if scenario == "S2":
            mode = "fedavg"
        elif scenario == "S2m":
            mode = "median"
        elif scenario in BASELINE_SCENARIOS:
            mode = BASELINE_SCENARIOS[scenario]
        elif scenario in TRUST_SCENARIOS:
            mode = "trust"
        else:
            raise ValueError(f"Unknown experiment scenario: {scenario}")
        return self.run_federated(
            task_name, splits, mean, std, n_classes, mode,
            attack_cfg=attack_cfg,
            security_cfg=self.scenario_security_config(scenario),
            scenario_name=scenario,
        )

    def run_experiment_grid(self, groups: list[str] | None = None) -> dict:
        """Run G0-G4 over configured seeds and return flat/export-ready results."""
        requested_groups = groups or list(self.config.get("experiments", {}).get("groups", ["G0"]))
        unknown = set(requested_groups) - {"G0", "G1", "G2", "G3", "G4"}
        if unknown:
            raise ValueError(f"Unknown experiment groups: {sorted(unknown)}")
        task_name = self.config.get("task", "binary")
        if task_name != "binary":
            raise ValueError("The poisoning experiment grid is defined for task='binary'")

        seeds = [int(seed) for seed in self.config.get("seeds", [self.seed])]
        malicious_by_count = self.config["experiments"]["malicious_clients_by_count"]
        rho_values = [float(rho) for rho in self.config["attack"]["rho_values"]]
        prepared: dict[int, tuple[dict, np.ndarray, np.ndarray, int]] = {}
        result_cache: dict[tuple, dict] = {}
        result_rows: list[dict] = []
        detailed_runs: dict[str, dict] = {}

        def prepare(seed: int) -> tuple[dict, np.ndarray, np.ndarray, int]:
            # Retain only the current seed's splits to bound memory use.
            if seed not in prepared:
                prepared.clear()
                splits = self.repo.build_device_splits(task_name, seed=seed)
                pooled_mean, pooled_std = self.repo.pooled_scaler(splits)
                prepared[seed] = (splits, pooled_mean, pooled_std, len(self.repo.class_names(task_name)))
            return prepared[seed]

        incremental_path = None
        logs_dir = self.config.get("paths", {}).get("logs_dir")
        if logs_dir:
            incremental_path = Path(logs_dir) / "rows_incremental.jsonl"
            incremental_path.parent.mkdir(parents=True, exist_ok=True)

        def persist_row(row: dict) -> None:
            result_rows.append(row)
            if incremental_path is not None:
                with incremental_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                        for key, value in row.items()
                    }) + "\n")

        def federated(group: str, scenario: str, seed: int, rho: float | None, count: int) -> dict:
            malicious = [] if count == 0 else list(malicious_by_count[str(count)])
            enabled = scenario != "S1" and count > 0
            attack = {
                **self.config["attack"],
                "enabled": enabled,
                "rho": 0.0 if rho is None else float(rho),
                "malicious_clients": malicious,
            }
            cache_key = (scenario, seed, attack["enabled"], attack["rho"], tuple(malicious))
            if cache_key not in result_cache:
                runner = ExperimentRunner(self.config, self.repo, seed=seed)
                splits, pooled_mean, pooled_std, n_classes = prepare(seed)
                result_cache[cache_key] = runner.run_federated_scenario(
                    scenario, task_name, splits, pooled_mean, pooled_std, n_classes, attack,
                )
            result = result_cache[cache_key]
            # Per-round payloads remain on disk in the audit chain.
            result.pop("rounds", None)
            run_key = f"{group}_{scenario}_rho{rho}_m{count}_seed{seed}"
            detailed_runs[run_key] = result
            attribution = result.get("attribution") or {}
            row = {
                "group": group,
                "scenario": scenario,
                "mode": result["mode"],
                "seed": seed,
                "rho": rho,
                "malicious_count": count,
                **result["metrics"],
                "attribution_precision": attribution.get("precision_cum"),
                "attribution_recall": attribution.get("recall_cum"),
                "mean_time_to_detection": attribution.get("mean_time_to_detection"),
                "false_positive_rate": attribution.get("false_positive_rate"),
                "suppression_fpr": attribution.get("suppression_fpr"),
                "mean_honest_suppressed_per_round": attribution.get("mean_honest_suppressed_per_round"),
                "final_malicious_weight_share": attribution.get("final_malicious_weight_share"),
                "ecobee_false_positive": attribution.get("ecobee_false_positive"),
                "audit_correctness": attribution.get("audit_correctness"),
                "audit_chain_valid": result["audit_chain_valid"],
            }
            persist_row(row)
            return result

        if "G0" in requested_groups:
            for seed in seeds:
                print(f"[GRID] G0 seed={seed}: centralized baseline")
                runner = ExperimentRunner(self.config, self.repo, seed=seed)
                splits, pooled_mean, pooled_std, n_classes = prepare(seed)
                centralized = runner.run_centralized(task_name, splits, pooled_mean, pooled_std, n_classes)
                persist_row({
                    "group": "G0", "scenario": "S1_centralized", "mode": "centralized",
                    "seed": seed, "rho": None, "malicious_count": 0, **centralized["metrics"],
                })
                print(f"[GRID] G0 seed={seed}: local-only baselines")
                local_rows = runner.run_local_only(task_name, splits, pooled_mean, pooled_std, n_classes)
                local_metrics = pd.DataFrame([row["metrics"] for row in local_rows]).mean(numeric_only=True).to_dict()
                persist_row({
                    "group": "G0", "scenario": "S1_local_only", "mode": "local_only",
                    "seed": seed, "rho": None, "malicious_count": 0, **local_metrics,
                })
                federated("G0", "S1", seed, None, 0)
                clean_attack = {**self.config["attack"], "enabled": False, "malicious_clients": []}
                trust_result = runner.run_federated(
                    task_name, splits, pooled_mean, pooled_std, n_classes, "trust",
                    attack_cfg=clean_attack,
                    security_cfg=runner.scenario_security_config("S4"),
                    scenario_name="S1_trust",
                )
                detailed_runs[f"G0_S1_trust_seed{seed}"] = trust_result
                attr = trust_result["attribution"] or {}
                persist_row({
                    "group": "G0", "scenario": "S1_trust", "mode": "trust",
                    "seed": seed, "rho": None, "malicious_count": 0,
                    **trust_result["metrics"],
                    "attribution_precision": attr.get("precision_cum"),
                    "attribution_recall": attr.get("recall_cum"),
                    "mean_time_to_detection": attr.get("mean_time_to_detection"),
                    "false_positive_rate": attr.get("false_positive_rate"),
                    "suppression_fpr": attr.get("suppression_fpr"),
                    "mean_honest_suppressed_per_round": attr.get("mean_honest_suppressed_per_round"),
                    "final_malicious_weight_share": attr.get("final_malicious_weight_share"),
                    "ecobee_false_positive": attr.get("ecobee_false_positive"),
                    "audit_correctness": attr.get("audit_correctness"),
                    "audit_chain_valid": trust_result["audit_chain_valid"],
                })

        if "G1" in requested_groups:
            for seed in seeds:
                federated("G1", "S1", seed, None, 0)
                for scenario in ("S2", "S2m", "S3", "S4", *BASELINE_SCENARIOS):
                    federated("G1", scenario, seed, 1.0, 2)

        if "G2" in requested_groups:
            for seed in seeds:
                federated("G2_baseline", "S1", seed, None, 0)
                for rho in rho_values:
                    for scenario in ("S2", "S4", "B_fltrust"):
                        federated("G2", scenario, seed, rho, 3)

        if "G3" in requested_groups:
            for seed in seeds:
                for count in (1, 2, 3):
                    for scenario in ("S2m", "S4", "B_mkrum"):
                        federated("G3", scenario, seed, 1.0, count)

        if "G4" in requested_groups:
            for seed in seeds:
                for scenario in ("S4", "S4_minus_CT1", "S4_minus_verif", "S4_minus_CT3", "S3"):
                    federated("G4", scenario, seed, 1.0, 3)

        runs_frame = pd.DataFrame(result_rows)
        summary_keys = ["group", "scenario", "mode", "rho", "malicious_count"]
        metric_columns = [
            column for column in (
                "accuracy", "precision", "recall", "f1", "f1_malicious", "f1_benign", "f1_macro",
                "attribution_precision", "attribution_recall", "mean_time_to_detection",
                "false_positive_rate", "suppression_fpr", "mean_honest_suppressed_per_round",
                "final_malicious_weight_share", "audit_correctness",
            ) if column in runs_frame.columns
        ]
        summary = runs_frame.groupby(summary_keys, dropna=False)[metric_columns].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        seeds_summary = summary.reset_index().fillna({column: 0.0 for column in summary.columns if column.endswith("_std")})

        rho_rows: list[dict] = []
        if "G2" in requested_groups:
            baseline = runs_frame[runs_frame["group"] == "G2_baseline"].set_index("seed")["accuracy"]
            g2 = runs_frame[runs_frame["group"] == "G2"]
            for rho in rho_values:
                row: dict = {"rho": rho}
                for scenario, prefix in (("S2", "s2"), ("S4", "s4"), ("B_fltrust", "fltrust")):
                    subset = g2[(g2["scenario"] == scenario) & (g2["rho"] == rho)]
                    if subset.empty:
                        continue
                    drops = [float(baseline.loc[item.seed] - item.accuracy) for item in subset.itertuples()]
                    row[f"{prefix}_accuracy_drop"] = float(np.mean(drops))
                    row[f"{prefix}_accuracy_drop_std"] = (
                        float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0
                    )
                s4 = g2[(g2["scenario"] == "S4") & (g2["rho"] == rho)]
                # Backward-compatible keys used by plot_rho_boundary and Excel.
                row["accuracy_drop"] = row.get("s2_accuracy_drop", 0.0)
                row["accuracy_drop_std"] = row.get("s2_accuracy_drop_std", 0.0)
                row["recall"] = float(s4["attribution_recall"].mean()) if not s4.empty else 0.0
                row["precision"] = float(s4["attribution_precision"].mean()) if not s4.empty else 0.0
                rho_rows.append(row)

        ablation = seeds_summary[seeds_summary["group"] == "G4"].reset_index(drop=True)
        return {
            "groups": requested_groups,
            "runs": runs_frame,
            "detailed_runs": detailed_runs,
            "seeds_summary": seeds_summary,
            "rho_sweep": pd.DataFrame(rho_rows),
            "ablation": ablation,
        }
