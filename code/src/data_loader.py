from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ECOBEE_DEVICE = "Ecobee_Thermostat"

KNOWN_DEVICES = (
    "Danmini_Doorbell",
    ECOBEE_DEVICE,
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
)

KNOWN_ATTACK_LABELS = (
    "gafgyt_scan",
    "gafgyt_tcp",
    "gafgyt_udp",
    "gafgyt_combo",
    "gafgyt_junk",
    "mirai_ack",
    "mirai_scan",
    "mirai_syn",
    "mirai_udpplain",
    "mirai_udp",
)

DEVICE_COLUMN_NAMES = ("device", "device_name", "client", "client_id")
LABEL_COLUMN_NAMES = ("label", "attack", "attack_type", "category", "class", "target")


def flip_labels(y_train: np.ndarray, rho: float, seed: int) -> np.ndarray:
    """Flip only binary training labels, using the requested Bernoulli attack."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"Label-flip rho must be in [0, 1], got {rho}")
    y = np.asarray(y_train, dtype=int)
    if y.ndim != 1 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Label flipping is defined only for one-dimensional binary labels")
    rng = np.random.default_rng(seed)
    mask = rng.random(len(y)) < rho
    y_out = y.copy()
    y_out[mask] = 1 - y_out[mask]
    return y_out


@dataclass
class DeviceSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class DataRepository:
    def __init__(self, config: dict, master_df: pd.DataFrame) -> None:
        self.config = config
        self.master_df = master_df
        configured_devices = list(config.get("devices", KNOWN_DEVICES))
        present_devices = set(master_df["device"].unique())
        missing_devices = [device for device in configured_devices if device not in present_devices]
        if missing_devices:
            raise ValueError(f"Configured devices missing from data: {missing_devices}")
        self.devices = [device for device in configured_devices if device in present_devices]

        metadata = {"device", "label", "binary_label", "source_file"}
        self.feature_columns = [
            column
            for column in master_df.columns
            if column not in metadata and pd.api.types.is_numeric_dtype(master_df[column])
        ]
        if not self.feature_columns:
            raise ValueError("No numeric feature columns were found")
        if master_df[self.feature_columns].isna().any().any():
            raise ValueError("Feature matrix contains missing values or inconsistent CSV columns")
        self.multiclass_encoder = LabelEncoder().fit(master_df["label"])
        self.resolve_malicious_clients(config.get("attack", {}))

    @classmethod
    def from_config(cls, config: dict) -> "DataRepository":
        raw_dir = Path(config["paths"]["raw_data_dir"])
        frames: List[pd.DataFrame] = []
        for csv_path in sorted(raw_dir.glob("*.csv")):
            frame = pd.read_csv(csv_path)
            lower_columns = {str(column).lower(): column for column in frame.columns}
            device_column = next((lower_columns[name] for name in DEVICE_COLUMN_NAMES if name in lower_columns), None)
            label_column = next((lower_columns[name] for name in LABEL_COLUMN_NAMES if name in lower_columns), None)

            file_device: str | None = None
            file_label: str | None = None
            if device_column is None or label_column is None:
                file_device, file_label = cls._parse_metadata(csv_path.name)

            if device_column is None:
                device_values = pd.Series(file_device, index=frame.index)
            else:
                device_values = frame[device_column].map(cls._canonical_device)

            if label_column is None:
                label_values = pd.Series(file_label, index=frame.index)
            else:
                label_values = frame[label_column].map(cls._canonical_label)

            # Exclude metadata from features and avoid duplicate metadata columns.
            drop_columns = [c for c in {device_column, label_column} if c is not None]
            if drop_columns:
                frame = frame.drop(columns=drop_columns)

            metadata = pd.DataFrame(
                {
                    "device": device_values.to_numpy(),
                    "label": label_values.to_numpy(),
                    "binary_label": np.where(label_values.to_numpy() == "benign", "benign", "malicious"),
                    "source_file": csv_path.name,
                }
            )
            frame = pd.concat([frame.reset_index(drop=True), metadata], axis=1, copy=False)
            frames.append(frame)

        if not frames:
            raise FileNotFoundError(f"No CSV files found in {raw_dir}")
        master_df = pd.concat(frames, ignore_index=True, sort=False)
        return cls(config=config, master_df=master_df)

    @staticmethod
    def _normalise_token(value: object) -> str:
        token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        return re.sub(r"_+", "_", token)

    @classmethod
    def _canonical_device(cls, value: object) -> str:
        token = cls._normalise_token(value)
        aliases = {
            "pt_737e_security_camera": "Provision_PT_737E_Security_Camera",
            "provision_737e_security_camera": "Provision_PT_737E_Security_Camera",
            "provision_838_security_camera": "Provision_PT_838_Security_Camera",
            "simplehome_xcs7_1002_wht": "SimpleHome_XCS7_1002_WHT_Security_Camera",
            "simplehome_xcs7_1003_wht": "SimpleHome_XCS7_1003_WHT_Security_Camera",
        }
        if token in aliases:
            return aliases[token]
        for device in KNOWN_DEVICES:
            if token == device.lower():
                return device
        raise ValueError(f"Unknown N-BaIoT device: {value}")

    @classmethod
    def _canonical_label(cls, value: object) -> str:
        token = cls._normalise_token(value).replace("gafgut", "gafgyt")
        if token in {"0", "normal", "benign", "benign_traffic"} or "benign" in token:
            return "benign"
        if token in {"1", "malicious", "attack"}:
            return "malicious"
        for label in sorted(KNOWN_ATTACK_LABELS, key=len, reverse=True):
            if label in token:
                return label
        raise ValueError(f"Unknown N-BaIoT label: {value}")

    @classmethod
    def _parse_metadata(cls, filename: str) -> tuple[str, str]:
        """Infer any of the nine devices and eleven labels from a CSV filename."""
        stem = Path(filename).stem
        match = re.match(r"^(.+?)_(\d+)sample_(.+)$", stem, flags=re.IGNORECASE)
        if match:
            device_token, label_token = match.group(1), match.group(3)
            return cls._canonical_device(device_token), cls._canonical_label(label_token)

        # Accept filenames without an explicit sample-count token.
        normalised = cls._normalise_token(stem).replace("gafgut", "gafgyt")
        device = next(
            (candidate for candidate in KNOWN_DEVICES if normalised.startswith(candidate.lower())),
            None,
        )
        if device is None and normalised.startswith("pt_737e_security_camera"):
            device = "Provision_PT_737E_Security_Camera"
        if device is None:
            raise ValueError(f"Cannot infer device from filename: {filename}")
        return device, cls._canonical_label(normalised)

    def resolve_malicious_clients(self, attack_cfg: dict | None = None) -> list[str]:
        """Resolve stable configured indices/names and enforce the Ecobee invariant."""
        cfg = attack_cfg or {}
        resolved: list[str] = []
        for client in cfg.get("malicious_clients", []):
            if isinstance(client, (int, np.integer)):
                index = int(client)
                if index < 0 or index >= len(self.devices):
                    raise ValueError(f"Malicious client index out of range: {index}")
                device = self.devices[index]
            else:
                device = self._canonical_device(client)
                if device not in self.devices:
                    raise ValueError(f"Malicious client is not configured: {device}")
            if device == ECOBEE_DEVICE:
                raise ValueError("Ecobee_Thermostat must always remain an honest client")
            if device not in resolved:
                resolved.append(device)
        return resolved

    def build_device_splits(self, task_name: str, seed: int | None = None) -> Dict[str, DeviceSplits]:
        """Build clean 70/15/15 device splits. Attack labels are applied later to train only."""
        split_cfg = self.config["split"]
        if not np.isclose(sum(split_cfg[key] for key in ("train", "val", "test")), 1.0):
            raise ValueError("Split fractions must sum to 1.0")
        test_size = split_cfg["val"] + split_cfg["test"]
        relative_test = split_cfg["test"] / test_size
        split_seed = self.config["random_seed"] if seed is None else int(seed)
        result: Dict[str, DeviceSplits] = {}

        for device in self.devices:
            device_df = self.master_df[self.master_df["device"] == device]
            y = self.encode_labels(device_df, task_name)
            train_idx, temp_idx = train_test_split(
                np.arange(len(device_df)),
                test_size=test_size,
                stratify=y,
                random_state=split_seed,
            )
            temp_y = y[temp_idx]
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=relative_test,
                stratify=temp_y,
                random_state=split_seed,
            )
            result[device] = DeviceSplits(
                train=device_df.iloc[train_idx].reset_index(drop=True),
                val=device_df.iloc[val_idx].reset_index(drop=True),
                test=device_df.iloc[test_idx].reset_index(drop=True),
            )
        return result

    def pooled_scaler(self, splits: Dict[str, DeviceSplits]) -> tuple[np.ndarray, np.ndarray]:
        counts, means, variances = [], [], []
        for device_splits in splits.values():
            values = device_splits.train[self.feature_columns].to_numpy(dtype=np.float64)
            counts.append(values.shape[0])
            means.append(values.mean(axis=0))
            variances.append(values.var(axis=0))
        counts_arr = np.array(counts, dtype=np.float64)
        total = counts_arr.sum()
        pooled_mean = sum(count * mean for count, mean in zip(counts_arr, means)) / total
        second_moment = sum(
            count * (variance + mean**2)
            for count, variance, mean in zip(counts_arr, variances, means)
        ) / total
        pooled_var = np.maximum(second_moment - pooled_mean**2, 1e-8)
        return pooled_mean, np.sqrt(pooled_var)

    def transform_features(self, frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        values = frame[self.feature_columns].to_numpy(dtype=np.float64)
        return (values - mean) / std

    def encode_labels(self, frame: pd.DataFrame, task_name: str) -> np.ndarray:
        if task_name == "binary":
            return (frame["label"] != "benign").astype(int).to_numpy(dtype=int)
        return self.multiclass_encoder.transform(frame["label"])

    def encode_training_labels(
        self,
        frame: pd.DataFrame,
        task_name: str,
        device: str,
        attack_cfg: dict,
        seed: int,
    ) -> np.ndarray:
        """Encode train labels and poison them only for configured malicious clients."""
        clean_labels = self.encode_labels(frame, task_name)
        if not attack_cfg.get("enabled", False):
            return clean_labels
        if attack_cfg.get("type", "label_flip") != "label_flip":
            raise ValueError(f"Unsupported attack type: {attack_cfg.get('type')}")
        if task_name != "binary":
            raise ValueError("The specified label-flip attack is valid only for the binary task")
        if device not in self.resolve_malicious_clients(attack_cfg):
            return clean_labels
        return flip_labels(clean_labels, float(attack_cfg["rho"]), seed)

    def class_names(self, task_name: str) -> list[str]:
        if task_name == "binary":
            return ["benign", "malicious"]
        return list(self.multiclass_encoder.classes_)

    def device_profile(self) -> pd.DataFrame:
        return (
            self.master_df.groupby(["device", "binary_label"])
            .size()
            .unstack(fill_value=0)
            .reindex(self.devices)
            .reset_index()
        )
