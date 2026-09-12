from __future__ import annotations

import json
from pathlib import Path


SCHEMA_VERSION = 2

DEVICES = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]

ATTACK_LABELS = [
    "gafgyt_scan",
    "gafgyt_tcp",
    "gafgyt_udp",
    "gafgyt_combo",
    "gafgyt_junk",
    "mirai_ack",
    "mirai_scan",
    "mirai_syn",
    "mirai_udp",
    "mirai_udpplain",
]


def build_runtime_config() -> dict:
    """Return the reproducible nine-client experiment configuration.

    G0 contains clean baselines. Select other groups with ``--groups``.
    """
    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "dataset20000"
    output_root = project_root / "outputs"
    runs_root = output_root / "runs"

    return {
        "schema_version": SCHEMA_VERSION,
        "project_name": "NBaIoT_FL_9Device_Poisoning",
        "random_seed": 42,
        "seeds": [0, 1, 2, 3, 4],
        "task": "binary",
        "paths": {
            "project_root": str(project_root),
            "raw_data_dir": str(raw_dir),
            "output_root": str(output_root),
            "runs_root": str(runs_root),
        },
        "devices": DEVICES.copy(),
        "labels": ["benign", *ATTACK_LABELS],
        "tasks": {
            "binary": {"enabled": True, "class_names": ["benign", "malicious"]},
            "multiclass": {"enabled": False, "class_names": ["benign", *ATTACK_LABELS]},
        },
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "model": {
            "learning_rate": 0.01,
            "batch_size": 64,
            "l2": 0.0,
            "centralized_epochs": 20,
            "local_epochs": 20,
            "federated_rounds": 50,
            "federated_local_epochs": 1,
        },
        "attack": {
            "enabled": False,
            "type": "label_flip",
            "rho": 1.0,
            "rho_values": [0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
            "malicious_clients": [5, 8, 2],
        },
        "security": {
            "reference": "median",
            "use_self_reported_loss": False,
            "use_memory": True,
            "beta": 0.8,
            "window_W": 5,
            "strike_W_neg": 3,
            # Log-norm z-scores measure scale-free deviations within the cohort.
            "cos_margin": 0.1,
            "norm_z_soft": 2.0,
            "norm_z_strike": 3.5,
            "norm_z_admission": 4.5,
            "log_norm_mad_floor": 0.05,
            "tau_rel": 0.5,
            "tau_abs": 0.05,
            "clip_multiplier": 2.0,
            "warmup_rounds": 5,
            "eps": 1.0e-6,
            "eps_w": 0.1,
            "T_init": 0.5,
            "min_client_train_samples": 500,
        },
        "defense": {
            "byzantine_f": None,
            "multikrum_m": None,
            "trim_k": None,
            "clip_to_median_multiplier": 1.0,
            "fltrust_root_size": 200,
        },
        "experiments": {
            "groups": ["G0"],
            "malicious_clients_by_count": {
                "1": [5],
                "2": [5, 8],
                "3": [5, 8, 2],
            },
        },
        "export": {
            "write_models": True,
            "write_figures": True,
            "write_excel": True,
        },
    }


def validate_raw_files(raw_dir: Path) -> list[str]:
    files = sorted(p.name for p in raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return files


def main() -> None:
    config = build_runtime_config()
    raw_dir = Path(config["paths"]["raw_data_dir"])
    files = validate_raw_files(raw_dir)

    for key, path_str in config["paths"].items():
        path = Path(path_str)
        if key.endswith("_dir") or key.endswith("_root"):
            path.mkdir(parents=True, exist_ok=True)

    config["detected_files"] = files
    runtime_path = Path(config["paths"]["output_root"]) / "runtime_config.json"
    runtime_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Runtime configuration saved to: {runtime_path}")
    print(f"Detected {len(files)} CSV files for {len(config['devices'])} devices.")


if __name__ == "__main__":
    main()
