from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _short_device(device: str) -> str:
    replacements = {
        "Philips_B120N10_Baby_Monitor": "Philips",
        "Provision_PT_737E_Security_Camera": "Provision 737E",
        "Provision_PT_838_Security_Camera": "Provision 838",
        "Samsung_SNH_1011_N_Webcam": "Samsung",
        "SimpleHome_XCS7_1002_WHT_Security_Camera": "SimpleHome 1002",
        "SimpleHome_XCS7_1003_WHT_Security_Camera": "SimpleHome 1003",
    }
    return replacements.get(device, device.replace("_", " "))


def plot_class_distribution(master_df: pd.DataFrame, output_path: Path) -> None:
    counts = master_df.groupby(["device", "label"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", figsize=(13, 6))
    plt.title("Class Distribution by Device")
    plt.ylabel("Samples")
    plt.xlabel("Device")
    _save_current(output_path)


def plot_confusion_matrix(confusion: np.ndarray, class_names: list[str], output_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(confusion, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            plt.text(col, row, int(confusion[row, col]), ha="center", va="center")
    _save_current(output_path)


def plot_round_curves(round_rows: list[dict], output_path: Path, title: str) -> None:
    if not round_rows:
        return
    grouped: dict[str, list[dict]] = {}
    for row in round_rows:
        grouped.setdefault(row.get("scenario", row["mode"]), []).append(row)
    plt.figure(figsize=(8, 5))
    for mode, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["round"])
        plt.plot(
            [row["round"] for row in rows],
            [row["metrics"]["f1"] for row in rows],
            marker="o", markersize=3, label=mode,
        )
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("F1 score")
    plt.legend()
    _save_current(output_path)


def plot_trust_scores(
    round_rows: list[dict],
    output_path: Path,
    title: str,
    malicious_clients: set[str] | None = None,
    trust_threshold: float | None = None,
) -> None:
    """Plot honest clients in green, attackers in red, and Ecobee dashed."""
    if not round_rows:
        return
    malicious = set(malicious_clients or [])
    grouped: dict[str, dict[str, list[float]]] = {}
    threshold_rounds: list[int] = []
    threshold_values: list[float] = []
    for row in round_rows:
        if row["mode"] not in {"trust", "trust_fed"}:
            continue
        if row.get("trust_threshold") is not None:
            threshold_rounds.append(row["round"])
            threshold_values.append(float(row["trust_threshold"]))
        for client in row["clients"]:
            device = client["device"]
            if client.get("is_malicious_ground_truth", False):
                malicious.add(device)
            grouped.setdefault(device, {"rounds": [], "trust": []})
            grouped[device]["rounds"].append(row["round"])
            grouped[device]["trust"].append(client["trust_score"])
    if not grouped:
        return

    honest_devices = [device for device in grouped if device not in malicious]
    malicious_devices = [device for device in grouped if device in malicious]
    honest_colors = plt.cm.Greens(np.linspace(0.45, 0.9, max(len(honest_devices), 2)))
    malicious_colors = plt.cm.Reds(np.linspace(0.55, 0.9, max(len(malicious_devices), 2)))
    color_by_device = {
        **{device: honest_colors[index] for index, device in enumerate(honest_devices)},
        **{device: malicious_colors[index] for index, device in enumerate(malicious_devices)},
    }

    plt.figure(figsize=(11, 6))
    for device, payload in grouped.items():
        is_ecobee = device == "Ecobee_Thermostat"
        label = _short_device(device)
        if device in malicious:
            label += " [malicious]"
        elif is_ecobee:
            label += " [Ecobee/honest]"
        plt.plot(
            payload["rounds"], payload["trust"],
            color=color_by_device[device],
            linestyle="--" if is_ecobee else "-",
            linewidth=2.2 if is_ecobee or device in malicious else 1.4,
            label=label,
        )
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Trust score $T_k$")
    plt.ylim(-0.02, 1.02)
    if threshold_rounds:
        plt.plot(
            threshold_rounds, threshold_values,
            color="black", linestyle=":", linewidth=1.4,
            label="rejection threshold (per round)",
        )
    elif trust_threshold is not None:
        plt.axhline(
            trust_threshold, color="black", linestyle=":", linewidth=1.2,
            label=f"$\\tau_T={trust_threshold:g}$",
        )
    plt.legend(fontsize=8, ncol=2)
    _save_current(output_path)


def plot_s1_diagnostics(
    round_rows: list[dict],
    output_dir: Path,
    warmup_rounds: int = 5,
    trust_threshold: float | None = None,
) -> list[Path]:
    """Export the three required clean-S1 diagnostic figures."""
    records: list[dict] = []
    for round_row in round_rows:
        if round_row["mode"] not in {"trust", "trust_fed"}:
            continue
        for client in round_row["clients"]:
            records.append(
                {
                    "round": round_row["round"],
                    "device": client["device"],
                    "cosine": client["cosine_to_reference"],
                    "delta_norm": client["delta_norm"],
                    "norm_ratio": client["delta_norm"] / (client["reference_norm"] + 1e-12),
                    "trust": client["trust_score"],
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        return []
    post_warmup = frame[frame["round"] > warmup_rounds]
    devices = list(dict.fromkeys(frame["device"]))
    labels = [_short_device(device) for device in devices]
    output_dir.mkdir(parents=True, exist_ok=True)

    cosine_path = output_dir / "s1_cosine_distribution.png"
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [post_warmup.loc[post_warmup["device"] == device, "cosine"] for device in devices],
        tick_labels=labels, showfliers=True,
    )
    plt.axhline(0.0, color="crimson", linestyle=":", linewidth=1)
    plt.ylabel(r"cos($\Delta_k$, coordinate-wise median)")
    plt.title("G0/S1 clean: cosine distribution after warm-up")
    plt.xticks(rotation=30, ha="right")
    _save_current(cosine_path)

    norm_path = output_dir / "s1_norm_distribution.png"
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [post_warmup.loc[post_warmup["device"] == device, "norm_ratio"] for device in devices],
        tick_labels=labels, showfliers=True,
    )
    plt.axhline(1.0, color="black", linestyle=":", linewidth=1)
    plt.ylabel(r"$\|\Delta_k\|$ / median norm")
    plt.title("G0/S1 clean: update-norm distribution after warm-up")
    plt.xticks(rotation=30, ha="right")
    _save_current(norm_path)

    trust_path = output_dir / "s1_trust_scores.png"
    plot_trust_scores(
        round_rows, trust_path, "G0/S1 clean: trust scores (9 clients)",
        trust_threshold=trust_threshold,
    )
    return [cosine_path, norm_path, trust_path]


def plot_attribution_curves(attribution: dict | None, output_path: Path, title: str) -> None:
    if not attribution or not attribution.get("per_round"):
        return
    frame = pd.DataFrame(attribution["per_round"])
    plt.figure(figsize=(9, 5))
    plt.plot(frame["round"], frame["precision_r"], color="tab:blue", alpha=0.45, label="Precision/round")
    plt.plot(frame["round"], frame["recall_r"], color="tab:orange", alpha=0.45, label="Recall/round")
    plt.plot(frame["round"], frame["precision_cum"], color="tab:blue", linewidth=2, label="Precision cumulative")
    plt.plot(frame["round"], frame["recall_cum"], color="tab:orange", linewidth=2, label="Recall cumulative")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Round")
    plt.ylabel("Attribution score")
    plt.title(title)
    plt.legend()
    _save_current(output_path)


def plot_rho_boundary(rho_sweep: pd.DataFrame, output_path: Path) -> None:
    if rho_sweep.empty:
        return
    frame = rho_sweep.sort_values("rho")
    figure, damage_axis = plt.subplots(figsize=(9, 5))
    detection_axis = damage_axis.twinx()
    damage_line = damage_axis.plot(
        frame["rho"], frame["accuracy_drop"], marker="o", color="tab:red",
        label="S2 (no defense) accuracy drop",
    )
    for column, label, color in (
        ("s4_accuracy_drop", "S4 (trust) accuracy drop", "tab:green"),
        ("fltrust_accuracy_drop", "FLTrust accuracy drop", "tab:orange"),
    ):
        if column in frame.columns and frame[column].notna().any():
            damage_line += damage_axis.plot(
                frame["rho"], frame[column], marker="^", color=color, label=label,
            )
    detection_line = detection_axis.plot(
        frame["rho"], frame["recall"], marker="s", color="tab:blue",
        label="S4 attribution recall",
    )
    damage_axis.set_xlabel(r"Label-flip ratio $\rho$")
    damage_axis.set_ylabel("Accuracy drop (S1 - S2)", color="tab:red")
    detection_axis.set_ylabel("Detection recall (S4)", color="tab:blue")
    detection_axis.set_ylim(-0.02, 1.02)
    damage_axis.set_title("Poisoning damage vs. trust detection boundary")
    lines = damage_line + detection_line
    damage_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_robustness_bars(summary: pd.DataFrame, output_path: Path) -> None:
    """One bar per experiment cell; scenarios repeated across malicious_count
    (e.g. G3) are labelled per count instead of silently dropped."""
    if summary.empty:
        return
    accuracy_column = "accuracy_mean" if "accuracy_mean" in summary else "accuracy"
    f1_column = "f1_macro_mean" if "f1_macro_mean" in summary else "f1_macro"
    frame = summary.copy()
    label = frame["scenario"].astype(str)
    if "malicious_count" in frame.columns and frame["scenario"].duplicated().any():
        label = label + frame["malicious_count"].map(
            lambda count: "" if pd.isna(count) else f" (m={int(count)})"
        )
    frame = frame.assign(_label=label).drop_duplicates("_label").set_index("_label")
    frame[[accuracy_column, f1_column]].plot(kind="bar", figsize=(11, 5))
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title("Robustness comparison")
    plt.legend(["Accuracy", "F1 macro"])
    plt.xlabel("")
    plt.xticks(rotation=25, ha="right")
    _save_current(output_path)


def plot_cross_device_bars(rows: list[dict], output_path: Path, title: str) -> None:
    if not rows:
        return
    scenarios = [row["scenario"] for row in rows]
    scores = [row["metrics"]["f1"] for row in rows]
    plt.figure(figsize=(14, 5))
    plt.bar(scenarios, scores)
    plt.title(title)
    plt.ylabel("F1 score")
    plt.xticks(rotation=65, ha="right", fontsize=7)
    _save_current(output_path)
