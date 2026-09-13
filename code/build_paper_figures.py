"""Publication-quality figures for the paper, generated from audited artifacts.

Every figure follows the same conventions: Okabe-Ito colorblind-safe palette
with marker/linestyle as secondary encoding, one y-axis per panel (never dual),
axes zoomed to the data, mean +/- std across the 5 seeds where applicable,
captions supplied by the manuscript, PDF + 300-dpi PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
RUNS = PROJECT / "outputs" / "runs"
OUT = PROJECT / "paper" / "figures"

G0_RUN = RUNS / "run_20260718_082416"
G1_RUN = RUNS / "run_20260718_103707"
G2_RUN = RUNS / "run_20260718_144214"
G3_RUN = RUNS / "run_20260718_112546"
G4_RUN = RUNS / "run_20260718_113441"

# Okabe & Ito (2008) colorblind-safe palette — fixed assignments, never cycled.
BLUE = "#0072B2"        # S4 / proposed
VERMILLION = "#D55E00"  # attack / no defense
ORANGE = "#E69F00"      # FLTrust / secondary baseline
GREEN = "#009E73"       # recall / Ecobee
PURPLE = "#CC79A7"      # precision
GRAY = "#7F7F7F"        # honest cohort / context

SHORT_DEVICE = {
    "Danmini_Doorbell": "Danmini",
    "Ecobee_Thermostat": "Ecobee",
    "Ennio_Doorbell": "Ennio",
    "Philips_B120N10_Baby_Monitor": "Philips",
    "Provision_PT_737E_Security_Camera": "Prov. 737E",
    "Provision_PT_838_Security_Camera": "Prov. 838",
    "Samsung_SNH_1011_N_Webcam": "Samsung",
    "SimpleHome_XCS7_1002_WHT_Security_Camera": "SimpleH. 1002",
    "SimpleHome_XCS7_1003_WHT_Security_Camera": "SimpleH. 1003",
}

plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.6,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # embed TrueType for camera-ready
    "ps.fonttype": 42,
})


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


def read_rounds(audit_path: Path) -> list[dict]:
    rows = []
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "round_summary":
                rows.append(record)
    return rows


# ---------------------------------------------------------------- fig 1: rho boundary
def fig1_rho_boundary() -> None:
    sweep = pd.read_csv(PROJECT / "paper" / "tables" / "g2_rho_sweep.csv").sort_values("rho")
    g0 = pd.read_csv(PROJECT / "paper" / "tables" / "g0_seeds_summary.csv")
    clean_cost = float(
        g0.loc[g0["scenario"] == "S1", "accuracy_mean"].iloc[0]
        - g0.loc[g0["scenario"] == "S1_trust", "accuracy_mean"].iloc[0]
    ) * 100
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(3.6, 3.9), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.12},
    )
    series = [
        ("s2_accuracy_drop", "s2_accuracy_drop_std", "FedAvg (no defense)", VERMILLION, "o", "-"),
        ("s4_accuracy_drop", "s4_accuracy_drop_std", "S4 trust (ours)", BLUE, "s", "-"),
        ("fltrust_accuracy_drop", "fltrust_accuracy_drop_std", "FLTrust", ORANGE, "^", "-"),
    ]
    for mean_col, std_col, label, color, marker, style in series:
        top.errorbar(
            sweep["rho"], sweep[mean_col] * 100, yerr=sweep[std_col] * 100,
            label=label, color=color, marker=marker, linestyle=style,
            markersize=4.5, capsize=2.5, linewidth=1.6,
        )
    top.axhline(clean_cost, color=BLUE, linestyle=":", linewidth=1.0, alpha=0.7)
    top.annotate(
        f"S4 clean-run cost ({clean_cost:.2f} pp)",
        xy=(0.32, clean_cost), xytext=(0.24, 0.62), fontsize=6.5, color=BLUE,
        arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.6},
    )
    low_rho = sweep[sweep["rho"] <= 0.7]
    max_gap_pp = float(abs(low_rho["s4_accuracy_drop"] - low_rho["s2_accuracy_drop"]).max() * 100)
    top.annotate(
        r"$\rho\leq0.7$: S4 vs. FedAvg" + "\n" + f"gap $\\leq$ {np.ceil(max_gap_pp * 100) / 100:.2f} pp",
        xy=(0.58, 0.85),
        fontsize=6.5, color=GRAY, ha="center", style="italic",
    )
    top.annotate(
        r"$1.77$ pp", xy=(1.0, 1.774), xytext=(0.84, 1.55),
        fontsize=7, color=VERMILLION,
        arrowprops={"arrowstyle": "->", "color": VERMILLION, "linewidth": 0.7},
    )
    top.set_ylabel("Accuracy drop vs. clean (pp)")
    top.legend(frameon=False, loc="upper left")
    top.set_ylim(bottom=-0.08)

    bottom.plot(
        sweep["rho"], sweep["recall"], color=GREEN, marker="s",
        markersize=4.5, label="Recall (cumulative)",
    )
    bottom.plot(
        sweep["rho"], sweep["precision"], color=PURPLE, marker="^",
        linestyle="--", markersize=4.5, label="Precision (cumulative)",
    )
    bottom.set_ylim(-0.05, 1.08)
    bottom.set_ylabel("S4 attribution score")
    bottom.set_xlabel(r"Label-flip ratio $\rho$")
    bottom.set_xticks(sweep["rho"])
    bottom.legend(frameon=False, loc="center right")
    save(fig, "fig1_rho_boundary")


# ------------------------------------------------------- fig 2: G1 main comparison
def fig2_main_comparison() -> None:
    summary = pd.read_csv(PROJECT / "paper" / "tables" / "g1_seeds_summary.csv")
    rows = summary[summary["group"] == "G1"].copy()
    labels = {
        "S1": "FedAvg (clean)",
        "B_cmean": "Clipped mean",
        "S4": "S4 trust (ours)",
        "B_tmean": "Trimmed mean†",
        "B_krum": "Krum",
        "S3": "S3 (legacy)‡",
        "S2m": "Coord. median",
        "B_mkrum": "Multi-Krum",
        "B_fltrust": "FLTrust",
        "S2": "FedAvg (no defense)",
        "B_fgold": "FoolsGold",
    }
    rows["label"] = rows["scenario"].map(labels)
    rows = rows.sort_values("accuracy_mean")
    x_lo = 98.2
    offscale = rows["accuracy_mean"] * 100 < x_lo
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for index, (_, row) in enumerate(rows.iterrows()):
        acc = row["accuracy_mean"] * 100
        std = row["accuracy_std"] * 100
        is_ours = row["scenario"] == "S4"
        is_clean = row["scenario"] == "S1"
        color = BLUE if is_ours else (GRAY if not is_clean else "#333333")
        if acc < x_lo:  # FoolsGold: annotate instead of distorting the axis
            ax.annotate(
                r"$\leftarrow$ " + f"{acc:.2f}", xy=(x_lo + 0.02, index),
                fontsize=7.5, color=GRAY, va="center", ha="left", clip_on=False,
                annotation_clip=False,
            )
            continue
        ax.errorbar(
            acc, index, xerr=std, fmt="D" if is_ours else "o",
            color=color, markersize=5 if is_ours else 4,
            capsize=2.5, elinewidth=1.1,
            markerfacecolor=color if is_ours else "white",
        )
    ax.axvline(
        float(rows.loc[rows["scenario"] == "S1", "accuracy_mean"].iloc[0]) * 100,
        color="#333333", linestyle=":", linewidth=1.0,
    )
    ax.set_yticks(y)
    ax.set_ylim(-0.7, len(rows) - 0.5)
    labels_fmt = [
        (f"$\\bf{{{lbl}}}$" if scenario == "S4" else lbl).replace(" ", r"\ ")
        if scenario == "S4" else lbl
        for lbl, scenario in zip(rows["label"], rows["scenario"])
    ]
    ax.set_yticklabels(labels_fmt)
    ax.set_xlim(x_lo, 99.78)
    ax.set_xlabel("Accuracy on clean global test (%)")
    ax.annotate(
        "attribution: P=R=1.0,\nTTD=8, FPR=0", xy=(99.399, list(rows["scenario"]).index("S4")),
        xytext=(98.42, list(rows["scenario"]).index("S4") + 2.2),
        fontsize=7, color=BLUE,
        arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.7},
    )
    ax.text(
        0.0, -0.24,
        "† not significantly different from S4 (paired $t$, $p=0.47$); all other"
        " pairwise gaps vs. S4: $p<0.01$.\n"
        "‡ S3 (legacy) rejects all clients after warm-up; its score therefore "
        "corresponds to the round-5 frozen model.",
        transform=ax.transAxes, fontsize=6.2, color=GRAY, va="top",
    )
    save(fig, "fig2_main_comparison")


# ------------------------------------------- fig 3: trust trajectories under attack
def fig3_trust_trajectories() -> None:
    rounds = read_rounds(G1_RUN / "audit" / "audit_S4_binary_seed0.jsonl")
    devices = [client["device"] for client in rounds[0]["clients"]]
    trust = {device: [] for device in devices}
    malicious = set()
    thresholds, flag_round = [], None
    for record in rounds:
        thresholds.append(record.get("trust_threshold"))
        for client in record["clients"]:
            trust[client["device"]].append(client["trust_score"])
            if client["is_malicious_ground_truth"]:
                malicious.add(client["device"])
        if flag_round is None and record["flagged_clients"]:
            flag_round = record["round"]
    xs = np.arange(1, len(rounds) + 1)

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.axvspan(0.5, 5.5, color="#EFEFEF", zorder=0)
    ax.text(3, 1.035, "warm-up", fontsize=7, color=GRAY, ha="center")
    for device in devices:
        if device in malicious:
            continue
        is_ecobee = device == "Ecobee_Thermostat"
        ax.plot(
            xs, trust[device],
            color=GREEN if is_ecobee else GRAY,
            linestyle="--" if is_ecobee else "-",
            linewidth=1.5 if is_ecobee else 0.9,
            alpha=1.0 if is_ecobee else 0.65,
            zorder=3 if is_ecobee else 2,
            label="Ecobee (honest)" if is_ecobee else None,
        )
    for index, device in enumerate(sorted(malicious)):
        ax.plot(
            xs, trust[device], color=VERMILLION,
            linestyle="-" if index == 0 else (0, (4, 1.5)),
            linewidth=1.8, zorder=4,
            label=f"{SHORT_DEVICE[device]} (attacker)",
        )
    ax.plot(
        xs, thresholds, color="black", linestyle=":", linewidth=1.2,
        label="Rejection threshold", zorder=5,
    )
    if flag_round is not None:
        ax.axvline(flag_round, color=VERMILLION, linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"attackers flagged\n(round {flag_round})", xy=(flag_round, 0.09),
            xytext=(flag_round + 4.5, 0.32), fontsize=7, color=VERMILLION,
            arrowprops={"arrowstyle": "->", "color": VERMILLION, "linewidth": 0.7},
        )
    honest_proxy = plt.Line2D([], [], color=GRAY, linewidth=0.9, alpha=0.65)
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(
        [honest_proxy, *handles], ["Honest clients", *labels_],
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
        columnspacing=1.0, handlelength=1.6,
    )
    ax.set_xlabel("Federated round")
    ax.set_ylabel(r"Trust score $T_k$")
    ax.set_ylim(-0.03, 1.09)
    ax.set_xlim(1, len(rounds))
    save(fig, "fig3_trust_trajectories")


# ------------------------------------------------ fig 4: attribution across rounds
def fig4_attribution_rounds() -> None:
    """Cumulative attribution for 2 vs 3 attackers.

    The m=2 case is perfect from round 8 with zero variance; the m=3 case shows
    the late-phase (round >= 20) false-positive strike flags eroding cumulative
    precision while recall and detection-window precision stay perfect.
    """
    m2 = pd.read_excel(G1_RUN / "tables" / "experiment_results.xlsx", sheet_name="attribution_S4")
    m3 = pd.read_excel(G4_RUN / "tables" / "experiment_results.xlsx", sheet_name="attribution_S4")

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.axvspan(0.5, 5.5, color="#EFEFEF", zorder=0)
    ax.text(3, 1.045, "warm-up", fontsize=7, color=GRAY, ha="center")
    recall = m2.groupby("round")["recall_cum"].mean()
    ax.plot(
        recall.index, recall.values, color=GREEN, linestyle="-",
        marker="s", markersize=3, markevery=5,
        label="Recall (m=2 and m=3)",
    )
    for sheet, label, style in (
        (m2, "Precision, m=2", "--"),
        (m3, "Precision, m=3", ":"),
    ):
        stats = sheet.groupby("round")["precision_cum"].agg(["mean", "std"])
        ax.plot(
            stats.index, stats["mean"], color=PURPLE, linestyle=style,
            marker="^", markersize=3, markevery=5, label=label,
        )
        ax.fill_between(
            stats.index,
            (stats["mean"] - stats["std"].fillna(0)).clip(0, 1),
            (stats["mean"] + stats["std"].fillna(0)).clip(0, 1),
            color=PURPLE, alpha=0.12, linewidth=0,
        )
    ax.annotate(
        "detection at round 8:\nP = R = 1.0, all seeds",
        xy=(8, 1.0), xytext=(12, 0.55), fontsize=7, color="#333333",
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 0.7},
    )
    ax.annotate(
        "late-phase FP flags\n(honest cosines shrink)",
        xy=(42.5, 0.95), xytext=(25, 0.74), fontsize=7, color=PURPLE,
        arrowprops={"arrowstyle": "->", "color": PURPLE, "linewidth": 0.7},
    )
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Cumulative attribution")
    ax.set_ylim(-0.03, 1.1)
    ax.set_xlim(1, int(recall.index.max()))
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)
    save(fig, "fig4_attribution_rounds")


# ----------------------------------------------------- fig 5: G0 clean diagnostics
def fig5_g0_diagnostics() -> None:
    cosines: dict[str, list[float]] = {}
    ratios: dict[str, list[float]] = {}
    for seed in range(5):
        for record in read_rounds(G0_RUN / "audit" / f"audit_S1_trust_binary_seed{seed}.jsonl"):
            if record["round"] <= 5:
                continue
            for client in record["clients"]:
                cosines.setdefault(client["device"], []).append(client["cosine_to_reference"])
                ratios.setdefault(client["device"], []).append(
                    client["delta_norm"] / (client["reference_norm"] + 1e-12)
                )
    devices = list(cosines)
    labels = [SHORT_DEVICE[device] for device in devices]
    highlight = [device == "SimpleHome_XCS7_1002_WHT_Security_Camera" for device in devices]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, data, ylabel in (
        (left, [cosines[d] for d in devices], r"cos($\Delta_k$, coord.-wise median)"),
        (right, [ratios[d] for d in devices], r"$\|\Delta_k\|$ / median norm"),
    ):
        boxes = ax.boxplot(
            data, tick_labels=labels, showfliers=False, patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.1},
            widths=0.55,
        )
        for patch, is_hot in zip(boxes["boxes"], highlight):
            patch.set_facecolor(ORANGE if is_hot else "#B8CFE5")
            patch.set_linewidth(0.7)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        for label_obj in ax.get_xticklabels():
            label_obj.set_ha("right")
    left.axhline(0.0, color=VERMILLION, linestyle=":", linewidth=1.0)
    right.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    right.set_yscale("log")
    right.set_yticks([0.5, 1.0, 2.0, 4.0, 8.0])
    right.set_yticklabels(["0.5", "1", "2", "4", "8"])
    right.minorticks_off()
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig5_g0_diagnostics")


# --------------------------------------------------------------- fig 6: G4 ablation
def fig6_ablation() -> None:
    summary = pd.read_csv(PROJECT / "paper" / "tables" / "g4_seeds_summary.csv")
    order = ["S4", "S4_minus_CT1", "S4_minus_verif", "S4_minus_CT3", "S3"]
    labels = {
        "S4": "S4 (full)",
        "S4_minus_CT1": "Mean-reference\nvariant",
        "S4_minus_verif": "Self-reported-\nloss variant",
        "S4_minus_CT3": "No-temporal-\nmemory\nvariant",
        "S3": "S3 (legacy)",
    }
    rows = summary.set_index("scenario").loc[order]
    y = np.arange(len(order))[::-1]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 2.8), sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.13},
    )
    for yi, scenario in zip(y, order):
        row = rows.loc[scenario]
        is_full = scenario == "S4"
        left.errorbar(
            row["accuracy_mean"] * 100, yi, xerr=row["accuracy_std"] * 100,
            fmt="D" if is_full else "o", color=BLUE if is_full else GRAY,
            markerfacecolor=BLUE if is_full else "white",
            markersize=5 if is_full else 4, capsize=2.5,
        )
        if scenario == "S4_minus_CT1":
            left.text(
                row["accuracy_mean"] * 100 + 0.06, yi - 0.42,
                f"SD = {row['accuracy_std'] * 100:.2f} pp", fontsize=6.5, color=GRAY,
            )
    left.set_yticks(y)
    left.set_yticklabels([labels[s] for s in order])
    left.set_xlabel("Accuracy (%)")
    left.set_xlim(97.3, 99.75)
    left.set_ylim(-0.65, len(order) - 0.35)

    bar_height = 0.34
    right.barh(
        y + bar_height / 2, rows["attribution_recall_mean"], height=bar_height,
        color=GREEN, label="Recall",
    )
    right.barh(
        y - bar_height / 2, rows["attribution_precision_mean"], height=bar_height,
        color=PURPLE, label="Precision",
    )
    right.set_yticks(y)
    right.tick_params(axis="y", labelleft=False)
    right.set_xlim(0, 1.05)
    right.set_xlabel("Attribution score")
    right.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2)
    for yi, scenario in zip(y, order):
        if rows.loc[scenario, "attribution_recall_mean"] == 0:
            right.text(0.03, yi, "no attribution", fontsize=7, color=GRAY, va="center")
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.24, top=0.97)
    save(fig, "fig6_ablation")


# ------------------------------------------------------ fig 7: malicious count (G3)
def fig7_malicious_count() -> None:
    summary = pd.read_csv(PROJECT / "paper" / "tables" / "g3_seeds_summary.csv")
    g0 = pd.read_csv(PROJECT / "paper" / "tables" / "g0_seeds_summary.csv")
    ceiling = g0.loc[g0["scenario"] == "S1_trust"].iloc[0]
    ceiling_mean = ceiling["accuracy_mean"] * 100
    ceiling_std = ceiling["accuracy_std"] * 100

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.axhspan(
        ceiling_mean - ceiling_std, ceiling_mean + ceiling_std,
        color=BLUE, alpha=0.10, zorder=0,
    )
    ax.axhline(ceiling_mean, color=BLUE, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(
        1.02, ceiling_mean + 0.012, "clean-trust ceiling (G0, no attack)",
        fontsize=6.5, color=BLUE,
    )
    for scenario, label, color, marker, style in (
        ("S4", "S4 trust (ours)", BLUE, "s", "-"),
        ("S2m", "Coord. median", GRAY, "o", "--"),
        ("B_mkrum", "Multi-Krum", ORANGE, "^", "-."),
    ):
        rows = summary[summary["scenario"] == scenario].sort_values("malicious_count")
        ax.errorbar(
            rows["malicious_count"], rows["accuracy_mean"] * 100,
            yerr=rows["accuracy_std"] * 100, label=label, color=color,
            marker=marker, linestyle=style, markersize=4.5, capsize=2.5,
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Number of malicious clients (of 9)")
    ax.set_ylabel("Accuracy (%)")
    # Per the paired per-seed analysis: the S4 slope is not significant
    # (p=0.084) and S4 at m=3 is indistinguishable from the clean ceiling
    # (p=0.66) -- the meaningful claims are ceiling-recovery and detection.
    ax.text(
        0.98, -0.30,
        "S4: recall = 1.0, TTD = 8 at every count;\n"
        "accuracy $\\approx$ clean ceiling (m=3: $p=0.66$)",
        transform=ax.transAxes, fontsize=6.8, color=BLUE, va="top", ha="right",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=6.8)
    save(fig, "fig7_malicious_count")


if __name__ == "__main__":
    # Publication set replaces the raw diagnostic copies.
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.png"):
        stale.unlink()
    fig1_rho_boundary()
    fig2_main_comparison()
    fig3_trust_trajectories()
    fig4_attribution_rounds()
    fig5_g0_diagnostics()
    fig6_ablation()
    fig7_malicious_count()
    print(f"[done] figures in {OUT}")
