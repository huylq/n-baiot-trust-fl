"""Verify archived S4 attribution logs and export Table 5 evidence and figures."""
from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference/client_attribution"
SHORT_DEVICE = {
    "Danmini_Doorbell": "Danmini", "Ecobee_Thermostat": "Ecobee",
    "Ennio_Doorbell": "Ennio", "Philips_B120N10_Baby_Monitor": "Philips",
    "Provision_PT_737E_Security_Camera": "Prov. 737E",
    "Provision_PT_838_Security_Camera": "Prov. 838",
    "Samsung_SNH_1011_N_Webcam": "Samsung",
    "SimpleHome_XCS7_1002_WHT_Security_Camera": "SimpleH. 1002",
    "SimpleHome_XCS7_1003_WHT_Security_Camera": "SimpleH. 1003",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_chain(path: Path) -> list[dict]:
    entries = []
    previous = "GENESIS"
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        require(record.get("previous_hash") == previous, f"Broken chain: {path.name}:{number}")
        payload = {k: v for k, v in record.items() if k != "current_hash"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
        require(digest == record.get("current_hash"), f"Invalid hash: {path.name}:{number}")
        previous = digest
        entries.append(record)
    return entries


def replay_rounds(rounds: list[dict], devices: list[str], malicious: set[str],
                  security: dict, m: int, seed: int) -> list[dict]:
    windows = {device: deque(maxlen=security["window_W"]) for device in devices}
    traces = []
    for record in rounds:
        ri = record["round"]
        warmup = ri <= security["warmup_rounds"]
        require(record["in_warmup"] == warmup, f"Warm-up mismatch: m={m}, seed={seed}, round={ri}")
        clients = record["clients"]
        require(len(clients) == len(devices) and {c["device"] for c in clients} == set(devices),
                f"Client set mismatch: m={m}, seed={seed}, round={ri}")
        norms = np.array([c["delta_norm"] for c in clients])
        require(np.all(np.isfinite(norms)) and np.all(norms > 0), "Invalid update norms")
        require(np.allclose(norms, record["norms"], rtol=0, atol=1e-12), "Norm order mismatch")
        logs = np.log(np.maximum(norms, security["eps"]))
        median = np.median(logs)
        scale = max(1.4826 * np.median(np.abs(logs - median)), security["log_norm_mad_floor"])
        zscores = (logs - median) / scale
        require(np.allclose(zscores, record["norm_zscores"], rtol=0, atol=1e-12),
                "Recomputed norm z-scores disagree with the log")
        require(abs(float(np.median(norms)) - record["reference_norm"]) <= 1e-12,
                "Median update norm mismatch")
        flagged = set()
        for ci, client in enumerate(clients):
            device = client["device"]
            cosine = float(client["cosine_to_reference"])
            require(np.isfinite(cosine) and -1.0000001 <= cosine <= 1.0000001, "Invalid cosine")
            require(bool(client["is_malicious_ground_truth"]) == (device in malicious),
                    "Ground-truth identity mismatch")
            cos_violation = cosine < -security["cos_margin"]
            norm_violation = bool(zscores[ci] > security["norm_z_strike"])
            if not warmup:
                windows[device].append(cos_violation or norm_violation)
            strike = sum(windows[device])
            flag = not warmup and strike >= security["strike_W_neg"]
            require(strike == client["strike"] and flag == client["flagged"] == client["detected"],
                    f"Decision mismatch: m={m}, seed={seed}, round={ri}, client={device}")
            if flag:
                flagged.add(device)
            traces.append({"m": m, "seed": seed, "round": ri, "device": device,
                           "is_malicious": device in malicious, "in_warmup": warmup,
                           "cosine": cosine, "update_norm": float(norms[ci]),
                           "median_update_norm": record["reference_norm"],
                           "norm_z": float(zscores[ci]), "cosine_violation": cos_violation,
                           "norm_violation": norm_violation, "strike": strike, "flagged": flag})
        require(flagged == set(record["flagged_clients"]) == set(record["detected_clients"]),
                "Round-level flags disagree with client records")
    return traces


def summarize_run(traces: list[dict], devices: list[str], malicious: set[str],
                  m: int, seed: int, window_length: int) -> tuple[dict, list[dict]]:
    summary = {"m": m, "seed": seed}
    first = {}
    for row in traces:
        if row["flagged"]:
            first.setdefault(row["device"], row["round"])
    for horizon in (10, 50):
        ever = {r["device"] for r in traces if r["round"] <= horizon and r["flagged"]}
        tp, fp = len(ever & malicious), len(ever - malicious)
        summary.update({f"TP_{horizon}": tp, f"FP_{horizon}": fp,
                        f"FN_{horizon}": len(malicious - ever),
                        f"TN_{horizon}": len(set(devices) - malicious - ever),
                        f"precision_{horizon}": tp / len(ever) if ever else 0.0,
                        f"recall_{horizon}": tp / len(malicious)})
    times = [first[d] for d in malicious if d in first]
    summary["mean_ttd"] = mean(times) if times else None
    summary["flag_fpr"] = summary["FP_50"] / (len(devices) - len(malicious))
    summary["min_benign_cosine"] = min(r["cosine"] for r in traces if not r["is_malicious"])
    events = []
    for device in sorted(set(first) - malicious):
        first_round = first[device]
        window = [r for r in traces if r["device"] == device
                  and first_round - window_length < r["round"] <= first_round]
        events.append({"m": m, "seed": seed, "device": device,
                       "first_false_flag_round": first_round,
                       "window_start": max(1, first_round - window_length + 1),
                       "window_end": first_round,
                       "cosine_violation_rounds": ";".join(str(r["round"]) for r in window if r["cosine_violation"]),
                       "norm_violation_rounds": ";".join(str(r["round"]) for r in window if r["norm_violation"]),
                       "flagged_rounds": ";".join(str(r["round"]) for r in traces if r["device"] == device and r["flagged"])})
    return summary, events


def check_paper(rows: list[dict], path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reference = {(int(r["malicious_count"]), int(r["seed"])): r
                     for r in csv.DictReader(stream) if r["group"] == "G3" and r["scenario"] == "S4"}
    require({(r["m"], r["seed"]) for r in rows} == set(reference), "Reference configuration mismatch")
    maximum = 0.0
    pairs = {"precision_50": "attribution_precision", "recall_50": "attribution_recall",
             "flag_fpr": "false_positive_rate", "mean_ttd": "mean_time_to_detection"}
    for row in rows:
        for actual, expected in pairs.items():
            difference = abs(row[actual] - float(reference[row["m"], row["seed"]][expected]))
            require(difference <= 1e-12, f"Paper mismatch: m={row['m']}, seed={row['seed']}, {actual}")
            maximum = max(maximum, difference)
    return {"compared_values": len(rows) * len(pairs), "max_absolute_difference": maximum,
            "absolute_tolerance": 1e-12}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_traces(traces: list[dict], config: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 7.5, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.titlesize": 8, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
    blue, orange = "#0072B2", "#D55E00"
    for m in config["attacker_counts"]:
        malicious = set(config["malicious_clients_by_count"][str(m)])
        devices = [d for d in config["devices"] if d not in malicious]
        devices += [d for d in config["devices"] if d in malicious]
        lookup = {(r["seed"], r["round"], r["device"]): r for r in traces if r["m"] == m}
        fig = plt.figure(figsize=(4.8, 3.55))
        top = fig.add_axes([0.205, 0.585, 0.67, 0.29])
        bottom = fig.add_axes([0.205, 0.125, 0.67, 0.345], sharex=top)
        cax = fig.add_axes([0.90, 0.125, 0.022, 0.345])
        rounds = np.arange(1, config["rounds"] + 1)
        top.axvspan(0.5, config["security"]["warmup_rounds"] + 0.5, color="0.9", zorder=0)
        for truth, color in [(False, blue), (True, orange)]:
            values = np.array([[lookup[seed, int(ri), device]["cosine"]
                                for seed in config["seeds"] for device in devices
                                if (device in malicious) == truth] for ri in rounds])
            top.fill_between(rounds, values.min(axis=1), values.max(axis=1),
                             color=color, alpha=0.22, linewidth=0)
            top.plot(rounds, values.mean(axis=1), color=color, linewidth=1.15)
        margin = config["security"]["cos_margin"]
        top.axhline(-margin, color="0.2", linewidth=0.8, linestyle="--")
        top.text(1.5, -margin - 0.06, rf"$\tau_{{\cos}}=-{margin:g}$",
                 ha="left", va="top", fontsize=6.5)
        first_round = min(r["round"] for r in lookup.values() if r["is_malicious"] and r["flagged"])
        first_cosine = mean(r["cosine"] for r in lookup.values()
                            if r["is_malicious"] and r["round"] == first_round)
        top.axvline(first_round, color="0.35", linestyle=":", linewidth=0.7)
        top.annotate(f"First attacker flags: round {first_round}",
                     xy=(first_round, first_cosine), xytext=(15, -0.87), fontsize=6.2,
                     arrowprops={"arrowstyle": "->", "lw": 0.65})
        top.set_ylim(-1.06, 1.08)
        top.set_yticks([-1, 0, 1])
        top.set_ylabel("Cosine similarity")
        top.tick_params(axis="x", labelbottom=False)
        top.set_title(f"(a) Update alignment ($m={m}$)", loc="left", pad=4)
        top.grid(axis="y", color="0.88", linewidth=0.4)
        fig.legend(handles=[Patch(facecolor=blue, alpha=0.5, label="Benign clients"),
                            Patch(facecolor=orange, alpha=0.5, label="Attackers")],
                   loc="upper center", bbox_to_anchor=(0.55, 1.0), frameon=False,
                   ncol=2, handlelength=1.4, columnspacing=1.8)
        fractions = [[mean(lookup[seed, int(ri), device]["flagged"] for seed in config["seeds"])
                      for ri in rounds] for device in devices]
        cmap = ListedColormap(["#FFFFFF", "#D7E8F4", "#A6CEE3", "#6BAED6", "#3182BD", "#08519C"])
        norm = BoundaryNorm(np.linspace(-0.1, 1.1, 7), cmap.N)
        image = bottom.imshow(fractions, aspect="auto", interpolation="nearest", origin="upper",
                              extent=[0.5, 50.5, 8.5, -0.5], cmap=cmap, norm=norm)
        bottom.set_yticks(range(9), labels=[SHORT_DEVICE[d] + ("*" if d in malicious else "") for d in devices])
        bottom.tick_params(axis="y", length=0, pad=4)
        for tick in bottom.get_yticklabels()[-m:]:
            tick.set_color(orange)
            tick.set_fontweight("bold")
        bottom.set_yticks(np.arange(-0.5, 9, 1), minor=True)
        bottom.grid(axis="y", which="minor", color="0.83", linewidth=0.35)
        bottom.tick_params(axis="y", which="minor", length=0)
        bottom.axhline(8.5 - m, color="0.45", linewidth=0.7)
        bottom.set_title("(b) Per-round flags across five seeds", loc="left", pad=5)
        bottom.set_xlabel("Training round", labelpad=3)
        bottom.set_xlim(0.5, 50.5)
        bottom.set_xticks([1, 5, 10, 20, 30, 40, 50])
        cbar = fig.colorbar(image, cax=cax, ticks=np.arange(0, 1.01, 0.2), spacing="uniform")
        cbar.ax.tick_params(length=2, pad=2, labelsize=6)
        cbar.set_label("Fraction of seeds flagged", fontsize=7, labelpad=4)
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"attribution_m{m}.{suffix}",
                        metadata={"CreationDate": None, "ModDate": None} if suffix == "pdf" else None)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=REFERENCE / "audit")
    parser.add_argument("--configuration", type=Path, default=REFERENCE / "configuration.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/client_attribution")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new --output-dir.")
    config = json.loads(args.configuration.read_text(encoding="utf-8"))
    require(config["seeds"] == list(range(5)) and config["attacker_counts"] == [1, 2, 3]
            and config["rounds"] == 50 and config["rho"] == 1, "Expected the full Table 5 attribution grid")
    require(config["security"]["reference"] == "median" and config["security"]["use_memory"],
            "Expected median-reference S4 with strike memory")
    expected_names = {f"audit_S4_binary_rho1_m{m}_seed{s}.jsonl"
                      for m in config["attacker_counts"] for s in config["seeds"]}
    require({p.name for p in args.audit_dir.glob("*.jsonl")} == expected_names,
            "Audit directory must contain exactly the 15 S4 Table 5 logs")
    manifest = {}
    if args.audit_dir.resolve() == (REFERENCE / "audit").resolve():
        with (REFERENCE / "audit_manifest.csv").open(encoding="utf-8", newline="") as stream:
            manifest = {r["file"]: r["sha256"] for r in csv.DictReader(stream)}
        require(set(manifest) == expected_names, "Incomplete audit manifest")
    traces, summaries, events, sources = [], [], [], []
    for m in config["attacker_counts"]:
        malicious = set(config["malicious_clients_by_count"][str(m)])
        require(len(malicious) == m and malicious <= set(config["devices"]), "Invalid attacker identities")
        for seed in config["seeds"]:
            path = args.audit_dir / f"audit_S4_binary_rho1_m{m}_seed{seed}.jsonl"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if manifest:
                require(digest == manifest[path.name], f"Manifest mismatch: {path.name}")
            entries = read_chain(path)
            rounds = [e for e in entries if e.get("event") == "round_summary"]
            require([r["round"] for r in rounds] == list(range(1, 51)), f"Incomplete rounds: {path.name}")
            require(all(r["scenario"] == "S4" and r["seed"] == seed and r["rho"] == 1
                        and r["task"] == "binary" and r["mode"] == "trust" for r in rounds),
                    f"Configuration mismatch: {path.name}")
            run_traces = replay_rounds(rounds, config["devices"], malicious, config["security"], m, seed)
            summary, run_events = summarize_run(run_traces, config["devices"], malicious,
                                                m, seed, config["security"]["window_W"])
            traces.extend(run_traces)
            summaries.append(summary)
            events.extend(run_events)
            sources.append({"file": path.name, "sha256": digest})
    comparison = check_paper(summaries, ROOT / "reference/tables/all_seed_runs.csv")
    with (ROOT / "reference/tables/table7_detection_window_attribution.csv").open(encoding="utf-8-sig", newline="") as stream:
        reference_group = {1: "G3", 2: "G1", 3: "G2"}
        detection = {(int(r["malicious_count"]), int(r["seed"])): r for r in csv.DictReader(stream)
                     if r["scenario"] == "S4" and float(r["rho"]) == 1
                     and r["group"] == reference_group.get(int(r["malicious_count"]))}
    require(len(detection) == 15, "Incomplete detection-window reference")
    for row in summaries:
        expected = detection[row["m"], row["seed"]]
        require(row["precision_10"] == float(expected["precision_at_detection"])
                and row["recall_10"] == float(expected["recall_at_detection"]),
                "Detection-window metric mismatch")
    groups = []
    for m in config["attacker_counts"]:
        rows = [r for r in summaries if r["m"] == m]
        group = {"m": m, "seed_count": len(rows)}
        for metric in ("precision_10", "recall_10", "precision_50", "recall_50", "flag_fpr", "mean_ttd"):
            values = [r[metric] for r in rows]
            group.update({f"{metric}_mean": mean(values), f"{metric}_std": stdev(values)})
        groups.append(group)
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    write_csv(output / "client_round_traces.csv", traces)
    write_csv(output / "per_seed_summary.csv", summaries)
    write_csv(output / "summary.csv", groups)
    write_csv(output / "false_positive_events.csv", events,
              ["m", "seed", "device", "first_false_flag_round", "window_start", "window_end",
               "cosine_violation_rounds", "norm_violation_rounds", "flagged_rounds"])
    plot_traces(traces, config, output / "figures")
    report = {"verified_runs": len(summaries), "verified_client_round_decisions": len(traces),
              "hash_chains_verified": len(sources), "archived_manifest_verified": bool(manifest),
              "reference_comparison": comparison, "detection_window_values_compared": 30,
              "false_positive_client_seed_cases": len(events), "sources": sources}
    (output / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(summaries)} runs and {len(traces)} client-round decisions.")
    print(f"Matched {comparison['compared_values']} cumulative and 30 detection-window reference values.")
    print(f"Maximum cumulative metric difference: {comparison['max_absolute_difference']:.3g}.")
    print(f"Exported {len(events)} false-positive client-seed cases and three PDF/PNG figures to {output}.")


if __name__ == "__main__":
    main()
