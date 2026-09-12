from __future__ import annotations

import json
from pathlib import Path

from config import SCHEMA_VERSION, build_runtime_config
from src.data_loader import DataRepository
from src.evaluation import confusion_matrix_table, dataset_profile_tables
from src.excel_export import export_excel_workbooks
from src.federated import ExperimentRunner
from src.plotting import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_cross_device_bars,
    plot_round_curves,
    plot_trust_scores,
)
from src.utils import ensure_directories, set_random_seed, timestamp_string


def load_config(project_root: Path) -> dict:
    runtime_path = project_root / "outputs" / "runtime_config.json"
    config = build_runtime_config()

    if runtime_path.exists():
        stored_config = json.loads(runtime_path.read_text(encoding="utf-8"))
        if stored_config.get("schema_version") == SCHEMA_VERSION:
            for key, value in stored_config.items():
                if key != "paths":
                    config[key] = value
        else:
            print("Ignoring legacy two-device outputs/runtime_config.json; using schema-v2 defaults.")

    config["paths"] = build_runtime_config()["paths"]
    return config


def prepare_run_config(config: dict) -> dict:
    run_id = f"run_{timestamp_string()}"
    runs_root = Path(config["paths"]["runs_root"])
    run_root = runs_root / run_id

    config["run_id"] = run_id
    config["paths"]["run_root"] = str(run_root)
    config["paths"]["tables_dir"] = str(run_root / "tables")
    config["paths"]["figures_dir"] = str(run_root / "figures")
    config["paths"]["audit_dir"] = str(run_root / "audit")
    config["paths"]["models_dir"] = str(run_root / "models")
    config["paths"]["logs_dir"] = str(run_root / "logs")
    return config


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = prepare_run_config(load_config(project_root))
    set_random_seed(config["random_seed"])
    ensure_directories(config)

    run_config_path = Path(config["paths"]["run_root"]) / "runtime_config.json"
    run_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    repo = DataRepository.from_config(config)
    runner = ExperimentRunner(config=config, repo=repo)
    outputs = runner.run_all()

    figures_dir = Path(config["paths"]["figures_dir"])
    plot_class_distribution(repo.master_df, figures_dir / "class_distribution.png")

    for task_name, task_output in outputs["tasks"].items():
        plot_confusion_matrix(
            task_output["centralized"]["confusion_matrix"],
            task_output["class_names"],
            figures_dir / f"confusion_centralized_{task_name}.png",
            title=f"Centralized Confusion Matrix ({task_name})",
        )
        plot_round_curves(
            task_output["federated_rounds"],
            figures_dir / f"round_curves_{task_name}.png",
            title=f"Federated Round Metrics ({task_name})",
        )
        plot_trust_scores(
            task_output["federated_rounds"],
            figures_dir / f"trust_scores_{task_name}.png",
            title=f"Trust Scores by Round ({task_name})",
        )
        plot_cross_device_bars(
            task_output["cross_device_rows"],
            figures_dir / f"cross_device_{task_name}.png",
            title=f"Cross-Device Generalization ({task_name})",
        )

    profile_tables = dataset_profile_tables(repo.master_df)
    confusion_tables = {
        task: confusion_matrix_table(task_output["centralized"]["confusion_matrix"], task_output["class_names"])
        for task, task_output in outputs["tasks"].items()
    }

    export_excel_workbooks(
        config=config,
        outputs=outputs,
        profile_tables=profile_tables,
        confusion_tables=confusion_tables,
    )

    summary_path = Path(config["paths"]["logs_dir"]) / "run_summary.json"
    summary_path.write_text(json.dumps(outputs["summary"], indent=2), encoding="utf-8")
    print(f"Run completed. Results written to {config['paths']['run_root']}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
