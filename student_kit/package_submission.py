"""Copy the chosen LoRA files and validate required submission artefacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results", default="results.json")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    required_adapter = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [name for name in required_adapter if not (checkpoint / name).is_file()]
    if missing:
        raise SystemExit(f"Checkpoint is missing: {', '.join(missing)}")
    results_path = Path(args.results)
    if not results_path.is_file():
        raise SystemExit(f"Missing {results_path}; run eval_self.py first")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    for key in ("base", "adapted", "delta"):
        if key not in results.get("summary", {}):
            raise SystemExit(f"results.json is missing summary.{key}")

    adapter_dir = Path("adapter")
    adapter_dir.mkdir(exist_ok=True)
    for name in required_adapter:
        shutil.copy2(checkpoint / name, adapter_dir / name)

    required_repo = [
        Path("adapter/adapter_config.json"), Path("adapter/adapter_model.safetensors"),
        Path("reward.py"), Path("student_kit/reward.py"), Path("train_config.yaml"),
        results_path, Path("report.md"),
    ]
    missing_repo = [str(path) for path in required_repo if not path.is_file()]
    if missing_repo:
        raise SystemExit("Submission is incomplete: " + ", ".join(missing_repo))
    print("Submission files are present. Remaining manual check: replace every TODO in report.md.")


if __name__ == "__main__":
    main()
