"""Print best checkpoints recorded by ms-swift/Transformers trainer states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="output")
    args = parser.parse_args()
    states = sorted(Path(args.output_dir).rglob("trainer_state.json"))
    if not states:
        raise SystemExit(f"No trainer_state.json found under {args.output_dir}")
    for state_path in states:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(f"state: {state_path}")
        print(f"best_model_checkpoint: {state.get('best_model_checkpoint')}")
        print(f"best_metric: {state.get('best_metric')}")
        print(f"global_step: {state.get('global_step')}")


if __name__ == "__main__":
    main()
