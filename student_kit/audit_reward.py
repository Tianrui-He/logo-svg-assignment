"""Calibrate reward.py on references plus deliberately bad controls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from .reward import score_svg
except ImportError:  # Allow: python student_kit/audit_reward.py
    from reward import score_svg


CONTROLS = {
    "malformed": ("A blue circle", '<svg viewBox="0 0 256 256"><circle>'),
    "empty": ("A detailed red and blue badge", '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>'),
    "background_only": (
        "A detailed red and blue badge",
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><rect width="256" height="256" fill="#fff"/></svg>',
    ),
    "unsafe_script": (
        "A blue circle",
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><script>alert(1)</script><circle cx="128" cy="128" r="80" fill="#369"/></svg>',
    ),
}


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["train.jsonl", "valid.jsonl"])
    parser.add_argument("--output", default="reward_calibration.json")
    args = parser.parse_args()

    scores = []
    issue_counts: dict[str, int] = {}
    for row in load_rows([Path(value) for value in args.data]):
        messages = row["messages"]
        result = score_svg(messages[-2]["content"], messages[-1]["content"])
        scores.append(result)
        for issue in result["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    totals = [value["total"] for value in scores]
    controls = {name: score_svg(*pair) for name, pair in CONTROLS.items()}
    report = {
        "reference_count": len(scores),
        "reference_total": {
            "min": min(totals), "mean": statistics.mean(totals),
            "median": statistics.median(totals), "max": max(totals),
        },
        "reference_breakdown_mean": {
            key: statistics.mean(item["breakdown"][key] for item in scores)
            for key in scores[0]["breakdown"]
        },
        "reference_issue_counts": issue_counts,
        "negative_controls": controls,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

