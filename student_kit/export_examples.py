"""Export representative base/adapted SVG pairs from results.json."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--output-dir", default="examples")
    args = parser.parse_args()
    report = json.loads(Path(args.results).read_text(encoding="utf-8"))
    cases = [case for case in report["cases"] if "adapted" in case]
    if not cases:
        raise SystemExit("results file has no adapted outputs")
    ranked = sorted(
        cases,
        key=lambda case: case["adapted"]["score"]["total"] - case["base"]["score"]["total"],
    )
    selected = [ranked[-1], ranked[len(ranked) // 2], ranked[0]]
    labels = ["best_delta", "median_delta", "worst_delta"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, case in zip(labels, selected):
        index = case["index"]
        base_name = f"{label}_{index:02d}_base.svg"
        adapted_name = f"{label}_{index:02d}_adapted.svg"
        (output / base_name).write_text(case["base"]["svg"], encoding="utf-8")
        (output / adapted_name).write_text(case["adapted"]["svg"], encoding="utf-8")
        delta = case["adapted"]["score"]["total"] - case["base"]["score"]["total"]
        rows.append(
            f"<h2>{label}: case {index}, reward delta {delta:+.3f}</h2>\n"
            f"<p>{html.escape(case['prompt'])}</p>\n"
            f"<div class='pair'><figure><img src='{base_name}'><figcaption>Base: "
            f"{case['base']['score']['total']}</figcaption></figure>"
            f"<figure><img src='{adapted_name}'><figcaption>Adapted: "
            f"{case['adapted']['score']['total']}</figcaption></figure></div>"
        )
    page = """<!doctype html><meta charset="utf-8"><title>SVG comparisons</title>
<style>body{font-family:sans-serif;max-width:1100px;margin:auto}.pair{display:flex;gap:30px}
figure{margin:0}img{width:360px;height:360px;border:1px solid #ccc}p{white-space:pre-wrap}</style>
""" + "\n".join(rows)
    (output / "comparisons.html").write_text(page, encoding="utf-8")
    print(f"Wrote {output / 'comparisons.html'}")


if __name__ == "__main__":
    main()
