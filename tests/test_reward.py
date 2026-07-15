import json
from pathlib import Path

from student_kit.reward import score_svg, validate_svg


GOOD = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
<circle cx="128" cy="128" r="100" fill="#1B3A5C"/>
<path d="M72 150 C92 100 164 100 184 150 Z" fill="#5DA88E"/>
<circle cx="128" cy="105" r="22" fill="#F2A93B"/>
</svg>"""


def test_good_svg_is_high_and_valid():
    prompt = "A navy circular badge with a teal leaf and a golden circle."
    result = score_svg(prompt, GOOD)
    assert result["total"] >= 85
    assert validate_svg(GOOD)


def test_unclosed_xml_is_low():
    result = score_svg("a blue circle", '<svg viewBox="0 0 256 256"><circle>')
    assert result["total"] <= 6
    assert "xml_parse_error" in result["issues"] or "extra_text_or_incomplete_svg" in result["issues"]


def test_script_is_capped_even_when_svg_parses():
    malicious = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
    <script>alert(1)</script><circle cx="128" cy="128" r="90" fill="#123456"/>
    <rect x="80" y="80" width="96" height="96" fill="#abcdef"/></svg>"""
    result = score_svg("a blue circular badge", malicious)
    assert result["total"] <= 20
    assert "safety_cap_applied" in result["issues"]


def test_extra_prose_loses_clean_output_points():
    clean = score_svg("a navy badge", GOOD)
    wrapped = score_svg("a navy badge", "Here is the logo:\n```svg\n" + GOOD + "\n```")
    assert wrapped["total"] < clean["total"]
    assert "extra_text_or_incomplete_svg" in wrapped["issues"]


def test_prompt_colour_changes_fidelity():
    correct = score_svg("A navy (#1B3A5C), teal (#5DA88E), golden (#F2A93B) badge", GOOD)
    wrong = score_svg("A red (#FF0000), pink (#FF66AA), purple (#800080) badge", GOOD)
    assert correct["breakdown"]["prompt_fidelity"] > wrong["breakdown"]["prompt_fidelity"] + 8


def test_blank_and_background_only_are_penalised():
    blank = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>'
    background = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><rect width="256" height="256" fill="#fff"/></svg>'
    assert score_svg("a detailed logo", blank)["total"] < 55
    result = score_svg("a detailed logo", background)
    assert result["total"] < 70
    assert "trivial_or_background_only" in result["issues"]


def test_all_published_reference_svgs_parse_and_score_high():
    root = Path(__file__).parents[1]
    for filename in ("train.jsonl", "valid.jsonl"):
        path = root / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prompt = row["messages"][-2]["content"]
            svg = row["messages"][-1]["content"]
            scored = score_svg(prompt, svg)
            assert scored["total"] >= 85, scored

