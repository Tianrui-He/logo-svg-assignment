"""Programmatic proxy reward for prompt-conditioned SVG logo generation.

The reward is deliberately dependency-free.  It measures properties that can be
checked reliably without pretending to be a learned visual judge:

* validity (30): clean, parseable, safe SVG with the required canvas;
* geometry (20): non-degenerate primitives, sane coordinates and references;
* composition (15): useful element/palette complexity and visible structure;
* prompt fidelity (25): requested colours, primitive families and styles;
* anti-degeneration (10): rejects blank, repeated, trivial and bloated outputs.

``score_svg`` returns the full breakdown used in analysis.  ``reward`` and
``compute_reward`` are small compatibility wrappers that return only 0..1.
"""

from __future__ import annotations

import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Iterable


SVG_NS = "http://www.w3.org/2000/svg"
DRAWABLE_TAGS = {
    "path", "circle", "ellipse", "rect", "polygon", "polyline", "line", "use"
}
STRUCTURAL_TAGS = {
    "svg", "g", "defs", "linearGradient", "radialGradient", "stop",
    "clipPath", "mask", "filter", "style",
}
SAFE_FILTER_TAGS = {
    "feGaussianBlur", "feDropShadow", "feColorMatrix", "feTurbulence",
    "feDisplacementMap", "feMerge", "feMergeNode",
}
UNSAFE_TAGS = {"script", "foreignObject", "image", "iframe", "object", "embed"}
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")
RGB_RE = re.compile(
    r"rgba?\(\s*(\d+(?:\.\d+)?)\s*[, ]\s*(\d+(?:\.\d+)?)\s*[, ]\s*(\d+(?:\.\d+)?)",
    re.I,
)
URL_RE = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.I)

# Prototypes are intentionally broad: named-colour matching is weaker evidence
# than matching an explicit hexadecimal colour in the prompt.
NAMED_COLOURS: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20), "white": (245, 245, 245),
    "gray": (128, 128, 128), "grey": (128, 128, 128),
    "silver": (190, 190, 190), "cream": (247, 237, 216),
    "beige": (220, 205, 175), "brown": (105, 66, 38),
    "walnut": (91, 58, 32), "red": (210, 55, 55),
    "coral": (232, 114, 76), "orange": (235, 145, 45),
    "gold": (211, 165, 55), "golden": (225, 170, 55),
    "yellow": (238, 199, 58), "green": (70, 155, 90),
    "teal": (65, 160, 150), "blue": (50, 105, 165),
    "navy": (28, 58, 92), "cyan": (45, 180, 190),
    "purple": (115, 75, 155), "violet": (125, 80, 165),
    "pink": (225, 120, 155), "magenta": (200, 60, 150),
}

FEATURE_RULES: dict[str, set[str]] = {
    "circle": {"circle", "ellipse", "path"},
    "circular": {"circle", "ellipse", "path"},
    "ring": {"circle", "ellipse", "path"},
    "dot": {"circle", "ellipse"},
    "dots": {"circle", "ellipse"},
    "oval": {"ellipse", "path"},
    "ellipse": {"ellipse", "path"},
    "square": {"rect", "path", "polygon"},
    "rectangle": {"rect", "path", "polygon"},
    "rectangular": {"rect", "path", "polygon"},
    "triangle": {"polygon", "path"},
    "triangular": {"polygon", "path"},
    "star": {"polygon", "path"},
    "sparkle": {"polygon", "path"},
    "line": {"line", "polyline", "path"},
    "ray": {"line", "polyline", "path", "polygon"},
    "rays": {"line", "polyline", "path", "polygon"},
    "leaf": {"path", "ellipse"},
    "shield": {"path", "polygon"},
    "heart": {"path"},
    "hand": {"path"},
    "roof": {"path", "polygon", "line"},
    "ribbon": {"path"},
    "swirl": {"path"},
    "note": {"path", "circle", "ellipse", "line"},
}


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _numbers(value: str | None) -> list[float]:
    if not value:
        return []
    result: list[float] = []
    for match in NUMBER_RE.findall(value):
        try:
            result.append(float(match))
        except ValueError:
            continue
    return result


def _extract_svg(raw: str) -> tuple[str, bool, int]:
    """Return candidate SVG, whether it was the whole response, and root count."""
    text = (raw or "").strip()
    starts = list(re.finditer(r"<svg\b", text, flags=re.I))
    ends = list(re.finditer(r"</svg\s*>", text, flags=re.I))
    if not starts or not ends or ends[-1].end() <= starts[0].start():
        return text, False, len(starts)
    candidate = text[starts[0].start() : ends[-1].end()]
    clean = candidate == text
    return candidate, clean, len(starts)


def _parse_hex(value: str) -> tuple[int, int, int] | None:
    value = value.strip().lower()
    if not value.startswith("#"):
        return None
    digits = value[1:]
    if len(digits) in (3, 4):
        digits = "".join(ch * 2 for ch in digits[:3])
    elif len(digits) in (6, 8):
        digits = digits[:6]
    else:
        return None
    try:
        return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def _all_colours(text: str) -> list[tuple[int, int, int]]:
    colours: list[tuple[int, int, int]] = []
    for token in HEX_RE.findall(text or ""):
        parsed = _parse_hex(token)
        if parsed is not None:
            colours.append(parsed)
    for match in RGB_RE.finditer(text or ""):
        colours.append(tuple(max(0, min(255, round(float(v)))) for v in match.groups()))  # type: ignore[arg-type]
    # Include CSS colour words only when they appear as complete attribute values.
    for match in re.finditer(r"(?:fill|stroke|stop-color)\s*=\s*['\"]([a-zA-Z]+)['\"]", text or ""):
        colour = NAMED_COLOURS.get(match.group(1).lower())
        if colour:
            colours.append(colour)
    return colours


def _colour_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    # Normalised Euclidean RGB is adequate for a transparent, programmatic proxy.
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _opacity(element: ET.Element) -> float:
    values: list[float] = []
    style = element.attrib.get("style", "")
    merged = dict(element.attrib)
    for item in style.split(";"):
        if ":" in item:
            key, value = item.split(":", 1)
            merged.setdefault(key.strip(), value.strip())
    for key in ("opacity", "fill-opacity", "stroke-opacity"):
        if key in merged:
            try:
                values.append(float(merged[key]))
            except ValueError:
                pass
    return min(values, default=1.0)


def _is_visible(element: ET.Element) -> bool:
    tag = _local_name(element.tag)
    if tag not in DRAWABLE_TAGS or _opacity(element) <= 0.01:
        return False
    fill = element.attrib.get("fill", "").strip().lower()
    stroke = element.attrib.get("stroke", "").strip().lower()
    style = element.attrib.get("style", "").replace(" ", "").lower()
    if "display:none" in style or "visibility:hidden" in style:
        return False
    if tag in {"line", "polyline"}:
        return bool(stroke and stroke != "none") or "stroke:" in style and "stroke:none" not in style
    return fill != "none" or (stroke and stroke != "none") or "stroke:" in style


def _viewbox(root: ET.Element) -> tuple[float, float, float, float] | None:
    values = _numbers(root.attrib.get("viewBox") or root.attrib.get("viewbox"))
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        return None
    return values[0], values[1], values[2], values[3]


def _element_degenerate(element: ET.Element) -> bool:
    tag = _local_name(element.tag)
    a = element.attrib
    if tag == "circle":
        return not _numbers(a.get("r")) or _numbers(a.get("r"))[0] <= 0
    if tag == "ellipse":
        rx, ry = _numbers(a.get("rx")), _numbers(a.get("ry"))
        return not rx or not ry or rx[0] <= 0 or ry[0] <= 0
    if tag == "rect":
        width, height = _numbers(a.get("width")), _numbers(a.get("height"))
        return not width or not height or width[0] <= 0 or height[0] <= 0
    if tag == "line":
        vals = [_numbers(a.get(k)) for k in ("x1", "y1", "x2", "y2")]
        return any(not v for v in vals) or (vals[0][0] == vals[2][0] and vals[1][0] == vals[3][0])
    if tag in {"polygon", "polyline"}:
        vals = _numbers(a.get("points"))
        minimum = 6 if tag == "polygon" else 4
        return len(vals) < minimum
    if tag == "path":
        d = a.get("d", "")
        return len(d.strip()) < 4 or len(_numbers(d)) < 2 or not re.search(r"[Mm]", d)
    if tag == "use":
        return not (a.get("href") or a.get("{http://www.w3.org/1999/xlink}href"))
    return False


def _is_backdrop_rect(element: ET.Element, box: tuple[float, float, float, float]) -> bool:
    if _local_name(element.tag) != "rect":
        return False
    a = element.attrib
    x = (_numbers(a.get("x")) or [0.0])[0]
    y = (_numbers(a.get("y")) or [0.0])[0]
    w = (_numbers(a.get("width")) or [0.0])[0]
    h = (_numbers(a.get("height")) or [0.0])[0]
    bx, by, bw, bh = box
    return x <= bx and y <= by and x + w >= bx + bw and y + h >= by + bh


def _coordinate_health(
    elements: Iterable[ET.Element], box: tuple[float, float, float, float]
) -> tuple[float, int, int]:
    """Approximate coordinate health; exact path geometry is intentionally avoided."""
    bx, by, bw, bh = box
    low, high = min(bx, by) - 2 * max(bw, bh), max(bx + bw, by + bh) + 2 * max(bw, bh)
    checked = outside = 0
    coordinate_keys = {
        "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
        "width", "height", "points", "d",
    }
    for element in elements:
        if _is_backdrop_rect(element, box):
            continue
        values: list[float] = []
        for key, value in element.attrib.items():
            if _local_name(key) in coordinate_keys:
                values.extend(_numbers(value))
        for value in values:
            checked += 1
            if not math.isfinite(value) or value < low or value > high:
                outside += 1
    if checked == 0:
        return 0.0, checked, outside
    return 1.0 - outside / checked, checked, outside


def _reference_health(root: ET.Element) -> tuple[float, list[str]]:
    ids = [value for e in root.iter() if (value := e.attrib.get("id"))]
    counts = Counter(ids)
    issues: list[str] = []
    if any(count > 1 for count in counts.values()):
        issues.append("duplicate_id")
    known = set(ids)
    refs: list[str] = []
    for element in root.iter():
        for key, value in element.attrib.items():
            if _local_name(key) == "href" and value.startswith("#"):
                refs.append(value[1:])
            for match in URL_RE.finditer(value):
                target = match.group(2)
                if target.startswith("#"):
                    refs.append(target[1:])
    dangling = [ref for ref in refs if ref not in known]
    if dangling:
        issues.append("dangling_reference")
    if not ids and not refs:
        return 1.0, issues
    penalty = min(1.0, (sum(c - 1 for c in counts.values() if c > 1) + len(dangling)) / max(1, len(ids) + len(refs)))
    return 1.0 - penalty, issues


def _unsafe_content(root: ET.Element, candidate: str) -> list[str]:
    issues: list[str] = []
    if re.search(r"<!DOCTYPE|<!ENTITY", candidate, flags=re.I):
        issues.append("doctype_or_entity")
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag in UNSAFE_TAGS:
            issues.append(f"unsafe_tag:{tag}")
        for key, value in element.attrib.items():
            local_key = _local_name(key).lower()
            lower_value = value.strip().lower()
            if local_key.startswith("on"):
                issues.append("event_handler")
            if "javascript:" in lower_value or lower_value.startswith(("http://", "https://", "data:")):
                issues.append("external_or_active_reference")
            for match in URL_RE.finditer(value):
                if not match.group(2).startswith("#"):
                    issues.append("external_url")
    return sorted(set(issues))


def _prompt_colour_score(prompt: str, svg_colours: list[tuple[int, int, int]]) -> tuple[float, dict[str, Any]]:
    explicit = [_parse_hex(v) for v in HEX_RE.findall(prompt)]
    explicit = [v for v in explicit if v is not None]
    words = set(re.findall(r"[a-zA-Z]+", prompt.lower()))
    named = {name: rgb for name, rgb in NAMED_COLOURS.items() if name in words}

    def coverage(requested: Iterable[tuple[int, int, int]], threshold: float) -> float:
        req = list(requested)
        if not req:
            return 1.0
        if not svg_colours:
            return 0.0
        return statistics.mean(
            min(_colour_distance(colour, actual) for actual in svg_colours) <= threshold
            for colour in req
        )

    explicit_cov = coverage(explicit, 28.0) if explicit else None
    named_cov = coverage(named.values(), 95.0) if named else None
    if explicit_cov is not None and named_cov is not None:
        score = 0.8 * explicit_cov + 0.2 * named_cov
    elif explicit_cov is not None:
        score = explicit_cov
    elif named_cov is not None:
        score = named_cov
    else:
        score = 0.65  # neutral, not perfect, when lexical checking is impossible
    return score, {
        "explicit_colours_requested": len(explicit),
        "named_colours_requested": sorted(named),
        "explicit_colour_coverage": explicit_cov,
        "named_colour_coverage": named_cov,
    }


def _prompt_feature_score(prompt: str, tags: set[str], element_count: int, palette_size: int) -> tuple[float, dict[str, Any]]:
    words = set(re.findall(r"[a-zA-Z]+", prompt.lower()))
    requested = {word: allowed for word, allowed in FEATURE_RULES.items() if word in words}
    matched = {word: bool(allowed & tags) for word, allowed in requested.items()}
    feature_score = statistics.mean(matched.values()) if matched else 0.65

    lower = prompt.lower()
    has_gradient = bool({"linearGradient", "radialGradient"} & tags)
    gradient_free = bool(re.search(r"gradient[- ]free|no gradient|without (?:a )?gradient|flat colo", lower))
    asks_gradient = "gradient" in lower and not gradient_free
    gradient_score = 1.0 if (gradient_free and not has_gradient) or (asks_gradient and has_gradient) else 0.0
    if not gradient_free and not asks_gradient:
        gradient_score = 0.65

    asks_simple = bool(re.search(r"\b(minimal|simple|clean|small sizes?)\b", lower))
    if asks_simple:
        style_score = 1.0 if element_count <= 35 and palette_size <= 12 else _clamp(1.0 - (element_count - 35) / 80)
    else:
        style_score = 0.65
    combined = 0.70 * feature_score + 0.15 * gradient_score + 0.15 * style_score
    return combined, {
        "features_requested": sorted(requested),
        "features_matched": sorted(word for word, ok in matched.items() if ok),
        "gradient_requested": asks_gradient,
        "gradient_free_requested": gradient_free,
    }


def score_svg(prompt: str, raw_svg: str) -> dict[str, Any]:
    """Score one output and return a JSON-serialisable explanation.

    Scores are proxy measurements, not claims of aesthetic quality.  A severe
    safety violation caps the total at 20; an unparseable document scores at
    most the four clean-format points.
    """
    candidate, clean, root_count = _extract_svg(raw_svg)
    issues: list[str] = []
    validity = 0.0
    if clean:
        validity += 4.0
    else:
        issues.append("extra_text_or_incomplete_svg")
    if root_count != 1:
        issues.append("svg_root_count_not_one")
    else:
        validity += 2.0

    result: dict[str, Any] = {
        "total": 0.0,
        "normalized": 0.0,
        "breakdown": {
            "validity": round(validity, 3), "geometry": 0.0,
            "composition": 0.0, "prompt_fidelity": 0.0,
            "anti_degeneration": 0.0,
        },
        "issues": issues,
        "metrics": {"raw_characters": len(raw_svg or "")},
    }
    if not candidate or re.search(r"<!DOCTYPE|<!ENTITY", candidate, flags=re.I):
        issues.append("unsafe_or_unparseable_preamble")
        result["total"] = round(validity, 3)
        result["normalized"] = round(validity / 100.0, 5)
        return result
    try:
        root = ET.fromstring(candidate)
    except ET.ParseError as exc:
        issues.append("xml_parse_error")
        result["metrics"]["parse_error"] = str(exc)
        result["total"] = round(validity, 3)
        result["normalized"] = round(validity / 100.0, 5)
        return result

    validity += 12.0
    root_tag = _local_name(root.tag)
    if root_tag == "svg":
        validity += 3.0
    else:
        issues.append("root_is_not_svg")
    if root.tag.startswith("{" + SVG_NS + "}") or root.attrib.get("xmlns") == SVG_NS or f'xmlns="{SVG_NS}"' in candidate:
        validity += 2.0
    else:
        issues.append("missing_svg_namespace")

    box = _viewbox(root)
    if box:
        bx, by, bw, bh = box
        canvas_similarity = min(bw, bh) / max(bw, bh)
        required_similarity = 1.0 - min(1.0, (abs(bx) + abs(by) + abs(bw - 256) + abs(bh - 256)) / 512)
        validity += 4.0 * (0.5 * canvas_similarity + 0.5 * required_similarity)
        if required_similarity < 0.95:
            issues.append("nonstandard_viewbox")
    else:
        box = (0.0, 0.0, 256.0, 256.0)
        issues.append("missing_or_invalid_viewbox")

    unsafe = _unsafe_content(root, candidate)
    if not unsafe:
        validity += 3.0
    else:
        issues.extend(unsafe)

    all_elements = list(root.iter())
    visible = [element for element in all_elements if _is_visible(element)]
    tags = {_local_name(element.tag) for element in all_elements}
    visible_tags = {_local_name(element.tag) for element in visible}
    unknown_tags = tags - DRAWABLE_TAGS - STRUCTURAL_TAGS - SAFE_FILTER_TAGS - {"animate"}
    if unknown_tags:
        issues.append("unknown_tags:" + ",".join(sorted(unknown_tags)))

    # Geometry: visible structure, valid primitives, coordinates, local refs.
    geometry = 0.0
    if visible:
        geometry += 4.0
    else:
        issues.append("no_visible_elements")
    degenerate = [element for element in visible if _element_degenerate(element)]
    degenerate_ratio = len(degenerate) / max(1, len(visible))
    coordinate_score, coordinate_count, outside_count = _coordinate_health(visible, box)
    # An empty document must not earn points merely because it contains no bad
    # primitives or coordinates.  These two terms are conditional on content.
    if visible:
        geometry += 5.0 * (1.0 - degenerate_ratio)
        geometry += 8.0 * coordinate_score
    if degenerate:
        issues.append("degenerate_primitives")
    if outside_count:
        issues.append("extreme_or_outlying_coordinates")
    reference_score, reference_issues = _reference_health(root)
    geometry += 3.0 * reference_score
    issues.extend(reference_issues)

    # Composition: soft bands rather than brittle cut-offs.
    count = len(visible)
    if 3 <= count <= 60:
        count_score = 1.0
    elif count < 3:
        count_score = count / 3.0
    else:
        count_score = _clamp(1.0 - (count - 60) / 100.0)
    colours = _all_colours(candidate)
    unique_colours = set(colours)
    palette_size = len(unique_colours)
    if 2 <= palette_size <= 14:
        palette_score = 1.0
    elif palette_size < 2:
        palette_score = palette_size / 2.0
    else:
        palette_score = _clamp(1.0 - (palette_size - 14) / 20.0)
    tag_variety = min(1.0, len(visible_tags) / 3.0)
    visible_ratio = len(visible) / max(1, sum(_local_name(e.tag) in DRAWABLE_TAGS for e in all_elements))
    composition = 6.0 * count_score + 5.0 * palette_score + 2.0 * tag_variety + 2.0 * visible_ratio

    # Prompt fidelity: lexical checks that are inspectable and hard to dominate.
    colour_score, colour_metrics = _prompt_colour_score(prompt or "", colours)
    feature_score, feature_metrics = _prompt_feature_score(prompt or "", tags, count, palette_size)
    prompt_fidelity = 14.0 * colour_score + 11.0 * feature_score

    # Anti-degeneration: structure, length, repetition and non-trivial foreground.
    anti = 0.0
    if len(candidate) >= 120:
        anti += 1.5
    if 3 <= count <= 120:
        anti += 2.0
    if visible_ratio >= 0.75:
        anti += 1.5
    signatures = [
        (_local_name(e.tag), tuple(sorted((k, v) for k, v in e.attrib.items() if _local_name(k) != "id")))
        for e in visible
    ]
    most_repeated = max(Counter(signatures).values(), default=0)
    repeated_ratio = most_repeated / max(1, len(signatures))
    if repeated_ratio <= 0.35 or len(signatures) <= 4:
        anti += 2.0
    else:
        anti += 2.0 * _clamp(1.0 - (repeated_ratio - 0.35) / 0.65)
        issues.append("high_exact_repetition")
    foreground = [e for e in visible if not _is_backdrop_rect(e, box)]
    if len(foreground) >= 2:
        anti += 3.0
    else:
        issues.append("trivial_or_background_only")

    breakdown = {
        "validity": validity,
        "geometry": geometry,
        "composition": composition,
        "prompt_fidelity": prompt_fidelity,
        "anti_degeneration": anti,
    }
    total = sum(breakdown.values())
    if unsafe:
        total = min(total, 20.0)
        issues.append("safety_cap_applied")
    total = _clamp(total, 0.0, 100.0)
    result.update({
        "total": round(total, 3),
        "normalized": round(total / 100.0, 5),
        "breakdown": {key: round(value, 3) for key, value in breakdown.items()},
        "issues": sorted(set(issues)),
        "metrics": {
            "raw_characters": len(raw_svg or ""),
            "candidate_characters": len(candidate),
            "visible_elements": count,
            "element_tags": dict(sorted(Counter(_local_name(e.tag) for e in all_elements).items())),
            "palette_size": palette_size,
            "degenerate_elements": len(degenerate),
            "coordinates_checked": coordinate_count,
            "coordinates_outlying": outside_count,
            "visible_ratio": round(visible_ratio, 4),
            "exact_repetition_ratio": round(repeated_ratio, 4),
            **colour_metrics,
            **feature_metrics,
        },
    })
    return result


def reward(prompt: str, svg: str) -> float:
    """Compatibility API: return reward in [0, 1]."""
    return float(score_svg(prompt, svg)["normalized"])


def compute_reward(prompt: str, svg: str) -> float:
    """Alias used by common student-kit baselines."""
    return reward(prompt, svg)


def validate_svg(svg: str) -> bool:
    """A strict validity predicate; it intentionally ignores prompt fidelity."""
    scored = score_svg("", svg)
    return scored["breakdown"]["validity"] >= 29.0 and not any(
        issue in scored["issues"]
        for issue in ("no_visible_elements", "xml_parse_error", "safety_cap_applied")
    )


def score(prompt: str, svg: str) -> dict[str, Any]:
    """Convenience alias returning the full diagnostic object."""
    return score_svg(prompt, svg)
