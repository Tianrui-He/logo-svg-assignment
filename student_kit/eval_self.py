"""Generate and score base/adapted outputs under identical decoding settings.

Example:
    python student_kit/eval_self.py \
      --model ./gemma3-270m-it \
      --adapter ./output/.../checkpoint-best \
      --valid valid.jsonl --output results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

try:
    from .reward import score_svg
except ImportError:
    from reward import score_svg


def load_examples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit else rows


def split_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str, str]:
    messages = row["messages"]
    context = [dict(message) for message in messages if message["role"] != "assistant"]
    assistants = [message["content"] for message in messages if message["role"] == "assistant"]
    if not context or not assistants:
        raise ValueError("Each row must contain input messages and one assistant SVG.")
    user_messages = [message["content"] for message in context if message["role"] == "user"]
    return context, user_messages[-1], assistants[-1]


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (ValueError, TypeError):
            # Some Gemma templates do not accept a system role.  Preserve its
            # content by prepending it to the first user turn.
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            merged: list[dict[str, str]] = []
            for message in messages:
                if message["role"] == "system":
                    continue
                item = dict(message)
                if item["role"] == "user" and system:
                    item["content"] = system + "\n\n" + item["content"]
                    system = ""
                merged.append(item)
            return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
    # The pretrained Gemma checkpoint may not ship a tokenizer chat template.
    # Match ms-swift's `gemma3_text` template exactly in that case.  Swift merges
    # the system instruction and user request into one user turn.
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    query = f"{system}\n\n{user}" if system else user
    return (
        "<bos><start_of_turn>user\n"
        + query
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def import_peft_without_transformer_engine() -> Any:
    """Import PEFT with its optional Transformer Engine integration disabled.

    Some hosted GPU images contain a partial or ABI-incompatible
    ``transformer_engine`` installation. PEFT only probes it as an optional
    acceleration backend; ordinary LoRA loading does not require it.
    """
    import importlib.machinery
    import sys
    import types

    if "peft" not in sys.modules:
        for module_name in list(sys.modules):
            if module_name == "transformer_engine" or module_name.startswith(
                "transformer_engine."
            ):
                sys.modules.pop(module_name, None)

        te_stub = types.ModuleType("transformer_engine")
        te_stub.__spec__ = importlib.machinery.ModuleSpec(
            "transformer_engine", loader=None, is_package=True
        )
        te_stub.__path__ = []
        sys.modules["transformer_engine"] = te_stub

    from peft import PeftModel

    return PeftModel


def load_model(model_path: str, adapter: str | None = None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    load_kwargs = {
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **load_kwargs)
    except TypeError:  # transformers 4.x compatibility
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **load_kwargs)
    if adapter:
        PeftModel = import_peft_without_transformer_engine()
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate_all(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    max_new_tokens: int,
    repetition_penalty: float,
) -> list[str]:
    import torch

    outputs: list[str] = []
    device = next(model.parameters()).device
    for index, row in enumerate(examples, 1):
        context, _, _ = split_messages(row)
        text = render_chat(tokenizer, context)
        # Chat-template strings already contain <bos>; adding special tokens a
        # second time changes the exact prompt that was used during training.
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        stop_ids: list[int] = []
        configured_eos = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(configured_eos, int):
            stop_ids.append(configured_eos)
        elif configured_eos:
            stop_ids.extend(int(value) for value in configured_eos)
        if tokenizer.eos_token_id is not None:
            stop_ids.append(int(tokenizer.eos_token_id))
        end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_turn_id, int) and end_turn_id >= 0 and end_turn_id != tokenizer.unk_token_id:
            stop_ids.append(end_turn_id)
        stop_ids = list(dict.fromkeys(stop_ids))
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=stop_ids or tokenizer.eos_token_id,
            )
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # The protocol requires exactly one SVG document.  Keep the first
        # complete document if a model emits text after its closing tag.
        close_index = output.lower().find("</svg>")
        if close_index >= 0:
            output = output[: close_index + len("</svg>")]
        outputs.append(output)
        print(f"generated {index}/{len(examples)}", flush=True)
    return outputs


def summarise(scored: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [item["total"] for item in scored]
    breakdown_keys = scored[0]["breakdown"] if scored else {}
    return {
        "count": len(scored),
        "mean_total": round(statistics.mean(totals), 4) if totals else None,
        "median_total": round(statistics.median(totals), 4) if totals else None,
        "valid_svg_rate": round(statistics.mean(item["breakdown"]["validity"] >= 29 for item in scored), 4) if scored else None,
        "mean_breakdown": {
            key: round(statistics.mean(item["breakdown"][key] for item in scored), 4)
            for key in breakdown_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="ModelScope model ID or local model directory")
    parser.add_argument("--adapter", help="LoRA checkpoint directory; omit for base-only evaluation")
    parser.add_argument("--valid", default="valid.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Fail fast before base-model generation, and cache a PEFT import that
    # cannot probe this image's broken optional Transformer Engine stack.
    if args.adapter:
        import_peft_without_transformer_engine()

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    examples = load_examples(Path(args.valid), args.limit)
    started = time.time()

    print("Loading base model...", flush=True)
    base_model, tokenizer = load_model(args.model)
    base_outputs = generate_all(
        base_model, tokenizer, examples, args.max_new_tokens, args.repetition_penalty
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapted_outputs: list[str] | None = None
    if args.adapter:
        print("Loading adapted model...", flush=True)
        adapted_model, adapted_tokenizer = load_model(args.model, args.adapter)
        adapted_outputs = generate_all(
            adapted_model,
            adapted_tokenizer,
            examples,
            args.max_new_tokens,
            args.repetition_penalty,
        )

    cases = []
    base_scores = []
    adapted_scores = []
    for index, (row, base_output) in enumerate(zip(examples, base_outputs)):
        _, prompt, reference = split_messages(row)
        base_score = score_svg(prompt, base_output)
        base_scores.append(base_score)
        case = {
            "index": index,
            "prompt": prompt,
            "reference_svg": reference,
            "base": {"svg": base_output, "score": base_score},
        }
        if adapted_outputs is not None:
            adapted_output = adapted_outputs[index]
            adapted_score = score_svg(prompt, adapted_output)
            adapted_scores.append(adapted_score)
            case["adapted"] = {"svg": adapted_output, "score": adapted_score}
        cases.append(case)

    base_summary = summarise(base_scores)
    summary: dict[str, Any] = {"base": base_summary}
    if adapted_scores:
        adapted_summary = summarise(adapted_scores)
        delta = {
            "mean_total": round(adapted_summary["mean_total"] - base_summary["mean_total"], 4),
            "valid_svg_rate": round(adapted_summary["valid_svg_rate"] - base_summary["valid_svg_rate"], 4),
            "mean_breakdown": {
                key: round(adapted_summary["mean_breakdown"][key] - base_summary["mean_breakdown"][key], 4)
                for key in base_summary["mean_breakdown"]
            },
        }
        summary.update({"adapted": adapted_summary, "delta": delta})
    report = {
        "meta": {
            "base_model": args.model, "adapter": args.adapter,
            "valid_file": args.valid, "seed": args.seed,
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": args.max_new_tokens,
                "repetition_penalty": args.repetition_penalty,
                "stop_on_end_of_turn": True,
                "truncate_after_first_svg": True,
            },
            "elapsed_seconds": round(time.time() - started, 2),
        },
        "summary": summary,
        "cases": cases,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
