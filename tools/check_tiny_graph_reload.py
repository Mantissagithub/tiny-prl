#!/usr/bin/env python3
"""Prepare and compare same-name LoRA reload probes for eager and graph engines."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def prepare_fixtures(adapter_dir: Path, output_dir: Path, scale: float) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = adapter_dir / "adapter_model.safetensors"
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if not source.is_file() or not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Expected adapter files under {adapter_dir}")

    b_matrix_count = 0
    for label in ("A", "B"):
        target = output_dir / label
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite fixture directory: {target}")
        shutil.copytree(adapter_dir, target)
        tensors: dict[str, Any] = {}
        with safe_open(source, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                value = weights.get_tensor(key)
                if "lora_B" in key or ".lora_B." in key:
                    b_matrix_count += label == "A"
                    value = torch.zeros_like(value) if label == "A" else value * scale
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"Non-finite LoRA B tensor after scaling: {key}")
                value = value.contiguous()
                tensors[key] = value
        save_file(tensors, target / "adapter_model.safetensors", metadata={"format": "pt"})
    if b_matrix_count == 0:
        raise ValueError(f"No LoRA B tensors found in {source}")


def request_json(url: str, payload: dict[str, Any], api_key: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def probe_engine(
    base_url: str,
    model: str,
    adapter_name: str,
    fixture_dir: Path,
    prompt_tokens: list[int],
    api_key: str | None,
    mode: str,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    records = []
    for state in ("A", "B", "A"):
        adapter_path = (fixture_dir / state).resolve()
        request_json(
            f"{base_url}/load_lora_adapter",
            {"lora_name": adapter_name, "lora_path": str(adapter_path)},
            api_key,
        )
        response = request_json(
            f"{base_url}/v1/completions",
            {
                "model": model,
                "prompt": prompt_tokens,
                "temperature": 0,
                "seed": 0,
                "max_tokens": 4,
                "min_tokens": 4,
                "ignore_eos": True,
                "logprobs": 20,
                "return_token_ids": True,
                "stream": False,
                "cache_salt": f"tiny-graph-reload-{mode}-{state}-{uuid.uuid4().hex}",
            },
            api_key,
        )
        records.append({"state": state, "response": response})
    return {
        "mode": mode,
        "model": model,
        "adapter_name": adapter_name,
        "prompt_token_ids": prompt_tokens,
        "records": records,
    }


def choice(record: dict[str, Any]) -> dict[str, Any]:
    return record["response"]["choices"][0]


def token_ids(record: dict[str, Any]) -> list[int]:
    return choice(record).get("token_ids", [])


def selected_logprobs(record: dict[str, Any]) -> list[float | None]:
    logprobs = choice(record).get("logprobs") or {}
    return logprobs.get("token_logprobs") or []


def top_logprobs(record: dict[str, Any]) -> list[dict[str, float] | None]:
    logprobs = choice(record).get("logprobs") or {}
    return logprobs.get("top_logprobs") or []


def compare_records(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> None:
    left_ids, right_ids = token_ids(left), token_ids(right)
    if len(left_ids) != 4 or len(right_ids) != 4:
        raise ValueError("Each completion must return exactly four generated token IDs")
    if left_ids != right_ids:
        raise ValueError(f"Token sequences differ: {left_ids} != {right_ids}")
    left_values, right_values = selected_logprobs(left), selected_logprobs(right)
    left_top, right_top = top_logprobs(left), top_logprobs(right)
    if len(left_values) != 4 or len(right_values) != 4:
        raise ValueError("Each completion must include four selected-token logprobs")
    if len(left_top) != 4 or len(right_top) != 4:
        raise ValueError("Each completion must include four top-logprob distributions")
    for index, (a, b) in enumerate(zip(left_values, right_values, strict=True)):
        if (
            a is None
            or b is None
            or not math.isfinite(a)
            or not math.isfinite(b)
            or not math.isclose(a, b, rel_tol=0, abs_tol=tolerance)
        ):
            raise ValueError(f"Selected-token logprob differs at generated token {index}: {a} != {b}")
        a_top, b_top = left_top[index], right_top[index]
        if (
            not a_top
            or not b_top
            or any(not math.isfinite(value) for value in a_top.values())
            or any(not math.isfinite(value) for value in b_top.values())
            or a_top.keys() != b_top.keys()
        ):
            raise ValueError(f"Top-logprob tokens differ at generated token {index}")
        for token in a_top:
            if not math.isclose(a_top[token], b_top[token], rel_tol=0, abs_tol=tolerance):
                raise ValueError(f"Top logprob differs at generated token {index}, token {token}")


def compare_runs(eager: dict[str, Any], graph: dict[str, Any], tolerance: float) -> dict[str, Any]:
    if eager["mode"] != "eager" or graph["mode"] != "graph":
        raise ValueError("Input files must be tagged as eager and graph, respectively")
    if eager["adapter_name"] != graph["adapter_name"] or eager["model"] != graph["model"]:
        raise ValueError("Eager and graph runs must use the same model and adapter name")
    if not eager["prompt_token_ids"] or eager["prompt_token_ids"] != graph["prompt_token_ids"]:
        raise ValueError("Eager and graph runs must use the same nonempty prompt token IDs")
    for run in (eager, graph):
        if [item["state"] for item in run["records"]] != ["A", "B", "A"]:
            raise ValueError(f"{run['mode']} result must contain A/B/A records")
        compare_records(run["records"][0], run["records"][2], tolerance)

    eager_records = eager["records"]
    graph_records = graph["records"]
    for left, right in zip(eager_records, graph_records, strict=True):
        compare_records(left, right, tolerance)

    eager_a, eager_b = eager_records[0], eager_records[1]
    b_changed = token_ids(eager_a) != token_ids(eager_b)
    for a, b in zip(selected_logprobs(eager_a), selected_logprobs(eager_b)):
        if a is not None and b is not None and abs(a - b) > tolerance:
            b_changed = True
    for a_top, b_top in zip(top_logprobs(eager_a), top_logprobs(eager_b)):
        if a_top is None or b_top is None or a_top.keys() != b_top.keys():
            b_changed = True
        elif any(abs(a_top[token] - b_top[token]) > tolerance for token in a_top):
            b_changed = True
    if not b_changed:
        raise ValueError("Scaled B fixture produced no measurable change from zero-B A")

    return {
        "eager_a_restored": True,
        "graph_a_restored": True,
        "eager_graph_parity": True,
        "scaled_b_changed_output_or_selected_logprobs": True,
        "eager_a_first_token": token_ids(eager_a)[:1],
        "eager_b_first_token": token_ids(eager_b)[:1],
        "eager_a_first_logprob": selected_logprobs(eager_a)[:1],
        "eager_b_first_logprob": selected_logprobs(eager_b)[:1],
    }


def parse_prompt_tokens(value: str) -> list[int]:
    tokens = [int(part) for part in value.split(",") if part]
    if not tokens:
        raise argparse.ArgumentTypeError("provide at least one comma-separated prompt token ID")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="create isolated zero-B/scaled-B fixtures")
    prepare.add_argument("adapter_dir", type=Path)
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument("--scale", type=float, default=1000.0)

    probe = commands.add_parser("probe", help="run A/B/A against one already-running server")
    probe.add_argument("base_url")
    probe.add_argument("mode", choices=("eager", "graph"))
    probe.add_argument("fixture_dir", type=Path)
    probe.add_argument("output", type=Path)
    probe.add_argument("--model", required=True)
    probe.add_argument("--adapter-name", required=True)
    probe.add_argument("--prompt-token-ids", required=True, type=parse_prompt_tokens)
    probe.add_argument("--api-key")

    compare = commands.add_parser("compare", help="validate eager/graph A/B/A result files")
    compare.add_argument("eager", type=Path)
    compare.add_argument("graph", type=Path)
    compare.add_argument("output", type=Path)
    compare.add_argument("--tolerance", type=float, default=1e-3)

    args = parser.parse_args()
    if args.command == "prepare":
        if not math.isfinite(args.scale) or args.scale <= 0:
            parser.error("--scale must be finite and positive")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prepare_fixtures(args.adapter_dir, args.output_dir, args.scale)
        return
    if args.command == "probe":
        result = probe_engine(
            args.base_url,
            args.model,
            args.adapter_name,
            args.fixture_dir,
            args.prompt_token_ids,
            args.api_key,
            args.mode,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return

    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance must be finite and nonnegative")
    result = compare_runs(
        json.loads(args.eager.read_text()),
        json.loads(args.graph.read_text()),
        args.tolerance,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
