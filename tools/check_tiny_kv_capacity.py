#!/usr/bin/env python3
"""Stress two concurrent full-context requests and verify the vLLM preemption counter."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

METRIC = "vllm:num_preemptions_total"
PROMPT_METRIC = "vllm:prompt_tokens_total"
GENERATION_METRIC = "vllm:generation_tokens_total"
PROMPT_TOKENS = 448
OUTPUT_TOKENS = 64
METRICS_POLL_DEADLINE_SECONDS = 15


def http_request(url: str, payload: dict[str, Any] | None, api_key: str | None) -> tuple[dict[str, Any] | str, float]:
    headers = {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers)
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        body = response.read().decode()
    elapsed = time.perf_counter() - start
    if payload is None:
        return body, elapsed
    return json.loads(body), elapsed


def preemption_counter(metrics_text: str) -> tuple[float, list[dict[str, Any]]]:
    counter_type_seen = False
    samples = []
    for line in metrics_text.splitlines():
        if line.startswith(f"# TYPE {METRIC} "):
            if line.split()[-1] != "counter":
                raise ValueError(f"{METRIC} is not exposed as a counter")
            counter_type_seen = True
            continue
        if line.startswith("#"):
            continue
        fields = line.split()
        if not fields:
            continue
        sample_name = fields[0]
        if sample_name != METRIC and not sample_name.startswith(f"{METRIC}{{"):
            continue
        if len(fields) < 2:
            raise ValueError(f"Malformed {METRIC} sample: {line}")
        value = float(fields[1])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {METRIC} sample: {line}")
        samples.append({"sample": sample_name, "value": value})
    if not counter_type_seen or not samples:
        raise ValueError(f"Required counter {METRIC} is missing from /metrics")
    return sum(sample["value"] for sample in samples), samples


def counter_total(metrics_text: str, metric: str) -> tuple[float, list[dict[str, Any]]]:
    counter_type_seen = False
    samples = []
    for line in metrics_text.splitlines():
        if line.startswith(f"# TYPE {metric} "):
            if line.split()[-1] != "counter":
                raise ValueError(f"{metric} is not exposed as a counter")
            counter_type_seen = True
            continue
        if line.startswith("#"):
            continue
        fields = line.split()
        if not fields:
            continue
        sample_name = fields[0]
        if sample_name != metric and not sample_name.startswith(f"{metric}{{"):
            continue
        if len(fields) < 2:
            raise ValueError(f"Malformed {metric} sample: {line}")
        value = float(fields[1])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {metric} sample: {line}")
        samples.append({"sample": sample_name, "value": value})
    if not counter_type_seen or not samples:
        raise ValueError(f"Required counter {metric} is missing from /metrics")
    return sum(sample["value"] for sample in samples), samples


def load_adapter(base_url: str, adapter_name: str, adapter_dir: Path, api_key: str | None) -> dict[str, Any]:
    result, _ = http_request(
        f"{base_url}/load_lora_adapter",
        {"lora_name": adapter_name, "lora_path": str(adapter_dir.resolve())},
        api_key,
    )
    return result


def completion_request(
    base_url: str,
    model: str,
    request_number: int,
    prompt: list[int],
    api_key: str | None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "add_special_tokens": False,
        "temperature": 0,
        "seed": 0,
        "max_tokens": OUTPUT_TOKENS,
        "min_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": False,
        "cache_salt": f"tiny-kv-stress-{request_number}-{uuid.uuid4().hex}",
    }
    response, elapsed = http_request(f"{base_url}/v1/completions", payload, api_key)
    return {"request": payload, "response": response, "elapsed_seconds": elapsed}


def validate_completion(record: dict[str, Any]) -> list[str]:
    response = record["response"]
    errors = []
    usage = response.get("usage") or {}
    if usage.get("prompt_tokens") != PROMPT_TOKENS:
        errors.append(f"prompt token count was {usage.get('prompt_tokens')!r}")
    if usage.get("completion_tokens") != OUTPUT_TOKENS:
        errors.append(f"completion token count was {usage.get('completion_tokens')!r}")
    choices = response.get("choices") or []
    if len(choices) != 1:
        errors.append(f"expected one completion choice, got {len(choices)}")
    elif len(choices[0].get("token_ids") or []) != OUTPUT_TOKENS:
        errors.append("completion did not return exactly 64 token IDs")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--adapter-name")
    parser.add_argument("--api-key")
    args = parser.parse_args()
    if (args.adapter_dir is None) != (args.adapter_name is None):
        parser.error("--adapter-dir and --adapter-name must be provided together")

    base_url = args.base_url.rstrip("/")
    prompt = [17] * PROMPT_TOKENS
    adapter_result = None
    if args.adapter_dir is not None:
        adapter_result = load_adapter(base_url, args.adapter_name, args.adapter_dir, args.api_key)

    metrics_before_raw, _ = http_request(f"{base_url}/metrics", None, args.api_key)
    before_total, before_samples = preemption_counter(metrics_before_raw)
    generation_before, generation_before_samples = counter_total(metrics_before_raw, GENERATION_METRIC)
    prompt_before, prompt_before_samples = counter_total(metrics_before_raw, PROMPT_METRIC)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(completion_request, base_url, args.model, index, prompt, args.api_key) for index in (1, 2)
        ]
        records = [future.result() for future in futures]
    elapsed = time.perf_counter() - start
    metrics_after_raw = None
    generation_after = None
    prompt_after = None
    generation_after_samples = None
    prompt_after_samples = None
    metric_deadline = time.monotonic() + METRICS_POLL_DEADLINE_SECONDS
    while True:
        metrics_after_raw, _ = http_request(f"{base_url}/metrics", None, args.api_key)
        generation_after, generation_after_samples = counter_total(metrics_after_raw, GENERATION_METRIC)
        prompt_after, prompt_after_samples = counter_total(metrics_after_raw, PROMPT_METRIC)
        if generation_after < generation_before or prompt_after < prompt_before:
            raise SystemExit("vLLM token counter reset during the stress test")
        if (
            generation_after - generation_before >= 2 * OUTPUT_TOKENS
            and prompt_after - prompt_before >= 2 * PROMPT_TOKENS
        ):
            break
        if time.monotonic() >= metric_deadline:
            break
        time.sleep(0.5)
    after_total, after_samples = preemption_counter(metrics_after_raw)

    errors = [error for record in records for error in validate_completion(record)]
    generation_delta = generation_after - generation_before
    prompt_delta = prompt_after - prompt_before
    if generation_delta < 2 * OUTPUT_TOKENS:
        errors.append(f"generation counter increased by only {generation_delta} tokens before deadline")
    if prompt_delta < 2 * PROMPT_TOKENS:
        errors.append(f"prompt counter increased by only {prompt_delta} tokens before deadline")
    if after_total < before_total:
        errors.append("preemption counter reset during the stress test")
        preemption_delta = None
    else:
        preemption_delta = after_total - before_total
        if preemption_delta != 0:
            errors.append(f"preemption counter increased by {preemption_delta}")

    result = {
        "model": args.model,
        "base_url": base_url,
        "adapter": (
            {"name": args.adapter_name, "path": str(args.adapter_dir.resolve()), "response": adapter_result}
            if args.adapter_dir is not None
            else None
        ),
        "parameters": {
            "concurrent_requests": 2,
            "prompt_token_id": 17,
            "prompt_tokens": PROMPT_TOKENS,
            "max_tokens": OUTPUT_TOKENS,
            "min_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 0,
            "distinct_cache_salts": True,
        },
        "requests": records,
        "elapsed_seconds": elapsed,
        "preemptions": {
            "metric": METRIC,
            "before_total": before_total,
            "before_samples": before_samples,
            "after_total": after_total,
            "after_samples": after_samples,
            "delta": preemption_delta,
        },
        "token_counters": {
            "poll_deadline_seconds": METRICS_POLL_DEADLINE_SECONDS,
            "generation": {
                "metric": GENERATION_METRIC,
                "before_total": generation_before,
                "before_samples": generation_before_samples,
                "after_total": generation_after,
                "after_samples": generation_after_samples,
                "delta": generation_delta,
            },
            "prompt": {
                "metric": PROMPT_METRIC,
                "before_total": prompt_before,
                "before_samples": prompt_before_samples,
                "after_total": prompt_after,
                "after_samples": prompt_after_samples,
                "delta": prompt_delta,
            },
        },
        "metrics_before_raw": metrics_before_raw,
        "metrics_after_raw": metrics_after_raw,
        "passed": not errors,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if errors:
        raise SystemExit("KV capacity stress failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
