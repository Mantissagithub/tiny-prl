"""Profile the sequence-chunked logprob head with a real tied embedding tensor.

Usage: uv run --no-sync python tools/bench_tiny_lm_head.py OUTPUT.json WEIGHTS.safetensors KEY [--vocab-tiles]
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open

from prime_rl.trainer.models.layers.lm_head import _SequenceChunkedLogProbEntropyFn

N_VALUES = (286, 298, 512)
HIDDEN_SIZE = 1024
VOCAB_SIZE = 151_936
CHUNK_SIZES = (128, 256, 512)
WARMUP = 2
ITERATIONS = 7


def run_once(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    inv_temperature: torch.Tensor,
    upstream: torch.Tensor,
    chunk_size: int,
    function: type[torch.autograd.Function] = _SequenceChunkedLogProbEntropyFn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_for_grad = hidden.detach().requires_grad_(True)
    logprobs, entropy = function.apply(hidden_for_grad, weight, labels, inv_temperature, chunk_size, None)
    logprobs.backward(upstream)
    assert hidden_for_grad.grad is not None
    return logprobs.detach(), entropy.detach(), hidden_for_grad.grad.detach().clone()


def time_case(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    inv_temperature: torch.Tensor,
    upstream: torch.Tensor,
    function: type[torch.autograd.Function],
) -> dict[str, float | int]:
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        warmup_result = run_once(hidden, weight, labels, inv_temperature, upstream, 128, function)
        del warmup_result
    torch.cuda.synchronize()

    wall_ms = []
    gpu_ms = []
    for _ in range(ITERATIONS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_event.record()
        result = run_once(hidden, weight, labels, inv_temperature, upstream, 128, function)
        end_event.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000)
        gpu_ms.append(start_event.elapsed_time(end_event))
        del result

    peak_allocated = torch.cuda.max_memory_allocated()
    return {
        "median_wall_ms": statistics.median(wall_ms),
        "median_gpu_ms": statistics.median(gpu_ms),
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "peak_delta_bytes": peak_allocated - baseline_allocated,
    }


def load_vocab_tile_candidate(output_dir: Path) -> tuple[type[torch.autograd.Function], dict[str, object]]:
    source_path = Path(inspect.getfile(_SequenceChunkedLogProbEntropyFn)).resolve()
    source = source_path.read_text()
    old = "vocab_chunk_size = min(vocab, 8192)"
    replacement_count = source.count(old)
    if replacement_count != 2:
        raise RuntimeError(f"Expected exactly two vocabulary-tile constants, found {replacement_count}")

    candidate_source = source.replace(old, "vocab_chunk_size = min(vocab, 16384)")
    candidate_path = output_dir / "lm_head_vocab_tile_16384.py"
    candidate_path.write_text(candidate_source)
    spec = importlib.util.spec_from_file_location("_tiny_lm_head_vocab_tile_candidate", candidate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load candidate module from {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = {
        "baseline_source": str(source_path),
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "candidate_source": str(candidate_path),
        "candidate_source_sha256": hashlib.sha256(candidate_source.encode()).hexdigest(),
        "replacement": {"from": old, "to": "vocab_chunk_size = min(vocab, 16384)", "count": replacement_count},
    }
    return module._SequenceChunkedLogProbEntropyFn, metadata


def run_vocab_tile_mode(output_path: Path, weights_file: Path, weight_key: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_function, source_metadata = load_vocab_tile_candidate(output_path.parent)
    device = torch.device("cuda")
    with safe_open(weights_file, framework="pt", device="cpu") as weights_file_handle:
        weight_cpu = weights_file_handle.get_tensor(weight_key)
    if weight_cpu.shape != (VOCAB_SIZE, HIDDEN_SIZE) or weight_cpu.dtype != torch.bfloat16:
        raise ValueError(
            f"Expected bf16 weight [{VOCAB_SIZE}, {HIDDEN_SIZE}], got {weight_cpu.dtype} {tuple(weight_cpu.shape)}"
        )
    weight = weight_cpu.to(device)
    del weight_cpu
    weight.requires_grad_(False)

    generator = torch.Generator(device="cpu").manual_seed(42)
    cases = {}
    measurements = []
    parity_passed = True
    for n in N_VALUES:
        hidden = torch.randn((n, HIDDEN_SIZE), generator=generator).to(torch.bfloat16).to(device)
        labels = torch.randint(VOCAB_SIZE, (n,), generator=generator, dtype=torch.long).to(device)
        inv_temperature = torch.ones((n,), dtype=torch.bfloat16, device=device)
        upstream = (torch.rand((n,), generator=generator, dtype=torch.float32) / n).to(device)
        cases[n] = hidden, labels, inv_temperature, upstream

        reference = run_once(hidden, weight, labels, inv_temperature, upstream, 128)
        candidate = run_once(hidden, weight, labels, inv_temperature, upstream, 128, candidate_function)
        logprobs_error = (candidate[0] - reference[0]).abs().max().item()
        entropy_error = (candidate[1] - reference[1]).abs().max().item()
        grad_error = (candidate[2].float() - reference[2].float()).abs().max().item()
        logprobs_ok = torch.allclose(candidate[0], reference[0], rtol=1e-4, atol=1e-4)
        entropy_ok = torch.allclose(candidate[1], reference[1], rtol=1e-4, atol=1e-4)
        grad_ok = torch.allclose(candidate[2], reference[2], rtol=1e-2, atol=1e-4)
        parity_passed &= logprobs_ok and entropy_ok and grad_ok
        measurements.append(
            {
                "n": n,
                "token_chunk_size": 128,
                "baseline_vocab_tile_size": 8192,
                "candidate_vocab_tile_size": 16384,
                "max_abs_logprobs_error": logprobs_error,
                "max_abs_entropy_error": entropy_error,
                "max_abs_hidden_grad_error": grad_error,
                "logprobs_parity": logprobs_ok,
                "entropy_parity": entropy_ok,
                "hidden_grad_parity": grad_ok,
            }
        )
        del reference, candidate
        torch.cuda.synchronize()

    timing = []
    if parity_passed:
        for measurement in measurements:
            n = measurement["n"]
            hidden, labels, inv_temperature, upstream = cases[n]
            baseline_time = time_case(
                hidden, weight, labels, inv_temperature, upstream, _SequenceChunkedLogProbEntropyFn
            )
            candidate_time = time_case(hidden, weight, labels, inv_temperature, upstream, candidate_function)
            timing.append(
                {"n": n, "token_chunk_size": 128, "baseline_8192": baseline_time, "candidate_16384": candidate_time}
            )

    result = {
        "mode": "vocab_tiles",
        "weights_file": str(weights_file),
        "weight_key": weight_key,
        "weight_shape": list(weight.shape),
        "weight_dtype": str(weight.dtype),
        "hidden_dtype": "torch.bfloat16",
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "warmup_iterations": WARMUP,
        "timed_iterations": ITERATIONS,
        "parity_tolerances": {
            "logprobs": {"rtol": 1e-4, "atol": 1e-4},
            "entropy": {"rtol": 1e-4, "atol": 1e-4},
            "hidden_grad": {"rtol": 1e-2, "atol": 1e-4},
        },
        "source": source_metadata,
        "parity_passed": parity_passed,
        "timing_skipped_due_to_parity_failure": not parity_passed,
        "measurements": measurements,
        "timing": timing,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    if not parity_passed:
        raise SystemExit(f"Numerical parity failed; candidate timings skipped; details saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("weights_file", type=Path)
    parser.add_argument("weight_key")
    parser.add_argument(
        "--vocab-tiles",
        action="store_true",
        help="compare vocabulary tiles 8192 and 16384 at fixed token chunk 128",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if args.vocab_tiles:
        run_vocab_tile_mode(args.output_json, args.weights_file, args.weight_key)
        return
    device = torch.device("cuda")
    with safe_open(args.weights_file, framework="pt", device="cpu") as weights_file:
        weight_cpu = weights_file.get_tensor(args.weight_key)
    if weight_cpu.shape != (VOCAB_SIZE, HIDDEN_SIZE) or weight_cpu.dtype != torch.bfloat16:
        raise ValueError(
            f"Expected bf16 weight [{VOCAB_SIZE}, {HIDDEN_SIZE}], got {weight_cpu.dtype} {tuple(weight_cpu.shape)}"
        )
    weight = weight_cpu.to(device)
    del weight_cpu
    weight.requires_grad_(False)

    generator = torch.Generator(device="cpu").manual_seed(42)
    references: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    cases: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for n in N_VALUES:
        hidden = torch.randn((n, HIDDEN_SIZE), generator=generator).to(torch.bfloat16).to(device)
        labels = torch.randint(VOCAB_SIZE, (n,), generator=generator, dtype=torch.long).to(device)
        inv_temperature = torch.ones((n,), dtype=torch.bfloat16, device=device)
        upstream = (torch.rand((n,), generator=generator, dtype=torch.float32) / n).to(device)
        cases[n] = hidden, labels, inv_temperature, upstream
        references[n] = run_once(hidden, weight, labels, inv_temperature, upstream, CHUNK_SIZES[0])
        torch.cuda.synchronize()

    measurements = []
    parity_ok = True
    for case_index, n in enumerate(N_VALUES):
        hidden, labels, inv_temperature, upstream = cases[n]
        reference_logprobs, reference_entropy, reference_grad = references[n]
        chunk_order = CHUNK_SIZES[case_index:] + CHUNK_SIZES[:case_index]
        for chunk_size in chunk_order:
            measured_logprobs, measured_entropy, measured_grad = run_once(
                hidden, weight, labels, inv_temperature, upstream, chunk_size
            )
            logprobs_error = (measured_logprobs - reference_logprobs).abs().max().item()
            entropy_error = (measured_entropy - reference_entropy).abs().max().item()
            grad_error = (measured_grad.float() - reference_grad.float()).abs().max().item()
            logprobs_ok = torch.allclose(measured_logprobs, reference_logprobs, rtol=1e-4, atol=1e-4)
            entropy_ok = torch.allclose(measured_entropy, reference_entropy, rtol=1e-4, atol=1e-4)
            grad_ok = torch.allclose(measured_grad, reference_grad, rtol=1e-2, atol=1e-4)
            parity_ok &= logprobs_ok and entropy_ok and grad_ok
            del measured_logprobs, measured_entropy, measured_grad

            torch.cuda.synchronize()
            baseline_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(WARMUP):
                warmup_result = run_once(hidden, weight, labels, inv_temperature, upstream, chunk_size)
                del warmup_result
            torch.cuda.synchronize()

            wall_ms = []
            gpu_ms = []
            for _ in range(ITERATIONS):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                wall_start = time.perf_counter()
                start_event.record()
                result = run_once(hidden, weight, labels, inv_temperature, upstream, chunk_size)
                end_event.record()
                torch.cuda.synchronize()
                wall_ms.append((time.perf_counter() - wall_start) * 1000)
                gpu_ms.append(start_event.elapsed_time(end_event))
                del result

            peak_allocated = torch.cuda.max_memory_allocated()
            measurements.append(
                {
                    "n": n,
                    "chunk_size": chunk_size,
                    "median_wall_ms": statistics.median(wall_ms),
                    "median_gpu_ms": statistics.median(gpu_ms),
                    "baseline_allocated_bytes": baseline_allocated,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_delta_bytes": peak_allocated - baseline_allocated,
                    "max_abs_logprobs_error": logprobs_error,
                    "max_abs_entropy_error": entropy_error,
                    "max_abs_hidden_grad_error": grad_error,
                    "logprobs_parity": logprobs_ok,
                    "entropy_parity": entropy_ok,
                    "hidden_grad_parity": grad_ok,
                }
            )

    result = {
        "weights_file": str(args.weights_file),
        "weight_key": args.weight_key,
        "weight_shape": list(weight.shape),
        "weight_dtype": str(weight.dtype),
        "hidden_dtype": "torch.bfloat16",
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "warmup_iterations": WARMUP,
        "timed_iterations": ITERATIONS,
        "parity": parity_ok,
        "measurements": measurements,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if not parity_ok:
        raise SystemExit(f"Numerical parity failed; details saved to {args.output_json}")


if __name__ == "__main__":
    main()
