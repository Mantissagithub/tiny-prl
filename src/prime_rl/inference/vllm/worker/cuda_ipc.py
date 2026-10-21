import time
from multiprocessing.connection import Client
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing

from prime_rl.inference.vllm.worker.weight_transfer import load_weights_checkpoint_layerwise

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


def _model(worker):
    model = worker.model_runner.model
    return model.runnable if hasattr(model, "runnable") else model


def _receive(connection, deadline: float):
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not connection.poll(remaining):
        raise TimeoutError("timed out waiting for CUDA IPC trainer")
    return connection.recv()


def _expected_parameter_names(model) -> tuple[set[str], set[str]]:
    from vllm.model_executor.models.llama import LlamaForCausalLM

    if not isinstance(model, LlamaForCausalLM):
        raise RuntimeError("cuda_ipc supports only vLLM LlamaForCausalLM")
    packed = getattr(model, "packed_modules_mapping", None)
    expected_packing = {
        "qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "gate_up_proj": ("gate_proj", "up_proj"),
    }
    if not isinstance(packed, dict) or any(
        tuple(packed.get(key, ())) != values for key, values in expected_packing.items()
    ):
        raise RuntimeError("vLLM Llama packed-module mapping is unsupported for cuda_ipc")

    parameter_names = {name for name, _ in model.named_parameters()}
    canonical_names: set[str] = set()
    for name in parameter_names:
        components = name.split(".")
        for index, component in enumerate(components):
            if component not in expected_packing:
                continue
            canonical_names.update(
                ".".join(components[:index] + [replacement] + components[index + 1 :])
                for replacement in expected_packing[component]
            )
            break
        else:
            canonical_names.add(name)
    return parameter_names, canonical_names


class CUDAIPCWeightUpdateWorker(Worker):
    def liveness_probe(self) -> None:
        return None

    def update_weights_from_connection(self, endpoint: dict[str, str]) -> None:
        model = _model(self)
        if not model.config.tie_word_embeddings or model.lm_head.weight is not model.model.embed_tokens.weight:
            raise RuntimeError("cuda_ipc requires an identity-verified tied lm_head")
        if next(model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError("cuda_ipc requires a BF16 vLLM model")
        expected_parameters, expected_canonical = _expected_parameter_names(model)
        device = next(model.parameters()).device
        deadline = time.monotonic() + int(endpoint["timeout"])
        imported_ptrs: set[int] = set()
        received: set[str] = set()

        def weights(connection):
            count = 0
            while True:
                message = _receive(connection, deadline)
                if isinstance(message, tuple) and len(message) == 2 and message[0] == "finish":
                    if message[1] != count:
                        raise RuntimeError("CUDA IPC tensor count mismatch")
                    break
                index, name, tensor = message
                if index != count or name in received or name == "lm_head.weight":
                    raise ValueError("invalid CUDA IPC tensor sequence")
                if tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16:
                    raise ValueError("CUDA IPC payload must be CUDA BF16")
                if tensor.numel() * tensor.element_size() > 64_000_000:
                    raise ValueError("CUDA IPC tensor exceeds the 64MB slab")
                received.add(name)
                imported_ptrs.add(tensor.untyped_storage().data_ptr())
                snapshot = tensor.clone()
                yield name, snapshot
                torch.cuda.synchronize(device)
                del tensor, snapshot, message
                connection.send(("ack", count))
                count += 1
            if not received:
                raise RuntimeError("empty CUDA IPC update")
            return count

        with Client(endpoint["socket"], family="AF_UNIX", authkey=bytes.fromhex(endpoint["authkey"])) as connection:
            loaded_parameters = load_weights_checkpoint_layerwise(
                model,
                weights(connection),
                self.model_runner.model_config,
                self.vllm_config,
            )
            if loaded_parameters is None:
                raise RuntimeError("vLLM did not return the loaded parameter set")
            torch.cuda.synchronize(device)
            model_ptrs = {tensor.untyped_storage().data_ptr() for tensor in (*model.parameters(), *model.buffers())}
            if imported_ptrs & model_ptrs:
                raise RuntimeError("vLLM model retains CUDA IPC staging storage")
            if loaded_parameters != expected_parameters:
                raise RuntimeError(
                    f"CUDA IPC loaded parameter set mismatch: missing={sorted(expected_parameters - loaded_parameters)}, "
                    f"unexpected={sorted(loaded_parameters - expected_parameters)}"
                )
            if received != expected_canonical:
                raise RuntimeError(
                    f"CUDA IPC canonical key mismatch: missing={sorted(expected_canonical - received)}, "
                    f"unexpected={sorted(received - expected_canonical)}"
                )
            connection.send(("complete", len(received)))
