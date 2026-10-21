import json
import os
import secrets
import tempfile
import time
from multiprocessing.connection import Listener
from pathlib import Path

import torch
import torch.multiprocessing
import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.configs.shared import BaseWeightBroadcastConfig
from prime_rl.trainer.world import get_world
from prime_rl.transports.weights.base import STARTED_MARKER, WeightReceiver, WeightSender
from prime_rl.utils.weights import _validate_streaming_llama, resolve_fqn, resolve_wire_dtype

MAX_IPC_BYTES = 64_000_000
IPC_META = ".cuda_ipc.json"


def _check_allocator() -> None:
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        allocator = os.environ.get(name, "")
        if "expandable_segments:true" in allocator.lower().replace(" ", ""):
            raise RuntimeError(f"cuda_ipc requires expandable_segments:False; {name} enables it")


def _receive(connection, deadline: float):
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not connection.poll(remaining):
        raise TimeoutError("timed out waiting for CUDA IPC peer")
    return connection.recv()


class CUDAIPCWeightSender(WeightSender):
    def __init__(self, output_dir: Path, config: BaseWeightBroadcastConfig):
        super().__init__(output_dir, config.timeout)
        self.config = config
        _check_allocator()

    def _validate_model(self, model: nn.Module) -> None:
        if get_world().world_size != 1:
            raise ValueError("cuda_ipc currently requires trainer world_size=1")
        candidate = _validate_streaming_llama(model)
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError("cuda_ipc currently requires an FP32 trainer model")
        if (
            not candidate.config.tie_word_embeddings
            or candidate.lm_head.weight is not candidate.model.embed_tokens.weight
        ):
            raise ValueError("cuda_ipc requires an identity-verified tied lm_head")

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        candidate = _validate_streaming_llama(model)
        meta_path = step_dir / IPC_META
        deadline = time.monotonic() + self.timeout
        while not meta_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("receiver did not publish CUDA IPC endpoint metadata")
            time.sleep(0.05)
        metadata = json.loads(meta_path.read_text())
        with Listener(metadata["socket"], family="AF_UNIX", authkey=bytes.fromhex(metadata["authkey"])) as listener:
            listener._listener._socket.settimeout(max(0.001, deadline - time.monotonic()))
            with listener.accept() as connection:
                slab = torch.empty(MAX_IPC_BYTES // 2, dtype=torch.bfloat16, device="cuda")
                keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
                state_dict = model.state_dict()
                tied = (
                    candidate.config.tie_word_embeddings
                    and candidate.lm_head.weight is candidate.model.embed_tokens.weight
                )
                count = 0
                for key, value in state_dict.items():
                    name = resolve_fqn(model, key)
                    if tied and name == "lm_head.weight":
                        continue
                    target_dtype = resolve_wire_dtype(keep_in_fp32, key, torch.bfloat16)
                    if target_dtype != torch.bfloat16:
                        raise ValueError(f"cuda_ipc supports only BF16 wire tensors, got {target_dtype} for {name}")
                    if isinstance(value, DTensor):
                        value = value.to(torch.bfloat16).full_tensor()
                    elif value.dtype != torch.bfloat16:
                        raise ValueError(f"cuda_ipc requires BF16 non-DTensor tensors, got {value.dtype} for {name}")
                    if value.numel() * value.element_size() > MAX_IPC_BYTES:
                        raise ValueError(f"tensor {name} exceeds the {MAX_IPC_BYTES}-byte CUDA IPC slab")
                    view = slab[: value.numel()].view(value.shape)
                    view.copy_(value)
                    torch.cuda.synchronize(slab.device)
                    connection.send((count, name, view))
                    if _receive(connection, deadline) != ("ack", count):
                        raise RuntimeError(f"receiver failed to acknowledge CUDA IPC tensor {count}")
                    count += 1
                    del value, view
                connection.send(("finish", count))
                completion = _receive(connection, deadline)
                if completion != ("complete", count):
                    raise RuntimeError("receiver did not confirm CUDA IPC model reload")
                del slab


class CUDAIPCWeightReceiver(WeightReceiver):
    async def receive(self, step: int) -> None:
        import asyncio

        async with asyncio.timeout(self.config.timeout):
            with tempfile.TemporaryDirectory(prefix="ci-", dir="/tmp") as directory:
                socket_path = str(Path(directory) / "s")
                if len(os.fsencode(socket_path)) >= 108:
                    raise RuntimeError("temporary CUDA IPC socket path exceeds the Unix socket limit")
                authkey = secrets.token_bytes(32)
                metadata = {
                    "socket": socket_path,
                    "authkey": authkey.hex(),
                    "timeout": str(self.config.timeout),
                }
                meta_path = self.step_dir(step) / IPC_META
                fd = os.open(meta_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "w") as stream:
                    stream.write(json.dumps(metadata))
                try:
                    self._ack(step)
                    started = self.step_dir(step) / STARTED_MARKER
                    while not started.exists():
                        await asyncio.sleep(0.05)
                    while not Path(socket_path).exists():
                        await asyncio.sleep(0.05)
                    await self.admin_plane.update_weights(
                        None, transport="cuda_ipc", step=step, on_paused=None, ipc_endpoint=metadata
                    )
                finally:
                    meta_path.unlink(missing_ok=True)
