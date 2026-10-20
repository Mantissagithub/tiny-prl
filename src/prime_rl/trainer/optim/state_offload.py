import copy
from collections.abc import Sequence

import torch
from torch.distributed.tensor import DTensor
from torch.optim import AdamW, Optimizer

from prime_rl.trainer.optim.base import OffloadOptimizer


class CPUOffloadOptimizer(OffloadOptimizer):
    """Wraps an optimizer to keep states on CPU, moving to GPU only for step().

    Unlike FSDP's CPUOffload which offloads weights too, this keeps weights on GPU.
    With activation checkpointing, activations and optimizer states are never on GPU
    at the same time: peak memory becomes max(activations, opt_states) instead of sum.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        pin_memory: bool = True,
        chunked: bool = False,
        gpu_state_budget_bytes: int = 0,
        parameter_names: dict[int, str] | None = None,
    ):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self.chunked = chunked
        self.gpu_state_budget_bytes = gpu_state_budget_bytes
        self.parameter_names = parameter_names or {}
        self._initialized = False
        self._cpu_buffers: dict[tuple[torch.Tensor, str], torch.Tensor] = {}
        self._state_chunks = self._build_state_chunks() if chunked else []
        self._param_groups_by_id = {
            id(param): group for group in optimizer.param_groups for param in group["params"]
        }

        if chunked:
            if not isinstance(optimizer, AdamW):
                raise TypeError("Chunked optimizer-state offload only supports AdamW")
            for group in optimizer.param_groups:
                if group.get("fused") or group.get("capturable"):
                    raise ValueError("Chunked optimizer-state offload requires non-fused, non-capturable AdamW")
        elif gpu_state_budget_bytes:
            raise ValueError("GPU-resident optimizer states require chunked optimizer-state offload")
        if gpu_state_budget_bytes < 0:
            raise ValueError("GPU-resident optimizer-state budget must be nonnegative")
        self._gpu_state_params = self._select_gpu_state_params() if chunked and gpu_state_budget_bytes else set()
        self.gpu_state_bytes = self._selected_gpu_state_bytes()
        self.gpu_state_parameter_names = [
            self.parameter_names.get(id(param), f"<unnamed:{tuple(self._local_param(param).shape)}>")
            for chunk in self._state_chunks
            for param in chunk
            if id(param) in self._gpu_state_params
        ]

    def _select_gpu_state_params(self) -> set[int]:
        selected: set[int] = set()
        used_bytes = 0
        for chunk in self._state_chunks:
            for param in chunk:
                local_param = self._local_param(param)
                if local_param.device.type != "cuda":
                    raise ValueError("GPU-resident optimizer-state staging requires CUDA parameters")
                if local_param.dtype != torch.float32:
                    raise ValueError("GPU-resident optimizer-state staging requires FP32 parameters and moments")
                group = self._param_groups_by_id[id(param)]
                moment_count = 3 if group.get("amsgrad", False) else 2
                state_bytes = local_param.numel() * local_param.element_size() * moment_count
                if used_bytes + state_bytes <= self.gpu_state_budget_bytes:
                    selected.add(id(param))
                    used_bytes += state_bytes
        return selected

    def _selected_gpu_state_bytes(self) -> int:
        total = 0
        for chunk in self._state_chunks:
            for param in chunk:
                if id(param) not in self._gpu_state_params:
                    continue
                local_param = self._local_param(param)
                group = self._param_groups_by_id[id(param)]
                moment_count = 3 if group.get("amsgrad", False) else 2
                total += local_param.numel() * local_param.element_size() * moment_count
        return total

    def _build_state_chunks(self) -> list[list[torch.Tensor]]:
        target_numel = 16 * 1024**2
        chunks: list[list[torch.Tensor]] = []
        chunk: list[torch.Tensor] = []
        chunk_numel = 0
        seen: set[int] = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if id(param) in seen:
                    continue
                seen.add(id(param))
                if chunk and chunk_numel + param.numel() > target_numel:
                    chunks.append(chunk)
                    chunk = []
                    chunk_numel = 0
                chunk.append(param)
                chunk_numel += param.numel()
                if param.numel() >= target_numel:
                    chunks.append(chunk)
                    chunk = []
                    chunk_numel = 0
        if chunk:
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _local_param(param: torch.Tensor) -> torch.Tensor:
        return param.to_local() if isinstance(param, DTensor) else param

    @staticmethod
    def _state_tensor(param: torch.Tensor, local_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(param, DTensor):
            return DTensor(local_tensor, param._spec, requires_grad=False)
        return local_tensor

    def _initialize_cpu_state(self, params: Sequence[torch.Tensor]) -> None:
        groups = {
            id(param): group
            for group in self.optimizer.param_groups
            for param in group["params"]
        }
        for param in params:
            state = self.optimizer.state[param]
            if state:
                continue
            local_param = self._local_param(param)
            step_dtype = torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
            state["step"] = torch.zeros((), dtype=step_dtype, device="cpu")
            state_device = local_param.device if id(param) in self._gpu_state_params else "cpu"
            state["exp_avg"] = self._state_tensor(
                param, torch.zeros_like(local_param, device=state_device, memory_format=torch.preserve_format)
            )
            state["exp_avg_sq"] = self._state_tensor(
                param, torch.zeros_like(local_param, device=state_device, memory_format=torch.preserve_format)
            )
            group = groups[id(param)]
            if group["amsgrad"]:
                state["max_exp_avg_sq"] = self._state_tensor(
                    param, torch.zeros_like(local_param, device=state_device, memory_format=torch.preserve_format)
                )

    def _offload_to_cpu(self, param: torch.Tensor, key: str, src: torch.Tensor) -> torch.Tensor:
        if not self.pin_memory:
            buffer = self._cpu_buffers.get((param, key))
            if buffer is None or buffer.shape != src.shape or buffer.dtype != src.dtype or buffer.is_pinned():
                buffer = torch.empty(src.shape, dtype=src.dtype, device="cpu")
                self._cpu_buffers[(param, key)] = buffer
            buffer.copy_(src, non_blocking=False)
            return buffer
        buffer = self._cpu_buffers.get((param, key))
        if buffer is None or buffer.shape != src.shape or buffer.dtype != src.dtype:
            buffer = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
            self._cpu_buffers[(param, key)] = buffer
        if buffer is not src:
            buffer.copy_(src, non_blocking=True)
        return buffer

    def _move_states(self, device: str, params: Sequence[torch.Tensor] | None = None):
        """Move optimizer states to CPU or back to GPU (matching each parameter's device)."""
        for param in self.optimizer.state if params is None else params:
            if param not in self.optimizer.state:
                continue
            if self.chunked and id(param) in self._gpu_state_params:
                continue
            state = self.optimizer.state[param]
            for key, value in state.items():
                if self.chunked and key == "step" and device == "cuda":
                    continue
                if self.chunked and key == "step" and device == "cpu" and value.device.type == "cpu":
                    continue
                if isinstance(value, DTensor):
                    local_tensor = value._local_tensor
                    if device == "cpu":
                        new_local = self._offload_to_cpu(param, key, local_tensor)
                    else:
                        new_local = local_tensor.to(device, non_blocking=True)
                    new_dtensor = copy.copy(value)
                    new_dtensor._local_tensor = new_local
                    state[key] = new_dtensor
                elif isinstance(value, torch.Tensor):
                    if device == "cpu":
                        state[key] = self._offload_to_cpu(param, key, value)
                    else:
                        state[key] = value.to(device, non_blocking=True)
        if device == "cpu" and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def step(self, closure=None):
        if self.chunked:
            if closure is not None:
                raise ValueError("Closures are not supported with chunked optimizer-state offload")
            return self._step_chunked()

        # First step initializes states on GPU - offload after
        if not self._initialized:
            result = self.optimizer.step(closure)
            self._move_states("cpu")
            self._initialized = True
            return result

        # Move states to GPU
        self._move_states("cuda")

        # Run optimizer step
        result = self.optimizer.step(closure)

        # Move states back to CPU
        self._move_states("cpu")

        return result

    def _step_chunked(self):
        result = None
        all_params = [param for group in self.optimizer.param_groups for param in group["params"]]
        for chunk in self._state_chunks:
            active_chunk = [param for param in chunk if param.grad is not None]
            if not active_chunk:
                continue
            chunk_ids = {id(param) for param in active_chunk}
            hidden_grads = [(param, param.grad) for param in all_params if id(param) not in chunk_ids]
            try:
                for param, _ in hidden_grads:
                    param.grad = None
                self._initialize_cpu_state(active_chunk)
                try:
                    self._move_states("cuda", active_chunk)
                    result = self.optimizer.step()
                finally:
                    self._move_states("cpu", active_chunk)
            finally:
                for param, grad in hidden_grads:
                    param.grad = grad
        self._initialized = True
        return result

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        if self.chunked:
            self.prepare_checkpoint_save()
            return self.optimizer.state_dict()
        # Move to GPU temporarily for consistent state dict
        if self._initialized:
            self._move_states("cuda")
            torch.cuda.synchronize()
        state_dict = self.optimizer.state_dict()
        if self._initialized:
            self._move_states("cpu")
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        self._move_states("cpu")
        self._initialized = True

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    @property
    def state(self):
        return self.optimizer.state

    @property
    def base_optimizer(self) -> Optimizer:
        return self.optimizer

    def checkpoint_optimizer(self) -> Optimizer:
        return self.optimizer

    def prepare_checkpoint_save(self) -> None:
        if self.chunked:
            self._initialize_cpu_state([param for group in self.optimizer.param_groups for param in group["params"]])
            self._initialized = True
            return
        if self._initialized:
            self._move_states("cuda")
            torch.cuda.synchronize()

    def finish_checkpoint_save(self) -> None:
        if not self.chunked:
            self._move_states("cpu")

    def finish_checkpoint_load(self) -> None:
        self._initialized = True
