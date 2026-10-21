import copy
from collections.abc import Sequence

import torch
from torch.distributed.tensor import DTensor
from torch.optim import AdamW, Optimizer
from torch.optim.adamw import adamw as functional_adamw

from prime_rl.trainer.optim.base import OffloadOptimizer

STATE_CHUNK_TARGET_NUMEL = 16 * 1024**2


def _step_parameter_in_slices(
    param: torch.Tensor,
    group: dict,
    state: dict[str, torch.Tensor],
) -> None:
    local_param = param.to_local() if isinstance(param, DTensor) else param
    grad = param.grad
    local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    max_exp_avg_sq = state.get("max_exp_avg_sq")
    local_exp_avg = exp_avg.to_local() if isinstance(exp_avg, DTensor) else exp_avg
    local_exp_avg_sq = exp_avg_sq.to_local() if isinstance(exp_avg_sq, DTensor) else exp_avg_sq
    local_max_exp_avg_sq = max_exp_avg_sq.to_local() if isinstance(max_exp_avg_sq, DTensor) else max_exp_avg_sq
    tracked_wrappers = []
    for tensor in (param, exp_avg, exp_avg_sq, max_exp_avg_sq):
        if isinstance(tensor, DTensor):
            tracked_wrappers.append((tensor, tensor._version))

    param_flat = local_param.view(-1)
    grad_flat = local_grad.view(-1)
    avg_flat = local_exp_avg.view(-1)
    avg_sq_flat = local_exp_avg_sq.view(-1)
    max_avg_sq_flat = local_max_exp_avg_sq.view(-1) if local_max_exp_avg_sq is not None else None
    foreach = group["foreach"]
    if foreach is None and isinstance(param, DTensor):
        foreach = False

    for start in range(0, param_flat.numel(), STATE_CHUNK_TARGET_NUMEL):
        end = min(start + STATE_CHUNK_TARGET_NUMEL, param_flat.numel())
        avg_cpu_slice = avg_flat[start:end]
        avg_sq_cpu_slice = avg_sq_flat[start:end]
        avg_slice = avg_cpu_slice.to(device=local_param.device)
        avg_sq_slice = avg_sq_cpu_slice.to(device=local_param.device)
        max_avg_sq_cpu_slice = max_avg_sq_flat[start:end] if max_avg_sq_flat is not None else None
        max_avg_sq_slice = (
            max_avg_sq_cpu_slice.to(device=local_param.device) if max_avg_sq_cpu_slice is not None else None
        )
        functional_adamw(
            params=[param_flat[start:end]],
            grads=[grad_flat[start:end]],
            exp_avgs=[avg_slice],
            exp_avg_sqs=[avg_sq_slice],
            max_exp_avg_sqs=[max_avg_sq_slice] if max_avg_sq_slice is not None else [],
            state_steps=[state["step"].detach().clone()],
            foreach=foreach,
            capturable=group["capturable"],
            differentiable=group["differentiable"],
            fused=group["fused"],
            has_complex=torch.is_complex(local_param),
            amsgrad=group["amsgrad"],
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
        )
        avg_cpu_slice.copy_(avg_slice)
        avg_sq_cpu_slice.copy_(avg_sq_slice)
        if max_avg_sq_cpu_slice is not None:
            max_avg_sq_cpu_slice.copy_(max_avg_sq_slice)
        del avg_slice, avg_sq_slice, max_avg_sq_slice

    state["step"].add_(1)
    for tensor, before in tracked_wrappers:
        if tensor._version == before:
            torch.autograd.graph.increment_version(tensor)


def _can_step_parameter_in_slices(param: torch.Tensor, state: dict[str, torch.Tensor]) -> bool:
    grad = param.grad
    if grad is None:
        return False
    local_param = param.to_local() if isinstance(param, DTensor) else param
    local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
    if (
        local_param.numel() <= STATE_CHUNK_TARGET_NUMEL
        or local_param.device.type != "cuda"
        or local_grad.device != local_param.device
        or local_param.layout != torch.strided
        or local_grad.layout != torch.strided
        or not local_param.is_contiguous()
        or not local_grad.is_contiguous()
        or state["step"].device.type != "cpu"
    ):
        return False
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if value is None:
            continue
        local_value = value.to_local() if isinstance(value, DTensor) else value
        if local_value.device.type != "cpu" or local_value.layout != torch.strided or not local_value.is_contiguous():
            return False
    return True


def _can_step_parameter_with_cute(
    param: torch.Tensor,
    group: dict,
    state: dict[str, torch.Tensor],
) -> bool:
    if not _can_step_parameter_in_slices(param, state):
        return False
    if (
        group["amsgrad"]
        or group["maximize"]
        or group["capturable"]
        or group["differentiable"]
        or group["fused"]
        or state.get("max_exp_avg_sq") is not None
    ):
        return False
    if group["foreach"] is not False and not (group["foreach"] is None and isinstance(param, DTensor)):
        return False
    if any(isinstance(group[key], torch.Tensor) for key in ("lr", "eps", "weight_decay")):
        return False
    if any(isinstance(beta, torch.Tensor) for beta in group["betas"]):
        return False

    grad = param.grad
    local_param = param.to_local() if isinstance(param, DTensor) else param
    local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
    moments = [state["exp_avg"], state["exp_avg_sq"]]
    local_moments = [value.to_local() if isinstance(value, DTensor) else value for value in moments]
    tensors = [local_param, local_grad, *local_moments]
    return (
        state["step"].dtype == torch.float32
        and state["step"].numel() == 1
        and all(tensor.shape == local_param.shape for tensor in tensors[1:])
        and all(
            tensor.dtype == torch.float32 and tensor.layout == torch.strided and tensor.is_contiguous()
            for tensor in tensors
        )
        and local_param.data_ptr() % 16 == 0
        and local_grad.data_ptr() % 16 == 0
        and all(tensor.device.type == "cpu" for tensor in local_moments)
    )


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
        cute_adamw_plan_type: type | None = None,
    ):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self.chunked = chunked
        self.gpu_state_budget_bytes = gpu_state_budget_bytes
        self.parameter_names = parameter_names or {}
        self._cute_adamw_plan_type = cute_adamw_plan_type
        self._cute_adamw_plans = {}
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
        chunks: list[list[torch.Tensor]] = []
        chunk: list[torch.Tensor] = []
        chunk_numel = 0
        seen: set[int] = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if id(param) in seen:
                    continue
                seen.add(id(param))
                if chunk and chunk_numel + param.numel() > STATE_CHUNK_TARGET_NUMEL:
                    chunks.append(chunk)
                    chunk = []
                    chunk_numel = 0
                chunk.append(param)
                chunk_numel += param.numel()
                if param.numel() >= STATE_CHUNK_TARGET_NUMEL:
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
            self._initialize_cpu_state(active_chunk)
            if len(active_chunk) == 1:
                param = active_chunk[0]
                state = self.optimizer.state[param]
                group = self._param_groups_by_id[id(param)]
                if self._cute_adamw_plan_type is not None and _can_step_parameter_with_cute(param, group, state):
                    with torch.no_grad():
                        self._step_parameter_in_cute_slices(param, group, state)
                    continue
                if not group["differentiable"] and _can_step_parameter_in_slices(param, state):
                    with torch.no_grad():
                        _step_parameter_in_slices(param, group, state)
                    continue
            chunk_ids = {id(param) for param in active_chunk}
            hidden_grads = [(param, param.grad) for param in all_params if id(param) not in chunk_ids]
            try:
                for param, _ in hidden_grads:
                    param.grad = None
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

    def _step_parameter_in_cute_slices(
        self,
        param: torch.Tensor,
        group: dict,
        state: dict[str, torch.Tensor],
    ) -> None:
        local_param = self._local_param(param)
        grad = param.grad
        local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
        avg = state["exp_avg"]
        avg_sq = state["exp_avg_sq"]
        local_avg = self._local_param(avg)
        local_avg_sq = self._local_param(avg_sq)
        tracked = [(tensor, tensor._version) for tensor in (param, avg, avg_sq) if isinstance(tensor, DTensor)]
        param_flat, grad_flat = local_param.view(-1), local_grad.view(-1)
        avg_flat, avg_sq_flat = local_avg.view(-1), local_avg_sq.view(-1)
        for start in range(0, param_flat.numel(), STATE_CHUNK_TARGET_NUMEL):
            end = min(start + STATE_CHUNK_TARGET_NUMEL, param_flat.numel())
            param_slice, grad_slice = param_flat[start:end], grad_flat[start:end]
            avg_cpu_slice, avg_sq_cpu_slice = avg_flat[start:end], avg_sq_flat[start:end]
            avg_slice = avg_cpu_slice.to(device=local_param.device)
            avg_sq_slice = avg_sq_cpu_slice.to(device=local_param.device)
            key = (end - start, local_param.device.index)
            plan = self._cute_adamw_plans.get(key)
            if plan is None:
                plan = self._cute_adamw_plan_type(param_slice)
                self._cute_adamw_plans[key] = plan
            plan.step_(
                param_slice,
                grad_slice,
                avg_slice,
                avg_sq_slice,
                state["step"],
                lr=group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
            )
            avg_cpu_slice.copy_(avg_slice)
            avg_sq_cpu_slice.copy_(avg_sq_slice)
            del avg_slice, avg_sq_slice
        state["step"].add_(1)
        for tensor, version in tracked:
            if tensor._version == version:
                torch.autograd.graph.increment_version(tensor)

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
