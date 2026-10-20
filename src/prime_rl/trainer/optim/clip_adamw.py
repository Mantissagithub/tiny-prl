from collections.abc import Callable

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import AdamW


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class TritonClipAdamW(AdamW):
    def __init__(self, params, *, lr, weight_decay, betas, plan_type: Callable):
        super().__init__(params, lr=lr, weight_decay=weight_decay, betas=betas, fused=True)
        self._plan_type = plan_type
        self._plan = None
        self._plan_param_ids = None
        self._clip_coef: torch.Tensor | None = None
        for group in self.param_groups:
            for param in group["params"]:
                self._initialize_state(param)

    def _initialize_state(self, param: torch.Tensor) -> dict:
        state = self.state[param]
        if not state:
            local_param = _local_tensor(param)
            state["step"] = torch.zeros((), dtype=torch.float32, device=local_param.device)
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        return state

    def set_clip_coef(self, clip_coef: torch.Tensor) -> None:
        self._clip_coef = clip_coef

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("TritonClipAdamW does not support optimizer closures")
        self._accelerator_graph_capture_health_check()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TritonClipAdamW does not support CUDA graph capture")
        if self._clip_coef is None:
            raise RuntimeError("TritonClipAdamW requires the trainer's global gradient clip coefficient")

        clip_coef = self._clip_coef
        self._clip_coef = None
        for group in self.param_groups:
            params = [param for param in group["params"] if param.grad is not None]
            if not params:
                continue

            grads = [param.grad for param in params]
            if any(grad.is_sparse for grad in grads):
                raise RuntimeError("AdamW does not support sparse gradients")
            state_steps = []
            exp_avgs = []
            exp_avg_sqs = []
            for param in params:
                state = self._initialize_state(param)
                state_steps.append(state["step"])
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

            local_params = [_local_tensor(param) for param in params]
            local_grads = [_local_tensor(grad) for grad in grads]
            local_exp_avgs = [_local_tensor(avg) for avg in exp_avgs]
            local_exp_avg_sqs = [_local_tensor(avg) for avg in exp_avg_sqs]
            param_ids = tuple(id(param) for param in params)
            if self._plan_param_ids != param_ids:
                self._plan = self._plan_type(local_params)
                self._plan_param_ids = param_ids

            self._plan.step_(
                local_params,
                local_grads,
                local_exp_avgs,
                local_exp_avg_sqs,
                state_steps,
                lr=group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                clip_coef=clip_coef,
            )
            for grad in grads:
                if isinstance(grad, DTensor):
                    torch.autograd.graph.increment_version(grad)


def validate_clip_adamw_parameters(named_params: list[tuple[str, nn.Parameter]]) -> None:
    params = [param for _, param in named_params if param.requires_grad]
    if not params:
        raise ValueError("AdamW fused_clip requires trainable parameters")
    local_params = [_local_tensor(param) for param in params]
    first = local_params[0]
    if not first.is_cuda or first.dtype != torch.float32 or not first.is_contiguous():
        raise ValueError("AdamW fused_clip requires contiguous CUDA FP32 local parameters")
    if any(
        param.device != first.device or param.dtype != torch.float32 or not param.is_contiguous()
        for param in local_params
    ):
        raise ValueError("AdamW fused_clip requires contiguous CUDA FP32 local parameters on one device")
