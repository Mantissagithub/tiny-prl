"""Request-scoped sampling support capture for vLLM 0.31's V2 sampler."""

from contextvars import ContextVar
from copy import copy
from functools import wraps

import numpy as np
import torch

REPLAY_KEY = "prl_replay"
PAIRED_CAPTURE_KEY = "prl_paired_capture"
_paired: ContextVar[np.ndarray | None] = ContextVar("prl_paired", default=None)
_capture: ContextVar[tuple[np.ndarray, np.ndarray] | None] = ContextVar("prl_capture", default=None)


def wants_replay(params) -> bool:
    return params is not None and (params.extra_args or {}).get(REPLAY_KEY, False) is True


def capture_selected_supports(original, logits, num_sampled_tokens, max_num_kept):
    selection = _capture.get()
    if selection is None:
        return original(logits, num_sampled_tokens, max_num_kept)
    enabled, top_k = selection
    if not enabled.any():
        return None
    keep = torch.as_tensor(enabled, device=num_sampled_tokens.device)
    committed = torch.where(keep, num_sampled_tokens, 0)
    tensors = original(logits, committed, int(top_k[enabled].max()))
    paired = _paired.get()
    if paired is not None and paired.any():
        from prime_rl.inference.vllm.paired_capture import pack_support_logprobs

        tensors = pack_support_logprobs(tensors, logits, paired & enabled)
    return tensors


def patch_request_replay():
    """Backport request-scoped capture using vLLM 0.31's three-argument mask API."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.engine.output_processor import RequestState
    from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    if getattr(Sampler, "_prime_request_replay", False):
        return

    validate = InputProcessor._validate_params

    @wraps(validate)
    def validate_params(self, params, supported_tasks):
        if isinstance(params, SamplingParams):
            if self.model_config.return_sampling_mask and self.speculative_config is not None:
                raise ValueError("Request-scoped sampling replay does not support speculative decoding")
            if wants_replay(params):
                if not self.model_config.return_sampling_mask:
                    raise ValueError("Replay requests require enable_return_sampling_mask")
            else:
                # vLLM 0.31 validates the engine-wide flag before admission. Validate against
                # a private view for eval/untruncated requests without mutating shared config.
                self = copy(self)
                self.model_config = copy(self.model_config)
                self.model_config.return_sampling_mask = False
        return validate(self, params, supported_tasks)

    add_request = Sampler.add_request

    @wraps(add_request)
    def add(self, req_idx, sampling_params):
        if not hasattr(self, "_prime_capture"):
            self._prime_capture = np.zeros(self.sampling_states.max_num_reqs, dtype=bool)
            self._prime_paired = np.zeros(self.sampling_states.max_num_reqs, dtype=bool)
        self._prime_capture[req_idx] = wants_replay(sampling_params)
        self._prime_paired[req_idx] = (sampling_params.extra_args or {}).get(PAIRED_CAPTURE_KEY, False)
        return add_request(self, req_idx, sampling_params)

    call = Sampler.__call__

    @wraps(call)
    def sample(self, logits, input_batch):
        indices = input_batch.idx_mapping_np
        enabled = getattr(self, "_prime_capture", np.zeros(self.sampling_states.max_num_reqs, dtype=bool))
        token = _capture.set((enabled[indices], self.sampling_states.top_k.np[indices]))
        paired = getattr(self, "_prime_paired", None)
        paired_token = _paired.set(paired[indices] if paired is not None else None)
        try:
            return call(self, logits, input_batch)
        finally:
            _capture.reset(token)
            _paired.reset(paired_token)

    from_logits = SamplingMaskTensors.from_logits

    @classmethod
    @wraps(from_logits)
    def capture(cls, logits, num_sampled_tokens, max_num_kept):
        return capture_selected_supports(from_logits, logits, num_sampled_tokens, max_num_kept)

    class DiscardChunks(list):
        def append(self, chunk):
            pass

    from_new_request = RequestState.from_new_request.__func__

    @classmethod
    @wraps(from_new_request)
    def new_request(cls, tokenizer, request, *args, **kwargs):
        state = from_new_request(cls, tokenizer, request, *args, **kwargs)
        if not wants_replay(request.sampling_params):
            state.sampling_mask_chunks = DiscardChunks()
        return state

    InputProcessor._validate_params = validate_params
    Sampler.add_request = add
    Sampler.__call__ = sample
    SamplingMaskTensors.from_logits = capture
    RequestState.from_new_request = new_request
    Sampler._prime_request_replay = True
