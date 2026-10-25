"""Opt-in fused decode path for SmolLM2-shaped Llama on vLLM's FLASH_ATTN backend.

Pure decode batches of 1 to 4 requests compute the final hidden states with one decode-step
call instead of vLLM's layer stack. lm_head, logits processing and sampling stay in vLLM.
Every other step runs vLLM's stock forward.
"""

import importlib.util
import os
from collections import Counter

import torch
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.llama import LlamaModel

logger = init_logger("vllm.prime_rl.decode_backend")

REFERENCE_ENV = "PRIME_DECODE_REFERENCE"
KERNEL_NAME = "decode_megakernel"
MAX_BATCH = 4

HIDDEN_SIZE = 576
INTERMEDIATE_SIZE = 1536
NUM_LAYERS = 30
NUM_HEADS = 9
NUM_KV_HEADS = 3
HEAD_DIM = 64
RMS_EPS = 1e-5

stats: Counter[str] = Counter()


def _load_reference(path: str):
    spec = importlib.util.spec_from_file_location("prime_rl_decode_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_step


def _resolve_decode_step():
    if os.environ.get(REFERENCE_ENV):
        return _load_reference(os.environ[REFERENCE_ENV])
    import prime_kernels

    if KERNEL_NAME in prime_kernels.KERNELS and prime_kernels.is_available(KERNEL_NAME):
        # Resolved once here; the call site takes the same keyword arguments as the reference.
        return prime_kernels.load(KERNEL_NAME).decode_step
    return None


def _model_eligible(model: LlamaModel) -> bool:
    """Checks that depend only on the loaded model, not on the batch."""
    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if type(model) is not LlamaModel or model.aux_hidden_state_layers:
        return False
    if (
        config.hidden_size != HIDDEN_SIZE
        or config.intermediate_size != INTERMEDIATE_SIZE
        or config.num_hidden_layers != NUM_LAYERS
        or config.num_attention_heads != NUM_HEADS
        or config.num_key_value_heads != NUM_KV_HEADS
        or head_dim != HEAD_DIM
        or config.rms_norm_eps != RMS_EPS
    ):
        return False
    if get_tensor_model_parallel_world_size() != 1 or get_pp_group().world_size != 1:
        return False
    if model.embed_tokens.weight.dtype != torch.bfloat16:
        return False
    return type(model.layers[0].self_attn.qkv_proj) is QKVParallelLinear


def _decode_metadata(model: LlamaModel):
    """FLASH_ATTN metadata when every request in the batch contributes one token, else None."""
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return None
    meta = attn_metadata.get(model.layers[0].self_attn.attn.layer_name)
    if type(meta).__name__ != "FlashAttentionMetadata" or meta.max_query_len != 1:
        return None
    return meta


def _decode_hidden_states(model: LlamaModel, decode_step, input_ids, positions, meta):
    batch = input_ids.shape[0]
    layers = model.layers
    return decode_step(
        input_ids=input_ids,
        positions=positions,
        seq_lens=meta.seq_lens[:batch],
        block_table=meta.block_table[:batch],
        slot_mapping=meta.slot_mapping[:batch],
        embed=model.embed_tokens.weight,
        layers=[
            (
                layer.input_layernorm.weight,
                layer.self_attn.qkv_proj.weight,
                layer.self_attn.o_proj.weight,
                layer.post_attention_layernorm.weight,
                layer.mlp.gate_up_proj.weight,
                layer.mlp.down_proj.weight,
            )
            for layer in layers
        ],
        final_norm=model.norm.weight,
        kv_caches=[layer.self_attn.attn.kv_cache for layer in layers],
        cos_sin_cache=layers[0].self_attn.rotary_emb.cos_sin_cache,
        eps=model.norm.variance_epsilon,
    )


def monkey_patch_decode_backend() -> None:
    """Route pure decode batches of up to MAX_BATCH requests through the decode step.

    Installed only when a decode step is available. Decode steps replayed from CUDA graphs do
    not re-enter Python, so the path each captured graph takes is fixed at capture time.
    """
    if getattr(LlamaModel.forward, "_prime_rl_decode_backend", False):
        return
    decode_step = _resolve_decode_step()
    if decode_step is None:
        return

    original_forward = LlamaModel.forward

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None, **kwargs):
        if inputs_embeds is None and _model_eligible(self):
            meta = _decode_metadata(self)
            batch = input_ids.shape[0]
            if meta is None:
                stats["stock_not_pure_decode"] += 1
            elif 0 < batch <= MAX_BATCH and meta.num_actual_tokens == batch:
                stats["backend_decode"] += 1
                return _decode_hidden_states(self, decode_step, input_ids, positions, meta)
            else:
                stats["stock_decode_over_max_batch"] += 1
        else:
            stats["stock_ineligible"] += 1
        return original_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    forward._prime_rl_decode_backend = True
    LlamaModel.forward = forward
    logger.info("Decode backend installed for SmolLM2-shaped Llama")
