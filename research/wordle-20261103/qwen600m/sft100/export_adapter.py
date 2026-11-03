import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.tensor import DTensor

root = Path('/home/pradheep/tiny-prl/prime-rl')
for directory in ['tools', 'deps/prime-kernels', 'packages/prime-rl-configs/src', 'src']:
    sys.path.insert(0, str(root / directory))
from convert_dcp_to_bf16 import resolve_dcp_dir, resolve_run_configs, setup_single_process_env
from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.lora import get_lora_state, save_lora_config
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.weights import save_state_dict

work = root / 'outputs/qwen-wordle-training-20261103'
step = work / 'training/qwen600m-wordle-sft-r1/checkpoints/step_100'
out = work / 'adapter-step100'
out.mkdir(exist_ok=False)
setup_logger('info')
model_config, _ = resolve_run_configs(step)
setup_single_process_env()
setup_torch_distributed()
resolve_ep(model_config)
model = setup_model(model_config, get_parallel_dims(model_config), loading_from_checkpoint_later=True)
dcp_load(state_dict={'app': AppState(model, [], None, None)}, checkpoint_id=resolve_dcp_dir(step))
state = get_lora_state().adapter_state_dict()
for key, value in state.items():
    if isinstance(value, DTensor):
        value = value.full_tensor()
    state[key] = value.detach().cpu()
    assert torch.isfinite(state[key]).all(), key
assert state and any(t.abs().sum() > 0 for key, t in state.items() if 'lora_B' in key)
save_state_dict(state, out, save_sharded=False, adapter=True)
lora = model_config.lora
save_lora_config(model, out, rank=lora.rank, alpha=lora.alpha, dropout=lora.dropout)
(work / 'adapter-export.json').write_text(json.dumps({'adapter': str(out), 'tensors': len(state), 'finite': True, 'nonzero_lora_B': True}, indent=2) + '\n')
dist.destroy_process_group()
