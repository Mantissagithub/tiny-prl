import os
import sys
from pathlib import Path

root = Path('/home/pradheep/tiny-prl/prime-rl')
work = root / 'outputs/wordle-learning-20261103'
os.environ['PYTHONPATH'] = ':'.join(str(root / p) for p in ['src', 'packages/prime-rl-configs/src', 'deps/prime-kernels'])
os.environ['TORCH_EXTENSIONS_DIR'] = str(root / 'outputs/kernel-cache')
checkpoint, output = sys.argv[1:] if len(sys.argv) > 1 else [str(work / 'training/concise-360m-r2/checkpoints/step_50'), str(work / 'model-step50')]
os.execv('/home/pradheep/.local/bin/uv', ['uv', 'run', '--offline', '--no-project', '--python', '/home/pradheep/.venv/bin/python', 'python', str(root / 'tools/convert_dcp_to_bf16.py'), checkpoint, output])
