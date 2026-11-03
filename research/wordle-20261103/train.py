import os
from pathlib import Path

root = Path('/home/pradheep/tiny-prl/prime-rl')
os.environ['PYTHONPATH'] = ':'.join(str(root / p) for p in ['src', 'packages/prime-rl-configs/src', 'deps/prime-kernels'])
os.environ['TORCH_EXTENSIONS_DIR'] = str(root / 'outputs/kernel-cache')
os.environ['MAX_JOBS'] = '1'
os.execv('/home/pradheep/.local/bin/uv', ['uv', 'run', '--offline', '--no-project', '--python', '/home/pradheep/.venv/bin/python', 'sft', '@', str(root / 'outputs/wordle-learning-20261103/sft.toml')])
