import os
from pathlib import Path

root = Path('/home/pradheep/tiny-prl/prime-rl')
work = root / 'outputs/wordle-learning-20261103'
os.environ['PYTHONPATH'] = ':'.join(str(root / p) for p in ['src', 'packages/prime-rl-configs/src', 'deps/prime-kernels', 'outputs/environments/primeintellect-wordle-0.1.7/dependencies'])
os.environ['NLTK_DATA'] = str(root / 'outputs/environments/primeintellect-wordle-0.1.7/nltk_data')
os.environ['TORCH_EXTENSIONS_DIR'] = str(root / 'outputs/kernel-cache')
os.environ['MAX_JOBS'] = '1'
args = ['uv', 'run', '--offline', '--no-project', '--python', '/home/pradheep/.venv/bin/python', 'rl']
for config in ['smollm2-360m-full-stream64', 'smollm2-360m-lossless', 'laptop', 'wordle']:
    args += ['@', str(root / f'examples/tiny/{config}.toml')]
args += ['--no-dashboard', '--max-steps', '2', '--model.name', str(work / 'model-step50'),
         '--run.name', 'warm-wordle-rl-r1', '--output-dir', str(work / 'rl')]
os.execv('/home/pradheep/.local/bin/uv', args)
