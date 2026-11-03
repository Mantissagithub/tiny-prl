import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path('/home/pradheep/tiny-prl/prime-rl')
WORK = ROOT / 'outputs/wordle-learning-20261103'
tag, model = sys.argv[1:3]
pass4 = len(sys.argv) > 3 and sys.argv[3] == 'pass4'
qwen = 'qwen' in sys.argv[4:]
roomy = 'roomy' in sys.argv[4:]
thinking = 'thinking' in sys.argv[4:]
large = 'large' in sys.argv[4:]
xlarge = 'xlarge' in sys.argv[4:]
xxlarge = 'xxlarge' in sys.argv[4:]
case = WORK / tag
case.mkdir(exist_ok=False)
prefix = ['/home/pradheep/.local/bin/uv', 'run', '--offline', '--no-project', '--python', '/home/pradheep/.venv/bin/python']
env = os.environ.copy()
env['PYTHONPATH'] = ':'.join(str(ROOT / p) for p in ['src', 'packages/prime-rl-configs/src', 'deps/prime-kernels', 'outputs/environments/primeintellect-wordle-0.1.7/dependencies'])
env['NLTK_DATA'] = str(ROOT / 'outputs/environments/primeintellect-wordle-0.1.7/nltk_data')
env['TORCH_EXTENSIONS_DIR'] = str(ROOT / 'outputs/kernel-cache')
env['MAX_JOBS'] = '1'
inference = json.loads((ROOT / 'outputs/wordle-hub-20261103/dryrun/wordle-config-r2/configs/attempt_1/resolved/inference.json').read_text())
inference['vllm']['model'] = model
if qwen:
    inference['decode_megakernel'] = False
    inference['env_vars'] = {}
    inference['vllm']['gpu_memory_utilization'] = 0.25
    inference['vllm']['reasoning_parser'] = 'qwen3'
if roomy:
    inference['vllm']['max_model_len'] = 2048
    inference['vllm']['max_num_batched_tokens'] = 2048
if large or xlarge or xxlarge:
    inference['vllm']['max_model_len'] = 4096 if large else 8192
    inference['vllm']['max_num_batched_tokens'] = 1024
    inference['vllm']['max_num_seqs'] = 1
    inference['vllm']['compilation_config']['cudagraph_capture_sizes'] = [1]
    if xlarge or xxlarge:
        inference['vllm']['gpu_memory_utilization'] = 0.30
(case / 'inference.json').write_text(json.dumps(inference, indent=2))
config = {
    'model': model, 'group_size': 2, 'select': {'limit': 8},
    'client': {'base_url': 'http://localhost:8000/v1'},
    'sampling': {'max_tokens': 32, 'temperature': 1.0, 'seed': 42},
    'concurrency': {'min_inflight': 2, 'max_inflight': 2},
    'source': [{'name': 'primeintellect-wordle', 'serve': {'pool': {'type': 'elastic', 'max_workers': 1}}, 'env': {
        'taskset': {'id': 'tiny_prl_wordle', 'split': 'eval', 'seed': 0},
        'player': {'harness': {'id': 'null'}, 'runtime': {'type': 'subprocess', 'fork_server': True},
                   'max_turns': 6, 'max_output_tokens': 192, 'max_input_tokens': 480, 'timeout': {'rollout': 60}}}}],
    'dashboard': False, 'output_dir': str(case / 'evaluation'), 'run': {'name': 'wordle'}, 'monitors': {'file': {}}}
if qwen:
    config['sampling']['chat_template_kwargs'] = {'enable_thinking': thinking}
if roomy:
    config['sampling']['max_tokens'] = 256
    player = config['source'][0]['env']['player']
    player['max_output_tokens'] = 1536
    player['max_input_tokens'] = 1792
if large or xlarge or xxlarge:
    budget = 4096 if xxlarge else 2048 if xlarge else 1024
    config['sampling']['max_tokens'] = budget
    config['group_size'] = 1
    config['select']['limit'] = 4
    config['concurrency'] = {'min_inflight': 1, 'max_inflight': 1}
    player = config['source'][0]['env']['player']
    player['max_output_tokens'] = 6 * budget
    player['max_input_tokens'] = inference['vllm']['max_model_len'] - budget
    player['timeout']['rollout'] = 120
(case / 'eval.json').write_text(json.dumps(config, indent=2))
with (case / 'inference.log').open('w') as log:
    proc = subprocess.Popen(prefix + ['inference', '@', str(case / 'inference.json')], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError('inference exited before readiness')
            try:
                if httpx.get('http://localhost:8000/liveness', timeout=1, trust_env=False).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError('inference startup')
        for seed in ([42, 43, 44, 45] if pass4 else [42]):
            name = f'seed-{seed}' if pass4 else 'wordle'
            if pass4:
                config['group_size'] = 1
                config['select']['limit'] = 20
                config['sampling']['seed'] = seed
                config['run']['name'] = name
            config_path = case / f'{name}.json'
            config_path.write_text(json.dumps(config, indent=2))
            with (case / f'{name}.log').open('w') as log:
                result = subprocess.run(prefix + ['eval', '@', str(config_path)], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=540 if large or xlarge or xxlarge else 240)
            (case / f'{name}-completion.json').write_text(json.dumps({'returncode': result.returncode}))
            if result.returncode:
                raise RuntimeError(f'evaluation failed; see {name}.log')
        (case / 'completion.json').write_text(json.dumps({'returncode': 0, 'seeds': [42, 43, 44, 45] if pass4 else [42]}))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
