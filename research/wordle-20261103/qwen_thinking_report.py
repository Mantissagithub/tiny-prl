import json
import subprocess
from pathlib import Path

work = Path(__file__).resolve().parent
results = []
reference = None
for thinking, tag in [(False, 'pass4-qwen-off256-r1'), (True, 'pass4-qwen-think256-r1')]:
    case = work / tag
    score = json.loads((case / 'pass-at.json').read_text())
    keys = {(e['task_id'], e['task_key'], e['answer'], e['seed']) for e in score['episodes_detail']}
    if reference is None:
        reference = keys
    assert keys == reference
    traces = []
    for seed in [42, 43, 44, 45]:
        config = json.loads((case / f'seed-{seed}.json').read_text())
        assert config['sampling'] == {'max_tokens': 256, 'temperature': 1.0, 'seed': seed,
                                     'chat_template_kwargs': {'enable_thinking': thinking}}
        player = config['source'][0]['env']['player']
        assert (player['max_turns'], player['max_output_tokens'], player['max_input_tokens']) == (6, 1536, 1792)
        for file in (case / f'evaluation/seed-{seed}/monitors/file/traces/stream').glob('*.zst'):
            traces.extend(json.loads(line)['traces'][0] for line in subprocess.check_output(['zstd', '-dc', str(file)]).splitlines())
    assert len(traces) == 80
    first = [next(n['message'] for n in t['nodes'] if n['message']['role'] == 'assistant') for t in traces]
    guard = json.loads((work / f'guard-inner-{tag}/result.json').read_text())
    assert guard['returncode'] == 0 and guard['stop_reason'] is None
    results.append({'thinking': thinking, 'tag': tag, 'games': 80,
        'pass_at_1': score['pass_at_1'], 'pass_at_4': score['pass_at_4'],
        'wins': score['wins'], 'calls': score['total_calls'], 'truncated_calls': score['total_truncated_calls'],
        'output_tokens': sum(t['num_output_tokens'] for t in traces),
        'reasoning_only_first_replies': sum(bool(m.get('reasoning_content')) and not m.get('content') for m in first),
        'first_moves_with_feedback': sum(any(n['message']['role'] == 'user' and 'guesses left' in (n['message'].get('content') or '') for n in t['nodes'][3:4]) for t in traces),
        'peak_gpu_mib': guard['peak_gpu_mib'], 'job_swap_bytes': guard['peak_swap_bytes']})
report = {'model': 'Qwen/Qwen3-0.6B', 'matched_task_keys_and_seeds': True,
          'max_tokens_per_turn': 256, 'max_model_len': 2048, 'max_turns': 6,
          'temperature': 1.0, 'hub_prompt_and_rewards_unchanged': True, 'results': results}
(work / 'qwen-thinking-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
