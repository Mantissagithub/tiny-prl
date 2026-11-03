import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

work = Path(__file__).resolve().parent
tag = sys.argv[1]
case = work / tag
tasks = defaultdict(dict)
episodes = []
for seed in [42, 43, 44, 45]:
    run = case / 'evaluation' / f'seed-{seed}'
    files = sorted((run / 'monitors/file/traces/stream').glob('*.zst'))
    rows = [json.loads(line) for file in files for line in subprocess.check_output(['zstd', '-dc', str(file)]).splitlines()]
    assert len(rows) == 20, (seed, len(rows))
    for row in rows:
        assert row['ok'] and not row['is_timeout'], row['id']
        trace = row['traces'][0]
        task = row['task']['data']
        assert trace['ok'] and not trace['is_timeout']
        assert trace['calls'] and all(call['sampling']['seed'] == seed for call in trace['calls'])
        assert seed not in tasks[task['id']]
        win = trace['rewards']['correct_answer']['score'] == 1
        outcome = trace['info'].get('game_outcome', '')
        assert win == ('guessed the word correctly' in outcome), outcome
        replies = [node['message']['content'] for node in trace['nodes'] if node['message']['role'] == 'assistant']
        record = {'task_id': task['id'], 'answer': task['answer'], 'task_key': row['task']['key'],
                  'seed': seed, 'win': win, 'outcome': outcome,
                  'first_guess_formatted': bool(re.fullmatch(r'\s*<guess>\s*\[[a-zA-Z]{5}\]\s*</guess>\s*', replies[0])),
                  'reward': sum(r['score'] * r['weight'] for r in trace['rewards'].values()),
                  'truncated_calls': sum(call['finish_reason'] == 'length' for call in trace['calls']),
                  'calls': len(trace['calls'])}
        tasks[task['id']][seed] = record
        episodes.append(record)
assert set(tasks) == {str(i) for i in range(2000, 2020)}
assert all(len(attempts) == 4 for attempts in tasks.values())
assert all(len({e['task_key'] for e in attempts.values()}) == 1 for attempts in tasks.values())
wins = sum(e['win'] for e in episodes)
solved = sum(any(e['win'] for e in attempts.values()) for attempts in tasks.values())
result = {'tag': tag, 'tasks': len(tasks), 'attempts_per_task': 4, 'episodes': len(episodes),
          'seeds': [42, 43, 44, 45], 'wins': wins, 'tasks_solved_within_4': solved,
          'pass_at_1': wins / len(episodes), 'pass_at_4': solved / len(tasks),
          'pass_at_1_definition': 'mean per-task c/4; c is successful complete games',
          'pass_at_4_definition': 'mean per-task indicator(c > 0), estimated from four samples',
          'seed42_single_attempt_win_rate': sum(e['win'] for e in episodes if e['seed'] == 42) / len(tasks),
          'first_guess_formatted': sum(e['first_guess_formatted'] for e in episodes),
          'mean_reward': sum(e['reward'] for e in episodes) / len(episodes),
          'total_truncated_calls': sum(e['truncated_calls'] for e in episodes),
          'total_calls': sum(e['calls'] for e in episodes), 'episodes_detail': episodes}
(case / 'pass-at.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'episodes_detail'}, indent=2))
