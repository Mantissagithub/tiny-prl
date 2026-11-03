import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

work = Path(__file__).resolve().parent
tag = sys.argv[1]
rows = []
for file in (work / tag / 'evaluation/wordle/monitors/file/traces/stream').glob('*.zst'):
    rows.extend(json.loads(line) for line in subprocess.check_output(['zstd', '-dc', str(file)]).splitlines())
episodes = []
for row in rows:
    trace = row['traces'][0]
    messages = [n['message']['content'] for n in trace['nodes'] if n['message']['role'] == 'assistant']
    guesses = []
    for message in messages:
        match = re.fullmatch(r'\s*<guess>\s*\[([a-zA-Z]{5})\]\s*</guess>\s*', message)
        guesses.append(match[1].lower() if match else None)
    rewards = trace['rewards']
    episodes.append({'task_id': row['task']['data']['id'],
        'reward': sum(r['score'] * r['weight'] for r in rewards.values()),
        'win': rewards['correct_answer']['score'] == 1,
        'partial': rewards['partial_answer']['score'],
        'format': rewards['format_reward']['score'],
        'guesses': guesses, 'replies': messages, 'outcome': trace['info'].get('game_outcome'),
        'truncated_calls': sum(c['finish_reason'] == 'length' for c in trace['calls']),
        'calls': len(trace['calls'])})
groups = defaultdict(list)
for episode in episodes:
    groups[episode['task_id']].append(episode['reward'])
result = {'episodes': len(episodes), 'unique_tasks': len(groups),
    'wins': sum(e['win'] for e in episodes),
    'first_guess_formatted': sum(bool(e['guesses'] and e['guesses'][0]) for e in episodes),
    'mean_reward': sum(e['reward'] for e in episodes) / len(episodes),
    'mean_partial_reward': sum(e['partial'] for e in episodes) / len(episodes),
    'mean_format_reward': sum(e['format'] for e in episodes) / len(episodes),
    'meaningful_reward_groups': sum(max(v) - min(v) >= 0.001 for v in groups.values()),
    'episodes_detail': episodes}
(work / tag / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'episodes_detail'}, indent=2))
