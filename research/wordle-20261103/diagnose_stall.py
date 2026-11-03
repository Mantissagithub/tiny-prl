import json
import re
from collections import Counter, defaultdict
from pathlib import Path

work = Path(__file__).resolve().parent
run = work / 'rl/wordle-200-r1'
annotations = {}
for file in (run / 'monitors/file/traces/annotations/orch').glob('*.jsonl'):
    for line in file.open():
        row = json.loads(line)
        annotations[row['trace_id']] = row
episodes, groups = [], defaultdict(list)
for file in (run / 'monitors/file/traces/stream').glob('*.jsonl'):
    for line in file.open():
        row = json.loads(line)
        trace = row['traces'][0]
        ann = annotations.get(trace['id'])
        replies = [n['message']['content'] for n in trace['nodes'] if n['message']['role'] == 'assistant']
        guesses = []
        for reply in replies:
            match = re.search(r'<guess>\s*(.*?)\s*</guess>', reply, re.DOTALL)
            guesses.append(match[1].strip() if match else None)
        episode = {'id': trace['id'], 'step': trace['info']['arrival']['step'],
            'reward': sum(r['score'] * r['weight'] for r in trace['rewards'].values()),
            'win': trace['rewards']['correct_answer']['score'] == 1,
            'outcome': trace['info'].get('game_outcome', ''), 'guesses': guesses,
            'effective': ann is not None, 'advantage': ann['info'].get('advantage') if ann else None,
            'tokens': trace['num_output_tokens']}
        episodes.append(episode)
        groups[row['group']['id']].append(episode)

def stats(selected):
    selected_ids = {e['id'] for e in selected}
    pairs = [v for v in groups.values() if len(v) == 2 and all(e['id'] in selected_ids for e in v)]
    tied = [p for p in pairs if abs(p[0]['reward'] - p[1]['reward']) < 1e-6]
    effective = [e for e in selected if e['effective']]
    return {'episodes': len(selected), 'wins': sum(e['win'] for e in selected),
        'groups_of_two': len(pairs), 'tied_reward_groups': len(tied),
        'identical_guess_groups': sum(p[0]['guesses'] == p[1]['guesses'] for p in pairs),
        'effective_episodes': len(effective),
        'effective_advantage_below_1e_6': sum(abs(e['advantage'] or 0) < 1e-6 for e in effective),
        'invalid_move_episodes': sum(e['outcome'].startswith('Invalid Move') for e in selected),
        'top_first_guesses': Counter(e['guesses'][0] for e in selected if e['guesses']).most_common(6),
        'top_later_guesses': Counter(g for e in selected for g in e['guesses'][1:]).most_common(10),
        'bad_length_guesses': sum(not g or re.fullmatch(r'\[[a-zA-Z]{5}\]', g) is None for e in selected for g in e['guesses']),
        'total_guesses': sum(len(e['guesses']) for e in selected)}

metrics = [json.loads(l) for l in (run / 'monitors/file/metrics.jsonl').open()]
grads = [r for r in metrics if 'optim/grad_norm' in r]
losses = [r for r in metrics if 'loss/mean' in r]
result = {'all': stats(episodes), 'first20': stats([e for e in episodes if e['step'] <= 20]),
    'after100': stats([e for e in episodes if e['step'] > 100]),
    'last_update': grads[-1]['step'], 'updates_with_gradient_above_1e_6': sum(r['optim/grad_norm'] > 1e-6 for r in grads),
    'loss_steps_with_masking': sum(r.get('is_masked/max', 0) > 0 for r in losses),
    'small_advantage_examples': [e for e in episodes if e['effective'] and 0 < abs(e['advantage'] or 0) < 1e-6][-4:],
    'last_episode_examples': episodes[-3:]}
(work / 'stall-diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
