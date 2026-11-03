import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

work = Path(__file__).resolve().parent
tag = sys.argv[1]
case = work / tag
episodes = []
for file in (case / 'evaluation/wordle/monitors/file/traces/stream').glob('*.zst'):
    for line in subprocess.check_output(['zstd', '-dc', str(file)]).splitlines():
        row = json.loads(line)
        trace = row['traces'][0]
        assert row['ok'] and trace['ok'] and not trace['is_timeout']
        assistants = [n['message'] for n in trace['nodes'] if n['message']['role'] == 'assistant']
        episodes.append({'task_id': row['task']['data']['id'],
            'engine_win': 'guessed the word correctly' in trace['info'].get('game_outcome', ''),
            'correct_answer_reward': trace['rewards']['correct_answer']['score'],
            'outcome': trace['info'].get('game_outcome', trace['stop_condition']),
            'first_final_answer': assistants[0].get('content') or '',
            'final_answers': [m.get('content') or '' for m in assistants],
            'first_reasoning_tail': (assistants[0].get('reasoning_content') or '')[-500:],
            'first_move_feedback': any(n['message']['role'] == 'user' and 'guesses left' in (n['message'].get('content') or '') for n in trace['nodes'][3:4]),
            'calls': len(trace['calls']), 'output_tokens': trace['num_output_tokens'],
            'truncated_calls': sum(c['finish_reason'] == 'length' for c in trace['calls'])})
assert len(episodes) == 4
result = {'tag': tag, 'games': len(episodes), 'engine_wins': sum(e['engine_win'] for e in episodes),
    'first_final_answers': sum(bool(e['first_final_answer']) for e in episodes),
    'first_moves_with_feedback': sum(e['first_move_feedback'] for e in episodes),
    'total_calls': sum(e['calls'] for e in episodes),
    'truncated_calls': sum(e['truncated_calls'] for e in episodes),
    'output_tokens': sum(e['output_tokens'] for e in episodes),
    'outcomes': Counter(e['outcome'] for e in episodes), 'episodes_detail': episodes}
(case / 'pilot-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
