import json
from pathlib import Path

work = Path(__file__).resolve().parent
cases = [('Original 360M', 'pass4-base-r1'), ('SFT 50 steps', 'pass4-warm-r1'),
         ('SFT 50 + RL 2 steps', 'pass4-rl-r1')]
results = []
task_keys = None
overlap = json.loads((work / 'rl-eval-answer-overlap.json').read_text())
for label, tag in cases:
    case = json.loads((work / tag / 'pass-at.json').read_text())
    keys = {(e['task_id'], e['seed'], e['task_key'], e['answer']) for e in case['episodes_detail']}
    if task_keys is None:
        task_keys = keys
    assert keys == task_keys
    for seed in case['seeds']:
        config = json.loads((work / tag / f'seed-{seed}.json').read_text())
        assert config['sampling'] == {'max_tokens': 32, 'temperature': 1.0, 'seed': seed}
        player = config['source'][0]['env']['player']
        assert (player['max_turns'], player['max_output_tokens'], player['max_input_tokens']) == (6,192,480)
    guard = json.loads((work / f'guard-inner-{tag}/result.json').read_text())
    assert guard['returncode'] == 0 and guard['stop_reason'] is None and guard['peak_swap_bytes'] == 0
    unseen = [e for e in case['episodes_detail'] if e['answer'] not in overlap['heldout_word_overlap']]
    unseen_tasks = {e['task_id'] for e in unseen}
    unseen_solved = {e['task_id'] for e in unseen if e['win']}
    subset = {'tasks': len(unseen_tasks), 'episodes': len(unseen),
              'pass_at_1': sum(e['win'] for e in unseen) / len(unseen),
              'pass_at_4': len(unseen_solved) / len(unseen_tasks)}
    results.append({'model': label, 'unseen_answer_subset': subset,
                    **{k: v for k, v in case.items() if k != 'episodes_detail'}})
report = {'protocol': {'taskset': 'tiny_prl_wordle', 'hub': 'primeintellect/wordle@0.1.7',
    'split': 'eval', 'task_ids': list(range(2000,2020)), 'seeds': [42,43,44,45],
    'temperature': 1.0, 'max_tokens_per_turn': 32, 'max_turns': 6,
    'max_output_tokens': 192, 'max_input_tokens': 480,
    'unchanged_prompt_and_rewards': True, 'matched_task_keys_and_budgets': True,
    'pass_at_1': 'mean per-task c/4', 'pass_at_4': 'fraction of tasks with at least one win in four attempts'},
    'rl_train_eval_answer_overlap': overlap['heldout_word_overlap'], 'results': results}
(work / 'pass-at-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
lines = ['# Wordle baseline and checkpoint evaluation', '',
    '20 eval-split tasks, four different seeds per word, 80 game attempts per checkpoint.',
    'Same Hub prompt and rewards, temperature 1, six-turn limit, 32 tokens per turn, 480 input tokens.', '',
    '| Checkpoint | pass@1 | pass@4 | Wins / games | Words solved / 20 | Formatted first guess | Mean reward |',
    '|---|---:|---:|---:|---:|---:|---:|']
for r in results:
    lines.append(f"| {r['model']} | {r['pass_at_1']:.1%} | {r['pass_at_4']:.1%} | {r['wins']}/80 | {r['tasks_solved_within_4']}/20 | {r['first_guess_formatted']}/80 | {r['mean_reward']:.5f} |")
lines += ['', 'pass@1 estimates the single-attempt win probability using all four samples per word. pass@4 counts a word as solved if any of its four attempts wins.',
    'All 240 episodes are retained, including invalid moves and generation or episode truncations. No evaluation errors or timeouts were accepted.',
    'These measurements establish the baseline and current checkpoint scores. Twenty words are a small evaluation; these scores do not predict eventual convergence.']
lines += ['', f"The original Hub task sampler uses replacement. RL rollouts exposed these eval answer words: {overlap['heldout_word_overlap']}. SFT source games with any eval answer were excluded before training.",
          'The strict subset excludes those answer words from every checkpoint comparison.']
for r in results:
    s = r['unseen_answer_subset']
    lines.append(f"- {r['model']}: {s['tasks']} unseen-answer words, {s['episodes']} attempts, pass@1 {s['pass_at_1']:.1%}, pass@4 {s['pass_at_4']:.1%}.")
(work / 'pass-at-comparison.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
