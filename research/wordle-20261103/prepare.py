import hashlib
import json
import os
import random
import re
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

ROOT = Path('/home/pradheep/tiny-prl/prime-rl')
WORK = ROOT / 'outputs/wordle-learning-20261103'
os.environ['NLTK_DATA'] = str(ROOT / 'outputs/environments/primeintellect-wordle-0.1.7/nltk_data')
import textarena as ta
from tiny_prl_wordle import SYSTEM_PROMPT

source = json.loads((WORK / 'source.json').read_text())
dataset = load_dataset('parquet', data_files=source['files'], split='train')
game = ta.make(env_id='Wordle-v0')
game.reset(num_players=1)
prompt = str(game.get_observation()[1])
rng = random.Random(0)
answers = [rng.choice(game.word_list) for _ in range(2020)]
heldout = set(answers[2000:])
tokenizer = AutoTokenizer.from_pretrained('/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951', local_files_only=True)
rows, lengths, targets = [], [], []
excluded = invalid = 0
for row in dataset:
    if row['answer'] in heldout:
        excluded += 1
        continue
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
    for message in row['completion']:
        if message['role'] == 'assistant':
            match = re.search(r'<guess>\s*\[([a-zA-Z]{5})\]\s*</guess>', message['content'])
            if not match or not game.dictionary.is_english_word(match[1].lower()):
                invalid += 1
                break
            target = '<guess>[' + match[1].lower() + ']</guess>'
            candidate = messages + [{'role': 'assistant', 'content': target}]
            length = len(tokenizer.apply_chat_template(candidate, tokenize=True, return_dict=False))
            if length > 512:
                break
            # One target per row keeps longer games from losing their final move to packing truncation.
            rows.append({'messages': list(candidate), 'message_loss_mask': [False] * len(messages) + [True]})
            lengths.append(length)
            targets.append(len(tokenizer.encode(target, add_special_tokens=False)))
            messages = candidate
        elif message['role'] == 'user':
            # A terminal observation is never followed by a teacher target.
            messages.append({'role': 'user', 'content': message['content']})
out = WORK / 'concise-data'
out.mkdir(exist_ok=True)
Dataset.from_list(rows).to_parquet(str(out / 'train.parquet'))
manifest = {**source, 'source_rows': len(dataset), 'excluded_heldout_answer_rows': excluded,
    'invalid_teacher_rows': invalid, 'training_rows': len(rows),
    'heldout_task_ids': list(range(2000,2020)), 'heldout_answers': answers[2000:],
    'max_sequence_tokens': max(lengths), 'max_target_tokens': max(targets),
    'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    'initial_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
    'parquet_sha256': hashlib.sha256((out/'train.parquet').read_bytes()).hexdigest()}
(WORK / 'data-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
