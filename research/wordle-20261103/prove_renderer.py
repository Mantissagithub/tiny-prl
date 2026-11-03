import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from renderers.base import build_training_sample
from smol_renderer import SmolRenderer, SmolConfig

work = Path(__file__).resolve().parent
tokenizer = AutoTokenizer.from_pretrained('/home/pradheep/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951', local_files_only=True)
renderer = SmolRenderer(tokenizer, SmolConfig())
dataset = load_dataset(str(work / 'concise-data'), split='train')
assert renderer.get_stop_token_ids() == [tokenizer.convert_tokens_to_ids('<|im_end|>')]
counts = []
for row in dataset:
    messages = row['messages']
    rendered = renderer.render(messages)
    assert rendered.token_ids == tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    prompt = messages[:-1]
    assert renderer.render(prompt, add_generation_prompt=True).token_ids == tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
    sample = build_training_sample(renderer, messages, message_loss_mask=row['message_loss_mask'], ensure_final_stop=True)
    trained = [v for v, mask in zip(sample.token_ids, sample.loss_mask) if mask]
    expected = tokenizer.encode(messages[-1]['content'] + '<|im_end|>', add_special_tokens=False)
    assert trained == expected
    counts.append(len(trained))
result = {'rows': len(dataset), 'exact_template_and_target_mask': True,
          'min_loss_tokens': min(counts), 'max_loss_tokens': max(counts)}
(work / 'renderer-proof.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
