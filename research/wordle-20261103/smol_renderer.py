from typing import Literal

from renderers.base import RenderedTokens
from renderers.configs import BaseRendererConfig
from renderers.default import DefaultRenderer


class SmolConfig(BaseRendererConfig):
    name: Literal['smollm2-chatml'] = 'smollm2-chatml'


class SmolRenderer(DefaultRenderer):
    config_class = SmolConfig

    def __init__(self, tokenizer, config):
        super().__init__(tokenizer)

    @property
    def is_prefix_stable(self):
        return True

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        if tools:
            raise ValueError('SmolLM2 Wordle SFT is text-only')
        ids, indices, sampled, content = [], [], [], []

        def emit(text, index, is_sampled, is_content):
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            ids.extend(tokens)
            indices.extend([index] * len(tokens))
            sampled.extend([is_sampled] * len(tokens))
            content.extend([is_content] * len(tokens))

        if messages[0]['role'] != 'system':
            emit('<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n', -1, False, False)
        for i, message in enumerate(messages):
            assistant = message['role'] == 'assistant'
            emit('<|im_start|>' + message['role'] + '\n', i, False, False)
            emit(message['content'], i, assistant, True)
            emit('<|im_end|>', i, assistant, assistant)
            emit('\n', i, False, False)
        if add_generation_prompt:
            emit('<|im_start|>assistant\n', -1, False, False)
        return RenderedTokens(token_ids=ids, message_indices=indices,
            message_roles=[m['role'] for m in messages], sampled_mask=sampled,
            is_content=content)
