from pathlib import Path

from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.trainer import TokenizerConfig
from prime_rl.utils.logger import get_logger


def setup_tokenizer(config: TokenizerConfig) -> PreTrainedTokenizer:
    logger = get_logger()
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)
    if config.chat_template is not None:
        path = Path(config.chat_template)
        if path.is_file():
            logger.info(f"Loading custom chat template from file: {path}")
            tokenizer.chat_template = path.read_text()
            logger.debug(f"Chat template content:\n{tokenizer.chat_template}")
        else:
            logger.info("Using inline custom chat template")
            tokenizer.chat_template = config.chat_template
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer
