import time
from pathlib import Path

from huggingface_hub import snapshot_download

from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_time


def pre_download_model(model_name: str, *, skip_weights: bool = False) -> None:
    """Cache model files before starting trainer and inference processes."""
    if Path(model_name).exists():
        get_logger().info(f"Model {model_name} found at local path, skipping download")
        return
    t0 = time.perf_counter()
    if skip_weights:
        get_logger().info(f"Pre-downloading config and tokenizer for {model_name} (random init, skipping weights)")
        path = snapshot_download(
            repo_id=model_name, repo_type="model", allow_patterns=["*.json", "*.txt", "tokenizer*", "*.jinja"]
        )
    else:
        get_logger().info(f"Pre-downloading model {model_name}")
        path = snapshot_download(repo_id=model_name, repo_type="model")
    get_logger().debug(
        f"Finished pre-downloading model {model_name} to {path} in {format_time(time.perf_counter() - t0)}"
    )
