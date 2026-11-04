# Laptop cleanup handoff

Clear this laptop's model caches and `/home/pradheep/tiny-prl` after preserving
any work that is not already backed up. Keep the installed `tiny-prl` command
working. Do the work yourself; do not delegate or launch training.

The command is `/home/pradheep/.local/bin/tiny-prl`, a real executable, not a
symlink into the checkout. Its runtime is `/home/pradheep/.local/share/tiny-prl`.
The pinned venv is `/home/pradheep/.venv`; its interpreter lives under
`/home/pradheep/.local/share/uv/python`. Keep those paths, CUDA, the NVIDIA driver,
uv, and the HF credential store. Never print or copy authentication tokens into
logs, source, archives or this prompt.

Private experiment and runtime backups are at
https://huggingface.co/datasets/Pradheep1647/tiny-prl-experiment-archive
and its `runtime/` folder. Check the latest runtime README and checksum manifest.
Current source is in https://github.com/Mantissagithub/tiny-prl and
https://github.com/Mantissagithub/prime-kernels. Check remote parity using
`git -C` on each repository. Preserve any newer local-only work before deleting.
The old IcePop worktree had uncommitted code; verify its archived copy too.

Two installed Qwen3 reverse-text smoke checkpoints were uploaded to private HF
repos `Pradheep1647/tiny-prl-qwen3-0.6b-reverse-text-r01` and `r02`. Their exact
remote files were verified before the uploader deleted their local model files.
Other remaining checkpoints may not be uploaded. Archive and verify anything
else that needs preserving before deleting it. These are integration checkpoints,
not demonstrated accuracy improvements. Exported HF weights omit optimizer state.

First confirm no training, export or upload is active. Inventory disk usage and
exact deletion targets. Remove confirmed model caches, including
`~/.cache/huggingface/hub/models--*`, legacy Transformers model caches, HF Xet
transfer caches, `~/.cache/tiny-prl/models`, and Torch Hub checkpoints when present.
Inventory other model stores before removing confirmed model-cache directories.
Preserve HF `token` and `stored_tokens`, datasets, unrelated projects, and all
runtime/interpreter paths above. Do not wipe all of `~/.cache` or `.local`.
Then remove only `/home/pradheep/tiny-prl` for the checkout cleanup.

Afterward, run `tiny-prl --help` and `tiny-prl doctor` from `/home/pradheep`.
Run a config-only check with `tiny-prl train qwen3-600m --env primeintellect:gsm8k
--output /tmp/tiny-prl-post-cleanup-check --dry-run`. This must not train or
download model weights. Confirm `tiny-prl initialize --help` works, and HF
authentication still identifies Pradheep1647. Report disk space recovered,
preserved paths, backup locations and any failures.

Future `tiny-prl initialize ~/my-training` downloads base weights, offers the
reviewed model/environment choices, and creates editable `algo.py`. HF naming is
`Pradheep1647/tiny-prl-<model>-<env>-rNN`, with the next free run number selected
at training start. New repos default to private. Local model files are deleted
only after a successful upload and content verification; failures keep weights.
