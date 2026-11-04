from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def default_advantages(rewards, lengths):
    """Length-weighted GRPO centering, with exactly zero credit for tied rewards."""
    centered_rewards = rewards - rewards[:1]
    baseline = (lengths * centered_rewards).sum() / lengths.sum()
    return centered_rewards - baseline


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the length-weighted group mean
    (optionally length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.length_penalty = config.length_penalty
        self.loss_aggregation = config.loss_aggregation
        self.advantage_fn = import_object(config.advantage_fn) if config.advantage_fn else default_advantages

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        import torch  # only the trainer-side extras ship torch; an eval process never scores a group

        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            shaped_rewards = rewards
        else:
            output = torch.tensor([trace.num_output_tokens for trace in traces], dtype=rewards.dtype)
            total = torch.tensor([trace.num_total_tokens for trace in traces], dtype=rewards.dtype)
            turns = torch.tensor([trace.num_turns for trace in traces], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
        lengths = torch.tensor(
            [sum(sum(node.mask) for node in trainable_nodes(trace)) for trace in traces], dtype=rewards.dtype
        )
        advantages = torch.as_tensor(self.advantage_fn(shaped_rewards, lengths), dtype=rewards.dtype)
        if advantages.shape != rewards.shape or not torch.isfinite(advantages).all():
            raise ValueError("advantage_fn must return one finite scalar per trainable trace")
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
        nodes = [node for trace in traces for node in trainable_nodes(trace)]
        if self.loss_aggregation == "prompt" and nodes:
            # rl weight 1/T_q per token: each group's weights sum to 1, and the trainer divides the
            # rl loss by the sum of rl weights, i.e. the number of groups.
            weight = 1.0 / sum(sum(node.mask) for node in nodes)
            for node in nodes:
                node.loss_weights = {**(node.loss_weights or {}), "rl": [weight if m else 0.0 for m in node.mask]}
