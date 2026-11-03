"""Known Hub rubric edge cases, including its nonzero malformed-format score."""

import pytest

pytest.importorskip("textarena")
from tiny_prl_wordle import parse_answer, rewards


def test_exact_solution_keeps_all_hub_components():
    completion = [{"role": "assistant", "content": "<guess>[rough]</guess>"}]
    assert rewards(completion, "rough") == {
        "correct_answer": 1.0,
        "partial_answer": 0.0,
        "length_bonus": 1.0,
        "format_reward": 1.0,
    }


def test_legacy_partial_and_missing_format_are_preserved():
    completion = [
        {"role": "assistant", "content": "invalid"},
        {"role": "user", "content": "guess\nG Y X G Y\nremaining"},
    ]
    result = rewards(completion, "rough")
    assert result["partial_answer"] == pytest.approx(0.6)
    assert result["format_reward"] == 0.2
    assert result["correct_answer"] == 0.0


def test_parser_keeps_first_tag_of_last_parseable_assistant():
    completion = [
        {"role": "assistant", "content": "<guess>[apple]</guess><guess>[rough]</guess>"},
        {"role": "assistant", "content": "malformed"},
    ]
    assert parse_answer(completion) == "[apple]"
