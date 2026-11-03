"""Taskset API port of primeintellect/wordle 0.1.7 (Hub version eletiahb7kg3mkh4qbh6ixlw).

Keeps the Hub prompt, XML parsing, seeded word selection and weighted rubric.
The engine is TextArena 0.7.4; the parser behavior is Verifiers 0.1.9.post3.
"""

import random
import re
from typing import Literal

import textarena as ta
import verifiers.v1 as vf
from pydantic import Field

__all__ = ["WordleTaskset", "WordleEnv"]
SYSTEM_PROMPT = """You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""
GUESS = re.compile(r"<guess>\s*(.*?)\s*</guess>", re.DOTALL)


def parse_answer(completion):
    for message in reversed(completion):
        if message["role"] == "assistant" and (match := GUESS.search(message["content"])):
            return match.group(1).strip()
    return None


def feedback(observation):
    latest = observation.split("[GAME]")[-1].strip()
    return latest.split("Feedback:")[-1] if "Feedback:" in latest else latest


def rewards(completion, answer):
    assistants = [m["content"] for m in completion if m["role"] == "assistant"]
    correct = float(parse_answer(completion) == "[" + answer + "]")
    count = sum(bool(re.search(r"<guess>.*</guess>", text)) for text in assistants)
    partial = 0.0
    if not correct:
        for message in reversed(completion):
            if message["role"] == "user":
                parts = message["content"].strip().split("\n")
                if len(parts) == 3:
                    scoring = parts[1].strip()
                    partial = 0.2 * scoring.count("G") + 0.1 * scoring.count("Y")
                    break
    # Single-field XMLParser's spacing check remains true even for missing tags.
    formats = [
        0.2
        + 0.4 * bool(GUESS.search(text))
        + 0.2 * text.strip().startswith("<guess>")
        + 0.2 * text.strip().endswith("</guess>")
        for text in assistants
    ]
    return {
        "correct_answer": correct,
        "partial_answer": partial,
        "length_bonus": correct / (count or 1),
        "format_reward": sum(formats) / len(formats) if formats else 0.0,
    }


class WordleData(vf.TaskData):
    answer: str


class WordleTask(vf.Task[WordleData]):
    pass


class WordleConfig(vf.TasksetConfig):
    num_train_examples: int = Field(2000, ge=1)
    num_eval_examples: int = Field(20, ge=1)
    seed: int = 0
    split: Literal["train", "eval"] = "train"


class WordleTaskset(vf.Taskset[WordleTask, WordleConfig]):
    def load(self):
        template = ta.make(env_id="Wordle-v0")
        template.reset(num_players=1)
        prompt = str(template.get_observation()[1])
        rng = random.Random(self.config.seed)
        total = self.config.num_train_examples + self.config.num_eval_examples
        for i in range(total):
            answer = rng.choice(template.word_list)
            if (i < self.config.num_train_examples) != (self.config.split == "train"):
                continue
            yield WordleTask(
                WordleData(id=str(i), name=f"wordle-{i}", prompt=prompt, system_prompt=SYSTEM_PROMPT, answer=answer),
                self.config.task,
            )


class WordleEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig()


class WordleEnv(vf.Env[WordleEnvConfig]):
    async def run(self, task, agents):
        game = ta.make(env_id="Wordle-v0")
        game.reset(num_players=1)
        game.state.game_state["secret_word"] = task.data.answer
        completion = []
        async with agents.player.interaction(task) as interaction:
            segment = await interaction.turn()
            while True:
                if segment.last_reply:
                    completion.append({"role": "assistant", "content": segment.last_reply})
                if segment.terminated:
                    break
                game.step(str(parse_answer(completion)))
                if game.state.done:
                    reason = str(game.state.game_info[0]["reason"])
                    completion.append({"role": "user", "content": reason})
                    interaction.trace.info["game_outcome"] = reason
                    break
                observation = feedback(str(game.get_observation()[1]))
                completion.append({"role": "user", "content": observation})
                segment = await interaction.turn(observation)
        for name, score in rewards(completion, task.data.answer).items():
            interaction.trace.record_reward(name, score, 0.2 if name == "format_reward" else 1.0)
        interaction.trace.info["hub_environment"] = "primeintellect/wordle@0.1.7"
