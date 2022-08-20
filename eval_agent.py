import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import aicrowd_gym
import numpy as np

from openai_vpt.agent import MineRLAgent
from tools import make_agent

TRACKED_ITEMS = [
    "stick",
    "crafting_table",
    "wooden_pickaxe",
    "cobblestone",
    "furnace",
    "stone_pickaxe",
    "iron_ore",
    "iron_ingot",
    "iron_pickaxe",
    "diamond",
    "diamond_shovel",
]


@dataclass
class RolloutResults:
    rewards: List[int]
    items: Dict[str, int]

    def to_dict(self) -> Dict:
        rewards = np.array(self.rewards)
        non_zero_idx = np.argwhere(rewards).squeeze()
        non_zero_vals = rewards[non_zero_idx]

        return dict(
            num_steps=len(self.rewards),
            non_zero_reward_steps=list(non_zero_idx),
            non_zero_rewards=list(non_zero_vals.tolist),
            items=self.items,
        )

    @staticmethod
    def from_dict(d: Dict) -> "RolloutResults":
        rewards = np.zeros(d["num_steps"])
        non_zero_idx = np.array(d["non_zero_reward_steps"])
        non_zero_vals = np.array(d["non_zero_rewards"])
        rewards[non_zero_idx] = non_zero_vals

        return RolloutResults(rewards.tolist(), d["items"])

    @staticmethod
    def list_to_json(results: List["RolloutResults"], json_file: Union[Path, str]):
        results = [result.to_dict() for result in results]
        json.dump(results, open(json_file, "w"), indent=2)

    @staticmethod
    def list_from_json(json_file_or_dir: Union[Path, str]) -> List["RolloutResults"]:
        if json_file_or_dir.is_dir():
            return sum(
                [
                    RolloutResults.list_from_json(json_file)
                    for json_file in json_file_or_dir.glob("*.json")
                ],
                [],
            )
        results = json.load(open(json_file_or_dir, "r"))
        return [RolloutResults.from_dict(result) for result in results]


def do_rollout(agent: MineRLAgent, max_steps: int = 18000, episode: int = 0):

    env = aicrowd_gym.make("MineRLObtainDiamondShovel-v0")
    obs = env.reset()
    agent.reset()

    rewards = []
    agent_total_time = 0
    env_total_time = 0

    for step in range(max_steps):

        agent_start = time.time()
        minerl_action = agent.get_action(obs)
        agent_total_time += time.time() - agent_start

        env_start = time.time()
        obs, reward, done, info = env.step(minerl_action)
        env_total_time += time.time() - env_start

        rewards.append(reward)

        if done:
            print("done")
            break

        if step % 100 == 0:
            print(f"{episode} {step}")

    items = {}
    for item in TRACKED_ITEMS:
        items[item] = obs["inventory"][item].item()

    env.close()

    print(
        f"Episode {episode} complete. Total reward: {sum(rewards)}, Agent time: {agent_total_time}s, MineRL time: {env_total_time}s"
    )

    return RolloutResults(rewards, items)


def run(
    model: Path, weights: Path, results_file: Path, num_episodes: int, max_steps: int
):

    agent = make_agent(model, weights)

    results_file.parent.mkdir(exist_ok=True)
    results = []

    for episode in range(num_episodes):
        result = do_rollout(agent, max_steps, episode)
        results.append(result)

    RolloutResults.list_to_json(results, results_file)


def main():

    parser = argparse.ArgumentParser(description="Evaluate an agent")

    parser.add_argument("--model", type=str, default=None, help="Path to the model")
    parser.add_argument("--weights", type=str, default=None, help="Path to the weights")
    parser.add_argument(
        "--results-file",
        type=str,
        default="results.json",
        help="Path to the results file",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Maximum number of steps per episode"
    )

    args = parser.parse_args()

    run(
        Path(args.model),
        Path(args.weights),
        Path(args.results_file),
        args.num_episodes,
        args.max_steps,
    )


if __name__ == "__main__":
    main()
