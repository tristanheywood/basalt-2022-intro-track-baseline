from pathlib import Path
import pickle
from typing import Optional, Union
from openai_vpt.agent import MineRLAgent

from test import MODEL, WEIGHTS


def make_agent(
    model: Optional[Union[Path, str]] = MODEL,
    weights: Optional[Union[Path, str]] = WEIGHTS,
) -> MineRLAgent:
    agent_parameters = pickle.load(open(MODEL, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(
        None, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs
    )
    agent.load_weights(WEIGHTS)

    return agent
