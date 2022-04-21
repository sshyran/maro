# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Callable, Dict

from maro.rl.policy import AbsPolicy
from maro.simulator import Env
from .config import action_shaping_conf, algorithm, num_agents, state_dim

action_num = len(action_shaping_conf["action_space"])


def get_policy_creator(env: Env) -> Dict[str, Callable[[str], AbsPolicy]]:
    if algorithm == "ac":
        from .algorithms.ac import get_ac, get_policy
        policy_creator = {f"ac_{i}.policy": partial(get_policy, state_dim, action_num) for i in range(num_agents)}
    elif algorithm == "ppo":
        from .algorithms.ppo import get_ppo, get_policy
        policy_creator = {f"ppo_{i}.policy": partial(get_policy, state_dim, action_num) for i in range(num_agents)}
    elif algorithm == "dqn":
        from .algorithms.dqn import get_dqn, get_policy
        policy_creator = {f"dqn_{i}.policy": partial(get_policy, state_dim, action_num) for i in range(num_agents)}
    elif algorithm == "discrete_maddpg":
        from .algorithms.maddpg import get_policy, get_maddpg
        policy_creator = {
            f"discrete_maddpg_{i}.policy": partial(get_policy, state_dim, action_num) for i in range(num_agents)
        }
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return policy_creator


def get_trainer_creator(env: Env) -> Dict[str, Callable[[str], AbsPolicy]]:
    if algorithm == "ac":
        from .algorithms.ac import get_ac, get_policy
        trainer_creator = {f"ac_{i}": partial(get_ac, state_dim) for i in range(num_agents)}
    elif algorithm == "ppo":
        from .algorithms.ppo import get_ppo, get_policy
        trainer_creator = {f"ppo_{i}": partial(get_ppo, state_dim) for i in range(num_agents)}
    elif algorithm == "dqn":
        from .algorithms.dqn import get_dqn, get_policy
        trainer_creator = {f"dqn_{i}": partial(get_dqn) for i in range(num_agents)}
    elif algorithm == "discrete_maddpg":
        from .algorithms.maddpg import get_policy, get_maddpg
        trainer_creator = {f"discrete_maddpg_{i}": partial(get_maddpg, state_dim, [1]) for i in range(num_agents)}
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return trainer_creator


def get_device_mapping(env: Env) -> Dict[str, str]:
    return {policy_name: "cpu" for policy_name in get_policy_creator(env)}
