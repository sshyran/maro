# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Callable, Dict, Set

from maro.rl.policy import AbsPolicy
from maro.simulator import Env


def single_mode_get_policy_creator(
    get_policy_creator: Callable[[Env], Dict[str, Callable[[str], AbsPolicy]]]
) -> Callable[[Env], Dict[str, Callable[[str], AbsPolicy]]]:
    policy_cache: Dict[str, AbsPolicy] = {}

    def _new_get_policy_creator(env: Env) -> Dict[str, Callable[[str], AbsPolicy]]:
        policy_creator = get_policy_creator(env)

        def _get_policy_func(policy_name: str) -> AbsPolicy:
            if policy_name not in policy_cache:
                policy_cache[policy_name] = policy_creator[policy_name](policy_name)
            return policy_cache[policy_name]

        return {policy_name: _get_policy_func for policy_name in policy_creator.keys()}

    return _new_get_policy_creator


def get_trainable_policy_creator(
    get_policy_creator: Callable[[Env], Dict[str, Callable[[str], AbsPolicy]]],
    trainable_policies: Set[str]
) -> Callable[[Env], Dict[str, Callable[[str], AbsPolicy]]]:
    def _get_trainable_policy_creator(env: Env) -> Dict[str, Callable[[str], AbsPolicy]]:
        policy_creator = get_policy_creator(env)
        return {
            policy_name: get_func
            for policy_name, get_func in policy_creator.items() if policy_name in trainable_policies
        }

    return _get_trainable_policy_creator


def get_trainable_agent2policy(
    get_agent2policy: Callable[[Env], Dict[Any, str]],
    trainable_policies: Set[str]
) -> Callable[[Env], Dict[Any, str]]:
    def _get_trainable_agent2policy(env: Env) -> Dict[Any, str]:
        agent2policy = get_agent2policy(env)
        return {
            agent_name: policy_name
            for agent_name, policy_name in agent2policy.items() if policy_name in trainable_policies
        }

    return _get_trainable_agent2policy
