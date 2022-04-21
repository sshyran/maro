# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys
from typing import Any, Callable, Dict, List

from maro.rl.policy import AbsPolicy
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer
from maro.simulator import Env


class Scenario(object):
    def __init__(self, path: str) -> None:
        super(Scenario, self).__init__()
        path = os.path.normpath(path)
        sys.path.insert(0, os.path.dirname(path))
        self._module = importlib.import_module(os.path.basename(path))

    @property
    def env_sampler_creator(self) -> Callable[[Callable[[Env], Dict[str, Callable[[str], AbsPolicy]]]], AbsEnvSampler]:
        return getattr(self._module, "env_sampler_creator")

    @property
    def get_agent2policy(self) -> Callable[[Env], Dict[Any, str]]:
        return getattr(self._module, "get_agent2policy")

    @property
    def get_policy_creator(self) -> Callable[[Env], Dict[str, Callable[[str], AbsPolicy]]]:
        return getattr(self._module, "get_policy_creator")

    @property
    def get_trainable_policies(self) -> Callable[[Env], List[str]]:
        return getattr(self._module, "get_trainable_policies", None)

    @property
    def get_trainer_creator(self) -> Callable[[Env], Dict[str, Callable[[str], AbsTrainer]]]:
        return getattr(self._module, "get_trainer_creator")

    @property
    def get_device_mapping(self) -> Callable[[Env], Dict[str, str]]:
        return getattr(self._module, "get_device_mapping", {})

    @property
    def post_collect(self) -> Callable[[list, int, int], None]:
        return getattr(self._module, "post_collect", None)

    @property
    def post_evaluate(self) -> Callable[[list, int], None]:
        return getattr(self._module, "post_evaluate", None)
