# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from typing import Union

import numpy as np
import torch

from maro.rl.exploration import NoiseExplorer
from maro.rl.model import PolicyValueNetForContinuousActionSpace
from maro.rl.utils import get_torch_loss_cls

from .abs_policy import AbsPolicy

DDPGExperience = namedtuple("DDPGExperience", ["state", "action", "reward", "next_state"])

class DDPGConfig:
    """Configuration for the DDPG algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        target_update_freq (int): Number of training rounds between policy target model updates.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        policy_loss_coefficient (float): The coefficient for policy loss in the total loss function, e.g.,
            loss = q_value_loss + ``policy_loss_coefficient`` * policy_loss. Defaults to 1.0.
        soft_update_coefficient (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coefficient) * eval_model + (1-soft_update_coefficient) * target_model.
            Defaults to 1.0.
    """
    __slots__ = [
        "reward_discount", "q_value_loss_func", "target_update_freq", "policy_loss_coefficient",
        "soft_update_coefficient"
    ]

    def __init__(
        self,
        reward_discount: float,
        target_update_freq: int,
        q_value_loss_cls="mse",
        policy_loss_coefficient: float = 1.0,
        soft_update_coefficient: float = 1.0,
    ):
        self.reward_discount = reward_discount
        self.target_update_freq = target_update_freq
        self.q_value_loss_func = get_torch_loss_cls(q_value_loss_cls)()
        self.policy_loss_coefficient = policy_loss_coefficient
        self.soft_update_coefficient = soft_update_coefficient


class DDPG(AbsPolicy):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        ac_net (PolicyValueNetForContinuousActionSpace): DDPG policy and q-value models.
        config (DDPGConfig): Configuration for DDPG algorithm.
    """
    def __init__(self, ac_net: PolicyValueNetForContinuousActionSpace, config: DDPGConfig, explorer: NoiseExplorer = None):
        if not isinstance(ac_net, PolicyValueNetForContinuousActionSpace):
            raise TypeError("model must be an instance of 'PolicyValueNetForContinuousActionSpace'")

        super().__init__(config)
        self.ac_net = ac_net
        self.target_ac_net = ac_net.copy() if model.trainable else None
        self._explorer = explorer
        self._train_cnt = 0

    def choose_action(self, states) -> Union[float, np.ndarray]:
        with torch.no_grad():
            actions = self.ac_net.choose_action(states)

        actions = actions.cpu().numpy()
        if self._explorer:
            action = self._explorer(action)
        
        return actions[0] if len(actions) == 1 else actions

    def update(self, experience_obj: DDPGExperience):
        if not isinstance(experience_obj, DDPGExperience):
            raise TypeError(f"Expected experience object of type DDPGExperience, got {type(experience_obj)}")

        states, next_states = experience_obj.state, experience_obj.next_state
        actual_actions = torch.from_numpy(experience_obj.action)
        rewards = torch.from_numpy(experience_obj.reward)
        if len(actual_actions.shape) == 1:
            actual_actions = actual_actions.unsqueeze(dim=1)  # (N, 1)

        current_q_values = self.ac_net(torch.cat([states, actual_actions], dim=1), task_name="q_value")
        current_q_values = current_q_values.squeeze(dim=1)  # (N,)
        next_actions = self.target_ac_net(states, task_name="policy", training=False)
        next_q_values = self.target_ac_net(
            torch.cat([next_states, next_actions], dim=1), task_name="q_value", training=False
        ).squeeze(1)  # (N,)
        target_q_values = (rewards + self.config.reward_discount * next_q_values).detach()  # (N,)
        q_value_loss = self.config.q_value_loss_func(current_q_values, target_q_values)
        actions_from_model = self.ac_net(states, task_name="policy")
        policy_loss = -self.ac_net(torch.cat([states, actions_from_model], dim=1), task_name="q_value").mean()
        self.ac_net.step(q_value_loss + self.config.policy_loss_coefficient * policy_loss)
        self._train_cnt += 1
        if self._train_cnt % self.config.target_update_freq == 0:
            self.target_ac_net.soft_update(self.ac_net, self.config.soft_update_coefficient)
