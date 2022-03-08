# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator import Env

env_conf = {
    "scenario": "supply_chain",
    # Currently available topologies are "sample" or "random". New topologies must consist of a single folder
    # that contains a single config.yml and should be placed under /maro/simulator/scenarios/supply_chain/topologies
    "topology": "sample",
    "durations": 100  # number of ticks per episode
}

# Sku related agent types
sku_agent_types = {"consumer", "consumerstore", "producer", "product", "productstore"}
distribution_features = ("remaining_order_quantity", "remaining_order_number")
seller_features = ("total_demand", "sold", "demand")

rl_algorithm = "dqn"

NUM_CONSUMER_ACTIONS = 10
NUM_RL_POLICIES = 5