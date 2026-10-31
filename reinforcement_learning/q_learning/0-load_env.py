#!/usr/bin/env python3
"""Create Frozen Lake environments."""

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load and return a FrozenLake environment.

    A random 8-by-8 map is generated when neither a description nor a named
    map is supplied.
    """
    if desc is None and map_name is None:
        desc = generate_random_map(size=8)

    try:
        env = gym.make(
            "FrozenLake-v0",
            desc=desc,
            map_name=map_name,
            is_slippery=is_slippery,
        )
    except gym.error.Error:
        env = gym.make(
            "FrozenLake-v1",
            desc=desc,
            map_name=map_name,
            is_slippery=is_slippery,
        )

    return getattr(env, "unwrapped", env)
