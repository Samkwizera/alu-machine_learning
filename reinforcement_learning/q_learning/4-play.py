#!/usr/bin/env python3
"""Play Frozen Lake using a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Display and play one greedy episode, returning its total reward."""
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    total_reward = 0
    env.render()

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        result = env.step(action)
        if len(result) == 5:
            state, reward, terminated, truncated, _ = result
            done = terminated or truncated
        else:
            state, reward, done, _ = result
        total_reward += reward
        env.render()

        if done:
            break

    return total_reward
