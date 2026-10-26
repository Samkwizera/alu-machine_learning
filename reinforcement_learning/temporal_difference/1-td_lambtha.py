#!/usr/bin/env python3
"""Temporal-difference learning with eligibility traces."""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Estimate state values with the TD(lambda) algorithm.

    Args:
        env: OpenAI Gym environment.
        V: Array containing the current value estimate for every state.
        policy: Callable returning an action for a supplied state.
        lambtha: Eligibility-trace decay factor.
        episodes: Number of episodes used for training.
        max_steps: Maximum number of transitions in one episode.
        alpha: Learning rate.
        gamma: Reward discount factor.

    Returns:
        The updated value estimate ``V``.
    """
    for _ in range(episodes):
        reset_result = env.reset()
        state = (reset_result[0] if isinstance(reset_result, tuple)
                 else reset_result)
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1
            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha
            state = next_state
            if done:
                break

    return V
