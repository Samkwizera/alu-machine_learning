#!/usr/bin/env python3
"""On-policy SARSA learning with eligibility traces."""

import numpy as np


def _epsilon_greedy(Q, state, epsilon):
    """Select an action from ``Q`` using an epsilon-greedy policy."""
    if np.random.uniform() < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """Update a Q table using SARSA(lambda).

    Args:
        env: OpenAI Gym environment.
        Q: State-action value table.
        lambtha: Eligibility-trace decay factor.
        episodes: Number of episodes used for training.
        max_steps: Maximum number of transitions in one episode.
        alpha: Learning rate.
        gamma: Reward discount factor.
        epsilon: Initial epsilon-greedy exploration threshold.
        min_epsilon: Lower bound for epsilon.
        epsilon_decay: Exponential epsilon decay rate.

    Returns:
        The updated state-action value table ``Q``.
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        reset_result = env.reset()
        state = (reset_result[0] if isinstance(reset_result, tuple)
                 else reset_result)
        action = _epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q)

        for _ in range(max_steps):
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            next_action = _epsilon_greedy(Q, next_state, epsilon)
            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])
            eligibility[state, action] += 1
            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha
            state, action = next_state, next_action
            if done:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * episode))

    return Q
