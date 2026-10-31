#!/usr/bin/env python3
"""Train a Frozen Lake agent using Q-learning."""

import numpy as np

epsilon_greedy = __import__("2-epsilon_greedy").epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Train an agent and return its Q-table and rewards per episode."""
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            result = env.step(action)
            if len(result) == 5:
                new_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                new_state, reward, done, _ = result

            if done and reward == 0:
                reward = -1

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )
            state = new_state
            episode_reward += reward

            if done:
                break

        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )
        total_rewards.append(episode_reward)

    return Q, total_rewards
