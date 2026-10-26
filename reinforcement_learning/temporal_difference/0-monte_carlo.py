#!/usr/bin/env python3
"""Monte Carlo value prediction."""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Estimate state values using constant-step-size Monte Carlo updates.

    Args:
        env: OpenAI Gym environment.
        V: Array containing the current value estimate for every state.
        policy: Callable returning an action for a supplied state.
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
        trajectory = []

        for _ in range(max_steps):
            action = policy(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            trajectory.append((state, reward))
            state = next_state
            if done:
                break

        total_return = 0
        for state, reward in reversed(trajectory):
            total_return = reward + gamma * total_return
            V[state] += alpha * (total_return - V[state])

    return V
