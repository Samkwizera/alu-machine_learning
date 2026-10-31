#!/usr/bin/env python3
"""Train a Deep Q-Network to play Atari Breakout."""

import gym
import numpy as np
from PIL import Image
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    """Convert Atari frames and rewards into values suitable for a DQN."""

    def process_observation(self, observation):
        """Resize an RGB game frame and convert it to grayscale."""
        image = Image.fromarray(observation)
        image = image.resize(INPUT_SHAPE).convert("L")
        return np.array(image, dtype=np.uint8)

    def process_state_batch(self, batch):
        """Normalize a batch of byte-valued frames."""
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        """Clip rewards to stabilize training."""
        return np.clip(reward, -1.0, 1.0)


def build_model(actions):
    """Build the convolutional policy network used by the DQN agent."""
    model = Sequential()
    model.add(Permute((2, 3, 1),
                      input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def main():
    """Train the Breakout agent and save its final policy weights."""
    env = gym.make("BreakoutDeterministic-v4")
    actions = env.action_space.n

    memory = SequentialMemory(limit=1000000,
                              window_length=WINDOW_LENGTH)
    policy = EpsGreedyQPolicy(eps=0.1)
    agent = DQNAgent(
        model=build_model(actions),
        nb_actions=actions,
        policy=policy,
        memory=memory,
        processor=AtariProcessor(),
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0,
        enable_dueling_network=True,
        dueling_type="avg",
    )
    agent.compile(Adam(lr=0.00025), metrics=["mae"])
    agent.fit(env, nb_steps=1750000, visualize=False, verbose=2)
    agent.save_weights("policy.h5", overwrite=True)
    env.close()


if __name__ == "__main__":
    main()
