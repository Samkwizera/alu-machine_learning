#!/usr/bin/env python3
"""Display games played by a trained Breakout DQN agent."""

import gym
import numpy as np
from PIL import Image
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    """Convert Atari frames into values suitable for the trained DQN."""

    def process_observation(self, observation):
        """Resize an RGB game frame and convert it to grayscale."""
        image = Image.fromarray(observation)
        image = image.resize(INPUT_SHAPE).convert("L")
        return np.array(image, dtype=np.uint8)

    def process_state_batch(self, batch):
        """Normalize a batch of byte-valued frames."""
        return batch.astype("float32") / 255.0


def build_model(actions):
    """Rebuild the policy network used while training."""
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
    """Load the trained policy and display five Breakout games."""
    env = gym.make("BreakoutDeterministic-v4")
    actions = env.action_space.n

    agent = DQNAgent(
        model=build_model(actions),
        nb_actions=actions,
        policy=GreedyQPolicy(),
        memory=SequentialMemory(limit=1000000,
                                window_length=WINDOW_LENGTH),
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
    agent.load_weights("policy.h5")
    agent.test(env, nb_episodes=5, visualize=True)
    env.close()


if __name__ == "__main__":
    main()
