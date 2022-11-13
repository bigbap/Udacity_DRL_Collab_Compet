from unityagents import UnityEnvironment
import os
import numpy as np


class EnvironmentTennis:
    def __init__(self, no_graphics=True, seed=0):
        PATH = (
            "./environment/Tennis.exe"
            if os.name == "nt"
            else "./environment/Tennis.x86_64"
        )
        self.env = UnityEnvironment(file_name=PATH, no_graphics=no_graphics, seed=seed,)

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # environment information
        self.action_size = self.brain.vector_action_space_size
        self.state_size = (
            self.brain.vector_observation_space_size
            * self.brain.num_stacked_vector_observations
        )
        self.num_agents = 2

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        state = env_info.vector_observations
        done = False

        return state, done

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        state_ = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done

        return reward, state_, done

    def close(self):
        self.env.close()


def episode(env, agent, train_mode=True, t_max=1000, rnd=False):
    agent.reset()
    scores = np.zeros(env.num_agents)
    state, done = env.reset(train_mode=train_mode)
    for t in range(t_max):
        action = np.clip(np.random.randn(2, 2), -1, 1) if rnd else agent.act(state)

        reward, state_, done = env.step(action)
        agent.step(state, action, reward, state_, done)

        scores = np.add(scores, reward)
        state = state_

        if done:
            break

    return scores
