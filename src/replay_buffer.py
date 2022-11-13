import numpy as np
import random
import torch as T


class Experience:
    def __init__(self, state, action, reward, state_, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.state_ = state_
        self.done = done

    def __call__(self):
        return (
            self.state,
            self.action,
            np.flip(self.action, 0),
            self.reward,
            self.state_,
            np.flip(self.state_, 0),
            self.done,
        )


class ReplayBuffer:
    def __init__(self, batch_size=64, max_size=1000000, device=T.device("cpu")):
        self.buffer = []
        self.batch_size = batch_size
        self.max_size = max_size
        self.head = -1
        self.device = device

    def add(self, state, action, action_other, reward, state_, state_other_, done):
        # move buffer head
        self.head = (self.head + 1) % self.max_size
        if self.__len__() < self.max_size:
            self.buffer += [None]

        self.buffer[self.head] = (
            state,
            action,
            action_other,
            reward,
            state_,
            state_other_,
            int(done),
        )

    def sample(self):
        batch = random.choices(self.buffer, k=self.batch_size)

        return self.prepare_batch(batch)

    def prepare_batch(self, batch):
        states = (
            T.from_numpy(np.vstack([e[0] for e in batch if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            T.from_numpy(np.vstack([e[1] for e in batch if e is not None]))
            .float()
            .to(self.device)
        )
        actions_other = (
            T.from_numpy(np.vstack([e[2] for e in batch if e is not None]))
            .float()
            .to(self.device)
        )
        rewards = (
            T.from_numpy(np.vstack([e[3] for e in batch if e is not None]))
            .float()
            .to(self.device)
        )
        states_ = (
            T.from_numpy(np.vstack([e[4] for e in batch if e is not None]))
            .float()
            .to(self.device)
        )
        states_other_ = (
            T.from_numpy(np.vstack([e[5] for e in batch if e is not None]))
            .float()
            .to(self.device)
        )
        dones = (
            T.from_numpy(np.vstack([e[6] for e in batch if e is not None]))
            .float()
            .to(self.device)
        )

        return states, actions, actions_other, rewards, states_, states_other_, dones

    def __len__(self):
        return len(self.buffer)

    def __call__(self, state, action, reward, state_, done):
        self.add(state, action, reward, state_, done)
