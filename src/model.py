import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Network(nn.Module):
    def __init__(self, name, seed=0):
        super(Network, self).__init__()

        self.path = f"models/{name}.pth"
        self.seed = torch.manual_seed(seed)

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        if os.path.isfile(self.path):
            self.load_state_dict(torch.load(self.path))


class Actor(Network):
    def __init__(self, s_dim, a_dim, name, seed=0):
        super(Actor, self).__init__(name, seed)

        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, a_dim)
        self.bn = nn.BatchNorm1d(s_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = self.bn(state)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        return torch.tanh(self.fc3(x))


class Critic(Network):
    def __init__(self, s_dim, a_dim, n_agents, name, seed=0):
        super(Critic, self).__init__(name, seed)

        self.fcs1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128 + a_dim * n_agents, 128)
        self.fc3 = nn.Linear(128, 1)
        self.bn = nn.BatchNorm1d(s_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, actions):
        xs = self.bn(state)
        xs = F.leaky_relu(self.fcs1(xs))
        x = torch.cat((xs, actions), dim=1)
        x = F.leaky_relu(self.fc2(x))

        return self.fc3(x)
