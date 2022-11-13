from src.model import Actor, Critic
from src.replay_buffer import ReplayBuffer
from src.noise import OUNoise
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 64
REPLAY_LENGTH = 10000
LEARN_EVERY = 1
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 2e-4
LR_CRITIC = 2e-4
WEIGHT_DECAY = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, s_dim, a_dim, n_agents, name="agent", seed=0):
        self.name = name

        # actor networks
        self.actor_local = Actor(
            s_dim=s_dim, a_dim=a_dim, name=f"{self.name}_actor_local", seed=seed
        ).to(device)
        self.actor_target = Actor(
            s_dim=s_dim, a_dim=a_dim, name=f"{self.name}_actor_target", seed=seed
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # critic networks
        self.critic_local = Critic(
            s_dim=s_dim,
            a_dim=a_dim,
            n_agents=n_agents,
            name=f"{self.name}_critic_local",
            seed=seed,
        ).to(device)
        self.critic_target = Critic(
            s_dim=s_dim,
            a_dim=a_dim,
            n_agents=n_agents,
            name=f"{self.name}_critic_target",
            seed=seed,
        ).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=REPLAY_LENGTH, batch_size=BATCH_SIZE, device=device
        )

        # noise process
        self.noise = OUNoise(n_agents * a_dim, seed)

        self.t_step = -1

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += np.reshape(self.noise.sample(), (-1, 2))

        return np.clip(action, -1, 1)

    def step(self, experience):
        self.t_step += 1
        self.replay_buffer.add(experience)

        if len(self.replay_buffer) >= BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
            self.learn()
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def learn(self):
        (
            states,
            actions,
            actions_other,
            rewards,
            states_,
            states_other_,
            dones,
        ) = self.replay_buffer()

        # update critic
        # 1. get actions for states_ and states_other_
        actions_ = self.actor_target(states_)
        actions_other_ = self.actor_target(states_other_)
        actions_all_ = torch.cat((actions_, actions_other_), dim=1)
        # 2. compute target Q values
        targets_ = self.critic_target(states_, actions_all_)
        targets = rewards + (GAMMA * targets_ * (1 - dones))
        # 3. compute current expected Q value
        actions_all = torch.cat((actions, actions_other), dim=1)
        predictions = self.critic_local(states, actions_all)

        loss = F.mse_loss(predictions, targets)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # update actor
        actions_prediction = self.actor_local(states)
        actions_all_pred = torch.cat((actions_prediction, actions_other), dim=1)

        loss = -self.critic_local(states, actions_all_pred).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save_weights(self):
        self.actor_local.save()
        self.actor_target.save()
        self.critic_local.save()
        self.critic_target.save()

    def load_weights(self):
        self.actor_local.load()
        self.actor_target.load()
        self.critic_local.load()
        self.critic_target.load()

