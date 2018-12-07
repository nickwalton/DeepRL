import gym
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.distributions import Normal
from time import time
import numpy as np
import cProfile
from threading import Thread


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 1.0)


# Define the Actor Critic
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.log_std = nn.Parameter(torch.ones(1, output_size) * std)
        self.apply(init_weights)

    def forward(self, x, get_val=True):
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        if get_val:
            value = self.critic(x)
            return dist, value
        else:
            return dist


class ExperienceDataset(Dataset):
    def __init__(self, experience):
        super(ExperienceDataset, self).__init__()
        self._exp = []
        for x in experience:
            self._exp.extend(x)
        self._length = len(self._exp)

    def __getitem__(self, index):
        return self._exp[index]

    def __len__(self):
        return self._length


# Computes Discounted Sum of Rewards
def compute_returns(rollout, gamma=0.9):
    ret = 0

    for i in reversed(range(len(rollout))):
        obs, reward, action_dist, action = rollout[i]
        ret = reward + gamma * ret
        rollout[i] = (obs, reward, action_dist, action, ret)


def ppo_update(model, optimizer, epsilon, exp_loader, val_loss_func, val_losses, policy_losses):

    entropy_coeff = 0.005
    # Train network on batches of states
    for observation, reward, old_log_prob, action, ret in exp_loader:
        optimizer.zero_grad()
        new_dist, value = model(observation.float())

        ret = ret.unsqueeze(1)

        advantage = ret.float() - value.detach()

        new_log_prob = new_dist.log_prob(action)

        r_theta = (new_log_prob - old_log_prob).exp()

        clipped = r_theta.clamp(1 - epsilon, 1 + epsilon)

        objective = torch.min(r_theta * advantage, clipped * advantage)

        policy_loss = -torch.mean(objective)
        val_loss = val_loss_func(ret.float(), value)
        entropy = new_dist.entropy().mean()

        loss = policy_loss + val_loss - entropy_coeff * entropy
        loss.backward()

        optimizer.step()
        val_losses.append(val_loss.detach().numpy())
        policy_losses.append(policy_loss.detach().numpy())


# Trains a model on the given environment.
def ppo_learn(env, model, epochs=20, env_samples=100, episode_length=200, lr=1e-3, render=False,
              time_it=False, model_save_path="ppo_model", save_model=False):

    gamma = 0.9
    ppo_epochs = 4
    batch_size = 256
    epsilon = 0.2
    epochs_per_save = 1
    lr = 3e-4

    val_loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    val_losses = []
    policy_losses = []

    episode_avg_rewards = []

    for e in range(epochs):

        experience = []
        rewards = []

        # Create env_samples number of episode rollouts
        t1 = time()
        for j in range(env_samples):

            state = env.reset()
            state = state
            rollout = []

            # Each action in an episode
            for k in range(episode_length):
                torch_state = torch.FloatTensor(state).unsqueeze(0)
                dist = model(torch_state, get_val=False)

                action = dist.sample().numpy()[0]

                obs, reward, terminal, _ = env.step(action)
                rewards.append(reward)

                log_prob = dist.log_prob(torch.tensor(action))

                rollout.append((state, reward, log_prob.detach().numpy()[0], action))
                state = obs

                if render is True and j is 0 and e%10 is 0:
                    env.render()

                if terminal:
                    break

            compute_returns(rollout, gamma=gamma)
            experience.append(rollout)

        print("Gathered ", len(experience), " rollouts.")
        avg_rewards = sum(rewards) / env_samples
        episode_avg_rewards.append(avg_rewards)
        print("Epoch: ", e+1, "/", epochs, " Avg Reward: ", avg_rewards)
        t2 = time()
        if time_it: print("Running all episodes took ", t2-t1, " seconds.")

        exp_data = ExperienceDataset(experience)
        exp_loader = DataLoader(exp_data, batch_size=batch_size, shuffle=True, pin_memory=True)

        t1 = time()
        for _ in range(ppo_epochs):
            ppo_update(model, optimizer, epsilon, exp_loader, val_loss_func, val_losses, policy_losses)
        t2 = time()
        if time_it: print("Training took ", t2-t1, " seconds.")
        print(" ")

        if e % epochs_per_save == 0 and save_model:
            torch.save(model, model_save_path+str(e))

    return model, episode_avg_rewards
