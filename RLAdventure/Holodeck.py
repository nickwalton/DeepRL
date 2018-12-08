# Copied from higgsfield RL-Adenture-2
# ://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from multienv import SubprocVecEnv
import holodeck
from holodeck.sensors import *

def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 1.0)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


# Masks allow it to be a huge sequential vector and not worry about where an episode stops and begins.
# Gamma allows us to control our trust in the value estimation.
# While tau allows us to assign more credit to recent actions 0 means you only get credit for immediate action.
# One would mean you get credit for all following actions
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.8):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                       rand_ids, :]


def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs,
                                                                         returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test_env(env, model, device, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def just_show(model_name):
    env_name = "BipedalWalker-v2"
    render = True
    env = gym.make(env_name)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = torch.load(model_name)
    test_env(env,model, device, vis=True)


def get_holo_state(state):
    state = np.append(state[Sensors.LOCATION_SENSOR], (state[Sensors.JOINT_ROTATION_SENSOR]))
    state = np.expand_dims(state, 0)
    return state


def get_holo_rew(state):
    return [state[Sensors.LOCATION_SENSOR][0]]


def train():
    # Number of envs to run in parallel
    num_envs = 1
    #env_name = "BipedalWalker-v2"
    env_name = "ExampleLevel"
    # env_name = "BipedalWalkerHardcore-v2"
    # env_name = 'Pendulum-v0'
    render = True

    env = holodeck.make(env_name)

    num_inputs = 97
    num_outputs = 94

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Max seen reward
    reward_max = -120

    # Hyper params:
    hidden_size = 128
    lr = 1e-4
    # The number of steps taken in each environment before training
    num_steps = 20
    mini_batch_size = 50
    ppo_epochs = 4
    epochs = 100

    model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Max frames is the max number of steps to run each env through (env is reset when done)
    optimizer_shake_up = 8000
    max_frames = 1000000 * 20
    new_model = optimizer_shake_up * 5
    frame_idx = 0
    test_rewards = []
    ep_len = 1000
    env_samples = 10

    state, reward, terminal, _ = env.reset()
    state = get_holo_state(state)

    early_stop = False

    for e in range(epochs):

        if frame_idx % optimizer_shake_up is 0:
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
            print("Reset optimizer")

        if frame_idx % new_model is 0:
            model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
            print("Reset model")

        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0

        for ep in range(env_samples):

            for _ in range(ep_len):
                state = torch.FloatTensor(state).to(device)
                dist, value = model(state)

                action = dist.sample()

                # Steps through each environment resetting if done
                next_state, reward, done, _ = env.step(action.cpu().numpy())
                reward = get_holo_rew(next_state)
                next_state = get_holo_state(next_state)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

                states.append(state)
                actions.append(action)

                state = next_state
                frame_idx += 1

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values

        ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)


if __name__ == '__main__':

    train()
    #just_show("../RLAdventure/Models/AwesomeWalker1")