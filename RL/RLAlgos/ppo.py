"""
Code modified from https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb?short_path=93fda56
"""
import gym
import torch
import torch.optim as optim
from RL.common.multiprocessing_env import SubprocVecEnv
import numpy as np

def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk


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


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
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


def ppo_learn(model, env_name, max_frames=15000, render=True):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #device = "cpu"
    model = model.to(device)

    num_envs = 8
    envs = [make_env(env_name) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    env = gym.make(env_name)

    # Hyper params:
    lr = 3e-4
    num_steps = 20
    mini_batch_size = 5
    ppo_epochs = 4
    threshold_reward = -200

    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    frame_idx = 0
    test_rewards = []

    state = envs.reset()
    early_stop = False

    while frame_idx < max_frames and not early_stop:

        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

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

            if frame_idx % 1000 == 0:
                test_reward = np.mean([test_env(env, model, device) for _ in range(10)])
                test_rewards.append(test_reward)
                test_env(env, model, device, vis=True)
                print("Frame: ", frame_idx, "Test Reward: ", test_reward)
                if test_reward > threshold_reward: early_stop = True

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

    return model, test_rewards
