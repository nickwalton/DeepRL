from RL.RLAlgos.ppo import *
import torch.nn as nn
from torch.distributions import Normal


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
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


def RunPendulum():
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden_size=24)

    model, avg_rewards = ppo_learn(model, env_name, render=True)


def RunBipedal(load_model=False, model_path = './biped_model'):
    env = gym.make('BipedalWalker-v2')

    if load_model:
        model = torch.load(model_path)
        model.eval()
    else:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden_size=32)

    model, avg_rewards = ppo_learn(env, model, )
#.save(model, model_path)


if __name__=='__main__':
    RunPendulum()
