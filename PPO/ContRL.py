from ppo import *


def RunPendulum():
    env = gym.make('Pendulum-v0')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden_size=10)

    model, avg_rewards = ppo_learn(env, model, episode_length=1000, render=True, render_wait=0.2)


def RunBipedal(load_model=False):
    env = gym.make('BipedalWalker-v2')

    if load_model:
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden_size=32)

    model, avg_rewards = ppo_learn(env, model, episode_length=1000, render=True)


if __name__=='__main__':
    RunBipedal()