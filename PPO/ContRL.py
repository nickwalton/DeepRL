from ppo import *


def RunPendulum():
    env = gym.make('Pendulum-v0')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden_size=10)

    model, avg_rewards = ppo_learn(env, model, episode_length=1000, render=True, render_wait=0.2)


def RunBipedal(load_model=False, model_path = './biped_model'):
    env = gym.make('BipedalWalker-v2')

    if load_model:
        model = torch.load(model_path)
        model.eval()
    else:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden_size=32)

    model, avg_rewards = ppo_learn(env, model, episode_length=2000, render=True, epochs=200)

    torch.save(model, model_path)


if __name__=='__main__':
    RunBipedal(load_model=True)