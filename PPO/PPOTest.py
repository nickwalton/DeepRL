from ppo import *
import cProfile


def RunPendulum():
    env = gym.make('Pendulum-v0')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden_size=10)

    model, avg_rewards = ppo_learn(env, model, episode_length=5000, render=False, epochs=50, env_samples=100, save_model=False)


def RunBipedal(load_model=False):
    env = gym.make('BipedalWalker-v2')
    model_path = './biped_models/biped_model'

    if load_model:
        model = torch.load(model_path)
        model.eval()
    else:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden_size=256)

    model, avg_rewards = ppo_learn(env, model, episode_length=2048, env_samples=4, render=True, epochs=800, model_save_path=model_path)


if __name__=='__main__':
    RunPendulum()