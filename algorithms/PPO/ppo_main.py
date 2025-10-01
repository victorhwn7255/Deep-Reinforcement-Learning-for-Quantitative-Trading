import gymnasium as gym
import numpy as np
import torch

from agent import Agent

def action_adapter(a, max_a):
    return 2 * (a-0.5)*max_a
  
def clip_reward(x):
    if x < -1:
      return -1
    elif x > 1:
      return 1
    else:
      return x

if __name__ == '__main__':
    env_id = 'Pendulum-v1'
    random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    env = gym.make(env_id)
    
    N = 2048
    batch_size = 64
    n_epochs = 10
    learning_rate = 0.0003
    max_action = env.action_space.high[0]
    
    agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size,
                  learning_rate=learning_rate, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)

    score_history = []
    max_steps = 300_000
    total_steps = 0
    traj_length = 0
    episode = 1

    # for i in range(n_games):
    while total_steps < max_steps:
        observation, _ = env.reset(seed=random_seed if episode == 1 else None)
        done = False
        truncated = False
        score = 0
        while not (done or truncated):
            action, prob = agent.choose_action(observation)
            act = action_adapter(action, max_action)
            observation_, reward, done, truncated, info = env.step(act)
            r = clip_reward(reward)
            total_steps += 1
            traj_length += 1
            score += reward
            agent.remember(observation, observation_, action,
                           prob, r, done or truncated)
            if traj_length % N == 0:
                agent.learn()
                traj_length = 0
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('{} Episode {} total steps {} avg score {:.1f}'.
              format(env_id, episode, total_steps, avg_score))
        episode += 1