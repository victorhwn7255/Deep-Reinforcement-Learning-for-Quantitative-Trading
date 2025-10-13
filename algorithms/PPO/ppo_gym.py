import gymnasium as gym
import numpy as np
import torch
import time

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
    
    N = 512
    max_action = env.action_space.high[0]
    
    agent = Agent(
          n_actions=env.action_space.shape[0], 
          batch_size=128,
          learning_rate=0.001, 
          n_epochs=18,
          input_dims=env.observation_space.shape,
          fc1_dims=256, 
          fc2_dims=256  
    )

    score_history = []
    best_score = -np.inf
    start_time = time.time()
    max_steps = 600_000
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

            r = reward
            total_steps += 1
            traj_length += 1
            score += reward
            agent.remember(observation, observation_, action, prob, r, done or truncated)
            if traj_length % N == 0:
                agent.learn()
                traj_length = 0
            observation = observation_
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        progress = (total_steps / max_steps) * 100
        elapsed_time = time.time() - start_time
        steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
        print(f'[{progress:5.1f}%] Ep {episode:3d}: Score {score:6.1f} | Avg {avg_score:6.1f} | Total Steps {total_steps:6d} | {steps_per_sec:.0f} steps/s')
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            print(f"  ✓ New best avg: {avg_score:.1f}, models saved")
        
        episode += 1