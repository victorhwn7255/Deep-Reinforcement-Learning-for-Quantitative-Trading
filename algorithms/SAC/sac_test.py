import os

import numpy as np
import gymnasium as gym

from agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env_id = 'Pendulum-v1'
    env = gym.make(env_id)
    
    #########################
    ### Pendulum-v1 ###
    #########################
    # Action space: 1D continuous torque [-2, 2]
    # Observation space: 3D continuous [cos(θ), sin(θ), angular_velocity]
    # Reward: -(θ² + 0.1*ω² + 0.001*action²)
    # Goal: Keep pendulum upright (minimize angle and angular velocity)
      
    agent = Agent(
      alpha=0.0003,              # Actor learning rate
      beta=0.0003,               # Critic learning rate  
      input_dims=[3],          # Observation space: [cos(θ), sin(θ), angular_velocity]
      tau=0.005,               # Polyak averaging factor
      env=env,                 # Environment (needed for action space)
      env_id="Pendulum-v1",  # Name for saved models
      gamma=0.99,              # Discount factor
      n_actions=1,             # Action space: 1D continuous torque [-2, 2]
      max_size=1000000,      # Replay buffer size
      layer1_size=256,         # Hidden layer 1
      layer2_size=256,         # Hidden layer 2
      batch_size=256,          # Training batch size
      reward_scale=1.0,        # Pendulum rewards are negative, scale down
      target_entropy=-0.5,     # Target entropy = -|action_dims| = -1
      alpha_lr=0.0001,           # Temperature learning rate
      policy_type="normal"     # Gaussian policy for continuous control
    )
    
    n_games = 1800
    
    filename = env_id + '_'+ str(n_games) + 'games_scale' + str(agent.scale) + '_clamp_on_sigma.png'
    figure_file = '../../plots/' + filename

    best_score = -np.inf
    score_history = []
    load_checkpoint = False
    
    ### for TESTING Mode:
    if load_checkpoint:
        agent.load_models()         # Load pre-trained weights
        env.render(mode='human')    # Show visual simulation
        
    steps = 0
    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            agent.remember(observation, action, reward, observation_, done)
            
            if not load_checkpoint and agent.memory.mem_cntr > 2100:
                agent.learn()
                
            score += reward
            observation = observation_
        
        score_history.append(score)
        
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        if i % 5 == 0:
            progress = i / n_games * 100
            print(f'[{progress:5.1f}%] Ep {i:3d}: Score {score:6.1f} | Avg {avg_score:6.1f} | Total Steps {steps:5d}')
    
    ### for TRAINING Mode:
    if not load_checkpoint:
        # Episode numbers [1,2,3...n_games]
        x = [i+1 for i in range(n_games)]  
        # Save learning curve plot 
        plot_learning_curve(x, score_history, figure_file)