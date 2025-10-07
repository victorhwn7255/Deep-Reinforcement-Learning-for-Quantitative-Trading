import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    # 1. CREATE ENVIRONMENT
    env_id = 'Pendulum-v1'
    env = gym.make(env_id)
    
    # 2. SET UP AGENT SETTINGS
    agent = Agent(
        alpha=3e-4,              # Actor learning rate
        beta=3e-4,               # Critic learning rate  
        input_dims=[3],          # Observation space: [cos(θ), sin(θ), angular_velocity]
        tau=0.005,               # Polyak averaging factor
        env=env,                 # Environment (needed for action space)
        env_id="Pendulum-v1",    # Name for saved models
        gamma=0.99,              # Discount factor
        n_actions=1,             # Action space: 1D continuous torque [-2, 2]
        max_size=1000000,        # Replay buffer size
        layer1_size=256,         # Hidden layer 1
        layer2_size=256,         # Hidden layer 2
        batch_size=256,          # Training batch size
        reward_scale=1.0,        # Reward scaling
        target_entropy=-0.5,     # Target entropy for exploration
        alpha_lr=1e-4,           # Temperature learning rate
        policy_type="normal"     # Gaussian policy for continuous control
    )
    
    # 3. TRAIN THE AGENT
    n_games = 1800
    steps = 0
    score_history = []
    best_score = -np.inf
    
    print("Starting SAC training...")
    
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
            
            # Start learning after sufficient experience
            if agent.memory.mem_cntr > 2100:
                agent.learn()
                
            score += reward
            observation = observation_
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        # Save best models during training
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        if i % 5 == 0:
            progress = i / n_games * 100
            print(f'[{progress:5.1f}%] Ep {i:3d}: Score {score:6.1f} | Avg {avg_score:6.1f} | Total Steps {steps:5d}')
    
    print(f"Training completed! Final average score: {avg_score:.1f}")
    
    # 4. SAVE FINAL MODEL
    agent.save_models()
    print("Final model saved!")
    
    # 5. VISUALIZE TRAINING CURVE
    filename = f'{env_id}_{n_games}games_scale{agent.scale}_final.png'
    figure_file = '../../plots/' + filename
    
    # Create plots directory if it doesn't exist
    os.makedirs('../../plots/', exist_ok=True)
    
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
    print(f"Training curve saved to: {figure_file}")
    
    # 6. RECORD VIDEO OF TRAINED AGENT
    print("Recording video of trained agent...")
    
    # Create environment with video recording
    video_env = RecordVideo(
        gym.make(env_id, render_mode='rgb_array'),
        video_folder='../../videos/',
        episode_trigger=lambda x: True,  # Record every episode
        name_prefix=f'sac_pendulum_final'
    )
    
    # Create videos directory if it doesn't exist
    os.makedirs('../../videos/', exist_ok=True)
    
    # Record 3 episodes of the trained agent
    for episode in range(3):
        observation, info = video_env.reset()
        done = False
        episode_score = 0
        
        while not done:
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = video_env.step(action)
            done = terminated or truncated
            episode_score += reward
            
        print(f"Recorded episode {episode+1}: Score {episode_score:.1f}")
    
    video_env.close()
    print("Video recording completed! Check ../../videos/ folder")
    print("Training and evaluation workflow finished!")