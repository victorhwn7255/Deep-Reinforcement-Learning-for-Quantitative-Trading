import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Dirichlet

from networks import AdvantageActorCritic


class Agent:
    def __init__(self, 
                 n_input, 
                 n_action, 
                 learning_rate,
                 gamma,
                 entropy_weight,
                 value_weight,
                 max_norm,
                 device
                 ):
        self.device = device
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.max_norm = max_norm
        self.learning_rate = learning_rate
        
        # Initialize network
        self.ac_network = AdvantageActorCritic(
            n_input=n_input,
            n_action=n_action,
            n_hidden=128
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=self.learning_rate)
    
    def sample_action(self, alpha):
        """Sample actions from Dirichlet distribution"""
        # Move to CPU for Dirichlet sampling (MPS not supported)
        alpha_cpu = alpha.cpu()
        distribution = Dirichlet(alpha_cpu)
        sample = distribution.sample()
        return sample.to(self.device)
    
    def compute_entropy_and_log_probs(self, alpha, actions):
        """Compute entropy and log probabilities for policy gradient"""
        # Move to CPU for Dirichlet operations (MPS not supported)
        alpha_cpu = alpha.cpu()
        actions_cpu = actions.cpu()
        distribution = Dirichlet(alpha_cpu)
        log_probs = distribution.log_prob(actions_cpu).to(self.device)
        entropy = distribution.entropy().to(self.device)
        return entropy, log_probs
    
    def np2torch(self, a, dtype=torch.float32):
        """Convert numpy array to torch tensor"""
        return torch.as_tensor(a, dtype=dtype, device=self.device)
    
    def learn(self, env, total_timesteps):
        """Main training loop"""
        episode_returns = []
        losses = []
        
        # track the best model during training
        best_avg_return = -np.inf
        best_model_state = None
        
        start_time = time.time()
        obs = env.reset()
        G = 0  # keep track of RL-return
        
        for global_step in range(total_timesteps):
            # Get model output
            alpha, value = self.ac_network(self.np2torch(obs))
            
            # Select action based on current policy
            actions = self.sample_action(alpha)
            actions_np = actions.detach().cpu().numpy().flatten()
            
            if global_step % 1000 == 0:
                print("Current Portfolio:", actions_np)
            
            # Take a step in the environment
            next_obs, reward, done = env.step(actions_np)
            G += reward
            
            # Value loss
            with torch.no_grad():
                _, value_next = self.ac_network(self.np2torch(next_obs))
                td_target = self.np2torch(reward) + self.gamma * (1 - done) * value_next
            value_loss = F.mse_loss(value, td_target)
            
            # Policy loss
            entropy, selected_log_probs = self.compute_entropy_and_log_probs(alpha, actions)
            advantage = td_target - value
            policy_loss = -torch.mean(selected_log_probs * advantage.detach())
            
            # Total loss
            loss = policy_loss - self.entropy_weight * entropy + self.value_weight * value_loss
            losses.append(loss.item())
            
            # Gradient descent step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_norm)
            self.optimizer.step()
            
            if global_step % 1000 == 0:
                progress = (global_step / total_timesteps) * 100
                steps_per_sec = int(global_step / (time.time() - start_time))
                print(f'[{progress:5.1f}%] Step {global_step:6d}: {steps_per_sec:4d} steps/s | Device: {self.device}')
            
            # Record returns for plotting
            if done:
                episode_returns.append(G)
                print(f"global_step={global_step}, episode={len(episode_returns)}, episode_return={G}")
                
                ##########################################################
                ### Track and save best model based on rolling average ###
                ##########################################################
                if len(episode_returns) >= 10:
                    avg_return = np.mean(episode_returns[-10:])  # Last 10 episodes
                    
                    if avg_return > best_avg_return:
                        best_avg_return = avg_return
                        # Deep copy the model state
                        best_model_state = {
                            'model_state_dict': {k: v.cpu().clone() for k, v in self.ac_network.state_dict().items()},
                            'optimizer_state_dict': {k: v.cpu().clone() if torch.is_tensor(v) else v 
                                                    for k, v in self.optimizer.state_dict().items()},
                            'episode': len(episode_returns),
                            'global_step': global_step,
                            'avg_return': avg_return,
                            'all_returns': episode_returns.copy()
                        }
                        # Save best model immediately to disk (don't wait for training to finish!)
                        torch.save(best_model_state, "models/a2c_portfolio_best.pth")
                        print(f"  ✓ New best model! Avg return (last 10 episodes): {avg_return:.4f}")
                        print(f"    → Saved to models/a2c_portfolio_best.pth")
                
                G = 0  # reset
                obs = env.reset()
            else:
                obs = next_obs
        
        return episode_returns, losses, best_model_state
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save(self.ac_network.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        self.ac_network.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        print(f"Model loaded from {path}")
        
    def load_best_model(self, best_model_state):
        """Load the best model from training"""
        if best_model_state is not None:
            self.ac_network.load_state_dict(best_model_state['model_state_dict'])
            print(f"Best model loaded (Episode {best_model_state['episode']}, "
                  f"Avg Return: {best_model_state['avg_return']:.4f})")
        else:
            print("No best model state available")
