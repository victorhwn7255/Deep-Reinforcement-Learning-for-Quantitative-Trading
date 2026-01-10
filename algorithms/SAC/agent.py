import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Dirichlet

from networks import SoftQNetwork, PolicyNetwork, ValueNetwork
from replay_buffer import ReplayBuffer

class Agent:
    """
    Soft Actor-Critic (SAC) Agent for Portfolio Management
    
    Key components:
    - Twin Q-networks (critics) to mitigate overestimation bias
    - Policy network (actor) with Dirichlet output for portfolio weights
    - Value network (optional, for stability)
    - Automatic entropy tuning
    - Soft target network updates (Polyak averaging)
    - Experience replay buffer for off-policy learning
    """
    
    def __init__(self, 
                 n_input, 
                 n_action, 
                 learning_rate=0.001,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 auto_entropy_tuning=True,
                 target_entropy=None,
                 buffer_size=1000000,
                 batch_size=256,
                 learning_starts=10000,
                 update_frequency=1,
                 n_hidden=256,
                 device='cpu'
                 ):
        """
        Initialize SAC agent
        
        Args:
            n_input: State dimension
            n_action: Action dimension (number of assets + cash)
            learning_rate: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            alpha: Initial entropy coefficient (temperature)
            auto_entropy_tuning: Whether to automatically tune alpha
            target_entropy: Target entropy for auto-tuning (default: -n_action)
            buffer_size: Replay buffer size
            batch_size: Batch size for updates
            learning_starts: Steps before starting to learn
            update_frequency: How often to update networks
            n_hidden: Hidden layer size
            device: Device for computation
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_frequency = update_frequency
        self.n_action = n_action
        self.learning_rate = learning_rate
        
        # Initialize Q-networks (critics)
        self.q1 = SoftQNetwork(n_input, n_action, n_hidden).to(device)
        self.q2 = SoftQNetwork(n_input, n_action, n_hidden).to(device)
        
        # Initialize target Q-networks
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        # Freeze target networks (no gradient computation)
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False
        
        # Initialize policy network (actor)
        self.policy = PolicyNetwork(n_input, n_action, n_hidden).to(device)
        
        # Initialize value network (optional, for stability)
        self.value = ValueNetwork(n_input, n_hidden).to(device)
        self.value_target = copy.deepcopy(self.value)
        for param in self.value_target.parameters():
            param.requires_grad = False
        
        # Initialize optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)
        
        # Entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            # Target entropy is -dim(A) by default
            if target_entropy is None:
                self.target_entropy = -n_action
            else:
                self.target_entropy = target_entropy
            
            # Log alpha for optimization (log for numerical stability)
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, 
                                         device=device, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(n_input, n_action, buffer_size, device)
        
        # Training state
        self.global_step = 0
        
    def np2torch(self, a, dtype=torch.float32):
        """Convert numpy array to torch tensor"""
        return torch.as_tensor(a, dtype=dtype, device=self.device)
    
    def select_action(self, state, evaluate=False):
        """
        Select action given state
        
        Args:
            state: Current state
            evaluate: If True, use deterministic action (mean)
            
        Returns:
            action: Numpy array of portfolio weights
        """
        # Flatten state
        if isinstance(state, np.ndarray):
            state = state.flatten()
        
        state_tensor = torch.as_tensor(state, dtype=torch.float32, 
                                        device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            if evaluate:
                # Deterministic action for evaluation
                action = self.policy.get_deterministic_action(state_tensor)
            else:
                # Stochastic action for exploration
                action, _, _ = self.policy.sample(state_tensor, self.device)
        
        return action.squeeze().cpu().numpy()
    
    def update(self):
        """
        Perform one SAC update step
        
        Returns:
            Dictionary of losses for logging
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # === Update Value Network ===
        # V(s) should fit: E_{a~pi}[ min(Q1(s,a), Q2(s,a)) - alpha * log pi(a|s) ]
        with torch.no_grad():
            # Sample actions from current policy for CURRENT states
            current_actions, current_log_probs, _ = self.policy.sample(states, self.device)

            # Compute Q-values for current state-action pairs
            current_q1 = self.q1(states, current_actions)
            current_q2 = self.q2(states, current_actions)
            current_q = torch.min(current_q1, current_q2)

            # Target value: Q(s,a) - alpha * log_pi(a|s)
            target_value = current_q - self.alpha * current_log_probs.unsqueeze(-1)
        
        # Current value estimate
        current_value = self.value(states)
        value_loss = F.mse_loss(current_value, target_value)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # === Update Q-Networks ===
        with torch.no_grad():
            # Use value target for computing Q targets
            target_v = self.value_target(next_states)
            q_target = rewards + self.gamma * (1 - dones) * target_v
        
        # Current Q estimates
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # === Update Policy Network ===
        # Sample new actions from current policy
        new_actions, log_probs, _ = self.policy.sample(states, self.device)
        
        # Compute Q-values for new actions
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Policy loss: maximize Q - alpha * log_pi
        # Equivalently: minimize alpha * log_pi - Q
        policy_loss = (self.alpha * log_probs.unsqueeze(-1) - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # === Update Alpha (entropy temperature) ===
        alpha_loss = None
        if self.auto_entropy_tuning:
            # Alpha loss: minimize -alpha * (log_pi + target_entropy)
            with torch.no_grad():
                _, log_probs_alpha, _ = self.policy.sample(states, self.device)
            
            alpha_loss = -(self.log_alpha * (log_probs_alpha + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # === Soft Update Target Networks ===
        self._soft_update(self.value, self.value_target)
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        
        # Return losses for logging
        losses = {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'alpha': self.alpha
        }
        if alpha_loss is not None:
            losses['alpha_loss'] = alpha_loss.item()
        
        return losses
    
    def _soft_update(self, source, target):
        """Soft update target network: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def learn(self, env, total_timesteps):
        """
        Main training loop
        
        Args:
            env: Training environment
            total_timesteps: Total number of environment steps
            
        Returns:
            episode_returns: List of returns per episode
            losses: List of loss dictionaries
            best_model_state: State dict of best model
        """
        episode_returns = []
        losses = []
        
        # Track best model
        best_avg_return = -np.inf
        best_model_state = None
        
        start_time = time.time()
        obs = env.reset()
        episode_return = 0
        episode_count = 0
        
        for step in range(total_timesteps):
            self.global_step = step
            
            # Select action
            if step < self.learning_starts:
                # Random action during warmup
                action = np.random.dirichlet(np.ones(self.n_action))
            else:
                action = self.select_action(obs, evaluate=False)
            
            # Take action in environment
            next_obs, reward, done = env.step(action)
            episode_return += reward
            
            # Handle terminal state
            if done and len(next_obs) == 0:
                # Environment returned empty state at episode end
                # Store with zeros for next_obs
                next_obs_flat = np.zeros_like(obs.flatten())
            else:
                next_obs_flat = next_obs.flatten()
            
            # Store transition in replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs_flat, done)
            
            # Update networks
            if step >= self.learning_starts and step % self.update_frequency == 0:
                loss_dict = self.update()
                if loss_dict is not None:
                    losses.append(loss_dict)
            
            # Logging
            if step % 1000 == 0:
                progress = (step / total_timesteps) * 100
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                print(f"Current Portfolio: {action}")
                print(f'[{progress:5.1f}%] Step {step:6d}: {steps_per_sec:4.0f} steps/s | '
                      f'Device: {self.device} | Alpha: {self.alpha:.4f}')
            
            # Episode done
            if done:
                episode_count += 1
                episode_returns.append(episode_return)
                print(f"global_step={step}, episode={episode_count}, episode_return={episode_return:.4f}")
                
                # Track best model
                if len(episode_returns) >= 10:
                    avg_return = np.mean(episode_returns[-10:])
                    
                    if avg_return > best_avg_return:
                        best_avg_return = avg_return
                        best_model_state = {
                            'policy_state_dict': {k: v.cpu().clone() 
                                                  for k, v in self.policy.state_dict().items()},
                            'q1_state_dict': {k: v.cpu().clone() 
                                             for k, v in self.q1.state_dict().items()},
                            'q2_state_dict': {k: v.cpu().clone() 
                                             for k, v in self.q2.state_dict().items()},
                            'value_state_dict': {k: v.cpu().clone() 
                                                for k, v in self.value.state_dict().items()},
                            'episode': episode_count,
                            'global_step': step,
                            'avg_return': avg_return,
                            'alpha': self.alpha,
                            'all_returns': episode_returns.copy()
                        }
                        # Save best model immediately
                        torch.save(best_model_state, "models/sac_portfolio_best.pth")
                        print(f"  ✓ New best model! Avg return (last 10 episodes): {avg_return:.4f}")
                        print(f"    → Saved to models/sac_portfolio_best.pth")
                
                # Reset episode
                episode_return = 0
                obs = env.reset()
            else:
                obs = next_obs
        
        return episode_returns, losses, best_model_state
    
    def choose_action_deterministic(self, observation):
        """Deterministic action selection for evaluation"""
        return self.select_action(observation, evaluate=True)
    
    def choose_action_stochastic(self, observation):
        """Stochastic action selection"""
        return self.select_action(observation, evaluate=False)
    
    def save_model(self, path):
        """Save the trained model"""
        state = {
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'value_target_state_dict': self.value_target.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha.item() if self.auto_entropy_tuning else None
        }
        torch.save(state, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle both old format (just policy dict) and new format (full state)
        if 'policy_state_dict' in state:
            self.policy.load_state_dict(state['policy_state_dict'])
            if 'q1_state_dict' in state:
                self.q1.load_state_dict(state['q1_state_dict'])
            if 'q2_state_dict' in state:
                self.q2.load_state_dict(state['q2_state_dict'])
            if 'value_state_dict' in state:
                self.value.load_state_dict(state['value_state_dict'])
            if 'alpha' in state:
                self.alpha = state['alpha']
        else:
            # Assume it's just the policy state dict
            self.policy.load_state_dict(state)
        
        print(f"Model loaded from {path}")
    
    def load_best_model(self, best_model_state):
        """Load the best model from training"""
        if best_model_state is not None:
            self.policy.load_state_dict(best_model_state['policy_state_dict'])
            self.q1.load_state_dict(best_model_state['q1_state_dict'])
            self.q2.load_state_dict(best_model_state['q2_state_dict'])
            self.value.load_state_dict(best_model_state['value_state_dict'])
            print(f"Best model loaded (Episode {best_model_state['episode']}, "
                  f"Avg Return: {best_model_state['avg_return']:.4f})")
        else:
            print("No best model state available")
