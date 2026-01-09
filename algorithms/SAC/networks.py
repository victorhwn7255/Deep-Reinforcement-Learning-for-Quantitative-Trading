import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


class SoftQNetwork(nn.Module):
    """
    Soft Q-Network (Critic) for SAC
    
    Estimates Q(s, a) - the expected return starting from state s,
    taking action a, and following the policy thereafter.
    
    SAC uses two Q-networks to mitigate overestimation bias.
    """
    
    def __init__(self, n_input, n_action, n_hidden=256):
        """
        Initialize Q-network
        
        Args:
            n_input: State dimension
            n_action: Action dimension
            n_hidden: Number of hidden units
        """
        super().__init__()
        
        # Q-network takes state-action pair as input
        self.network = nn.Sequential(
            nn.Linear(n_input + n_action, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
                
    def forward(self, state, action):
        """
        Forward pass
        
        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            
        Returns:
            Q-value tensor (batch_size, 1)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class PolicyNetwork(nn.Module):
    """
    Policy Network (Actor) for SAC with Dirichlet Distribution
    
    Outputs parameters of a Dirichlet distribution for portfolio weights.
    The Dirichlet distribution naturally ensures weights sum to 1.
    
    Uses the reparameterization trick for gradient estimation:
    - Sample from a simpler distribution (Gamma)
    - Transform to get Dirichlet samples
    """
    
    def __init__(self, n_input, n_action, n_hidden=256, log_std_min=-20, log_std_max=2):
        """
        Initialize policy network
        
        Args:
            n_input: State dimension
            n_action: Action dimension (number of assets + cash)
            n_hidden: Number of hidden units
            log_std_min: Minimum log standard deviation (for numerical stability)
            log_std_max: Maximum log standard deviation
        """
        super().__init__()
        
        self.n_action = n_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )
        
        # Output Dirichlet concentration parameters
        # Use softplus to ensure positivity, then add small constant
        self.alpha_head = nn.Linear(n_hidden, n_action)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        # Initialize alpha head with smaller weights for stability
        nn.init.xavier_uniform_(self.alpha_head.weight, gain=0.1)
        nn.init.constant_(self.alpha_head.bias, 1.0)  # Start with uniform-ish
        
    def forward(self, state):
        """
        Forward pass - compute Dirichlet concentration parameters
        
        Args:
            state: State tensor
            
        Returns:
            alpha: Dirichlet concentration parameters (batch_size, n_action)
        """
        features = self.feature_net(state)
        
        # Ensure alpha > 0 (required for Dirichlet)
        # softplus(x) = log(1 + exp(x)) ensures smooth positive output
        alpha = F.softplus(self.alpha_head(features)) + 0.1
        
        return alpha
    
    def sample(self, state, device='cpu'):
        """
        Sample action from policy (reparameterized)
        
        Uses the fact that if X_i ~ Gamma(alpha_i, 1), then
        X / sum(X) ~ Dirichlet(alpha)
        
        Args:
            state: State tensor
            device: Device for computation
            
        Returns:
            action: Sampled action (portfolio weights)
            log_prob: Log probability of action
            alpha: Dirichlet parameters
        """
        alpha = self.forward(state)
        
        # Create Dirichlet distribution
        # Move to CPU for Dirichlet (MPS not supported)
        alpha_cpu = alpha.cpu()
        dist = Dirichlet(alpha_cpu)
        
        # Sample using reparameterization
        action_cpu = dist.rsample()
        
        # Compute log probability
        log_prob_cpu = dist.log_prob(action_cpu)
        
        # Move back to original device
        action = action_cpu.to(device)
        log_prob = log_prob_cpu.to(device)
        
        return action, log_prob, alpha
    
    def evaluate(self, state, action, device='cpu'):
        """
        Evaluate log probability of an action given state
        
        Args:
            state: State tensor
            action: Action tensor (portfolio weights)
            device: Device for computation
            
        Returns:
            log_prob: Log probability of action
            entropy: Entropy of distribution
        """
        alpha = self.forward(state)
        
        # Create Dirichlet distribution
        alpha_cpu = alpha.cpu()
        action_cpu = action.cpu()
        
        dist = Dirichlet(alpha_cpu)
        
        # Clamp action slightly away from 0 and 1 for numerical stability
        action_clamped = torch.clamp(action_cpu, min=1e-6, max=1.0 - 1e-6)
        # Renormalize to sum to 1
        action_clamped = action_clamped / action_clamped.sum(dim=-1, keepdim=True)
        
        log_prob_cpu = dist.log_prob(action_clamped)
        entropy_cpu = dist.entropy()
        
        log_prob = log_prob_cpu.to(device)
        entropy = entropy_cpu.to(device)
        
        return log_prob, entropy
    
    def get_deterministic_action(self, state):
        """
        Get deterministic action (mean of Dirichlet)
        
        For evaluation/deployment where we want reproducible behavior.
        
        Args:
            state: State tensor
            
        Returns:
            action: Mean portfolio weights (alpha / sum(alpha))
        """
        alpha = self.forward(state)
        # Mean of Dirichlet(alpha) is alpha / sum(alpha)
        action = alpha / alpha.sum(dim=-1, keepdim=True)
        return action


class ValueNetwork(nn.Module):
    """
    Value Network for SAC (optional)
    
    Estimates V(s) - the expected return starting from state s.
    While the original SAC paper uses a separate value network,
    later versions compute V from Q and the policy.
    
    Including it here for stability, as mentioned in the paper.
    """
    
    def __init__(self, n_input, n_hidden=256):
        """
        Initialize value network
        
        Args:
            n_input: State dimension
            n_hidden: Number of hidden units
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
                
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            V-value tensor
        """
        return self.network(state)


# For backward compatibility with the naming convention
class TwinQNetwork(nn.Module):
    """
    Twin Q-Networks for SAC
    
    Wraps two Q-networks and provides convenient access to both.
    """
    
    def __init__(self, n_input, n_action, n_hidden=256):
        super().__init__()
        self.q1 = SoftQNetwork(n_input, n_action, n_hidden)
        self.q2 = SoftQNetwork(n_input, n_action, n_hidden)
        
    def forward(self, state, action):
        """
        Get Q-values from both networks
        
        Returns:
            q1_value, q2_value
        """
        return self.q1(state, action), self.q2(state, action)
    
    def q1_forward(self, state, action):
        """Get Q-value from first network only"""
        return self.q1(state, action)
    
    def q2_forward(self, state, action):
        """Get Q-value from second network only"""
        return self.q2(state, action)
