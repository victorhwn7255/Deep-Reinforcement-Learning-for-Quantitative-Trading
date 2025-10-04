import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet
import numpy as np


########################
### Critic (Q-value) ###
########################
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir='models/'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac_critic')

        #####################
        ### NETWORK LAYER ###
        #####################
        ### nn.Linear(num of inputs, num of outputs)    
        # Handle multi-dimensional state spaces (1D, 2D, etc.)
        self.fc1 = nn.Linear(np.prod(self.input_dims) + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)
        self.q2 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        if len(state.shape) > 2:
            state = state.view(state.size(0), -1)
            
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q1 = self.q1(action_value)
        q2 = self.q2(action_value)

        return q1, q2

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


######################
### Actor (Policy) ###
######################
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_action,
                 n_actions, name, chkpt_dir='models/'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac_actor')
        self.reparam_noise = 1e-6
        
        #####################
        ### NETWORK LAYER ###
        #####################
        ### nn.Linear(num of inputs, num of outputs)    
        self.fc1 = nn.Linear(np.prod(self.input_dims), self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Gaussian policy
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        # Dirichlet policy
        self.alpha_head = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward_normal(self, state):
        if len(state.shape) > 2:
            state = state.view(state.size(0), -1)
            
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        mu = self.mu(prob)

        log_sigma = self.sigma(prob)
        log_sigma = T.clamp(log_sigma, min=-20, max=2)
        sigma = log_sigma.exp()

        return mu, sigma

    def forward_dirichlet(self, state):
        if len(state.shape) > 2:
            state = state.view(state.size(0), -1)
            
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        alpha = self.alpha_head(prob)
        alpha = F.softplus(alpha) + 1e-6
        return alpha

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward_normal(state)
        dist = T.distributions.Normal(mu, sigma)

        actions = dist.rsample() if reparameterize else dist.sample()
        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)

        log_probs = dist.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def sample_dirichlet(self, state, reparameterize=True):
        alpha = self.forward_dirichlet(state)
        dist = T.distributions.Dirichlet(alpha)

        action = dist.rsample() if reparameterize else dist.sample()
        log_probs = dist.log_prob(action).unsqueeze(1)

        return action, log_probs

    # EXAMPLE
    # state.shape = [batch_size, state_dim(number of features)], eg [32,10]
    # action.shape = [batch_size, n_actions(number of stocks)], eg [32, 4]
    # log_probs.shape = [batch_size, 1], eg [32, 1]
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


####################
### Value (V(s)) ###
#################### 
class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, name, chkpt_dir='models/'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac_v')

        self.fc1 = nn.Linear(np.prod(self.input_dims), self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        if len(state.shape) > 2:
            state = state.view(state.size(0), -1)
            
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))