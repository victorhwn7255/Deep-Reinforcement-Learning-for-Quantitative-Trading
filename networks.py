import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta, Categorical, Dirichlet

########################
### Actor for Policy ###
########################
class ContinuousActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, learning_rate,
                 fc1_dims=128, fc2_dims=128, chkpt_dir='models/'):
        super(ContinuousActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_continuous_ppo')
        
        ### Network Architecture ###
        # State Input → FC1(128) → FC2(128) → Split into Alpha & Beta outputs
        # Input -> Hidden 1
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        # Hidden 2 -> Hidden 2
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # Hidden 2 - > Alpha params, Beta params
        self.alpha = nn.Linear(fc2_dims, n_actions)
        self.beta = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device(
            'cuda:0' if T.cuda.is_available() else 
            'mps' if T.backends.mps.is_available() else 
            'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        alpha = F.relu(self.alpha(x)) + 1.0
        beta = F.relu(self.beta(x)) + 1.0
        # dist = Beta(alpha, beta)
        if state.device.type == 'mps':
          alpha_cpu = alpha.cpu()
          beta_cpu = beta.cpu()
          dist = Beta(alpha_cpu, beta_cpu)
          return dist
        else:
          dist = Beta(alpha, beta)
          return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

#################################
### Critic for Value Function ###
#################################
class ContinuousCriticNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate,
                 fc1_dims=128, fc2_dims=128, chkpt_dir='models/'):
        super(ContinuousCriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_continuous_ppo')
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device(
            'cuda:0' if T.cuda.is_available() else 
            'mps' if T.backends.mps.is_available() else 
            'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        v = self.v(x)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))