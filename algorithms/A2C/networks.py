import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvantageActorCritic(nn.Module):
    def __init__(self, n_input, n_action, n_hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
        )

        # Actor (Policy) Head
        self.actor = nn.Linear(n_hidden, n_action)

        # Critic (Value Function) Head
        self.critic = nn.Linear(n_hidden, 1)

    def forward(self, x):
        features = self.network(x.reshape(1, -1))

        alpha = F.softplus(self.actor(features))  # Dirichlet parameter
        
        value = self.critic(features)  # State-value function

        return alpha, value