import os
import torch as T
import torch.nn.functional as F
import numpy as np

from replay_buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent:
    def __init__(self, 
                 alpha,           # LR for Actor
                 beta,            # LR for Critics
                 input_dims,      # e.g. [state_dim] or [lag*features]     
                 tau,             # Polyak factor for target critics (e.g., 0.005–0.02)
                 env, 
                 env_id, 
                 gamma=0.99, 
                 max_size=1_000_000, 
                 layer1_size=256, 
                 layer2_size=256, 
                 batch_size=100, 
                 reward_scale=2, 
                 target_entropy=None, 
                 alpha_lr=3e-4,
                 policy_type = "normal", # "normal" -> "Normal Distribution"
                                         # "dirichlet" -> "Dirichlet Distribution"
                 
                 ## Action Space Dimension ##
                 n_actions=2,     # the Dimensionality of action vector
                                  # e.g, for Pendulum, n_actions = 1
                                  # e.g, for Car Sim, n_actions = 2 (steer, throttle)
                                  # e.g, for Portfolio Optimization, 
                                  # n_actions = num_of_stocks + 1 (for cash)
                 ):
        #######################
        ### Hyperparameters ###
        #######################
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale
        self.policy_type = policy_type.lower()
        
        #####################
        ### Replay Buffer ###
        #####################
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        
        ################
        ### NETWORKS ###
        ################
        # ACTOR
        self.actor = ActorNetwork(alpha, 
                                  input_dims, 
                                  layer1_size,
                                  layer2_size, 
                                  n_actions=n_actions,
                                  name=env_id+'_actor', 
                                  max_action=env.action_space.high
                                  )
        
        # CRITICS (trained by Gradient Descent every update)
        self.critic_1 = CriticNetwork(beta, 
                                      input_dims, 
                                      layer1_size,
                                      layer2_size, 
                                      n_actions=n_actions,
                                      name=env_id+'_critic_1'
                                      )
        self.critic_2 = CriticNetwork(beta, 
                                      input_dims, 
                                      layer1_size,
                                      layer2_size, 
                                      n_actions=n_actions,
                                      name=env_id+'_critic_2'
                                      )
        
        # TARGET CRITICS (slow moving with Soft updates via Polyak)
        self.critic_1_target = CriticNetwork(beta, 
                                             input_dims, 
                                             layer1_size, 
                                             layer2_size,
                                             n_actions=n_actions, 
                                             name=env_id + "_critic_1_target"
                                             )
        self.critic_2_target = CriticNetwork(beta, 
                                             input_dims, 
                                             layer1_size, 
                                             layer2_size,
                                             n_actions=n_actions, 
                                             name=env_id + "_critic_2_target"
                                             )
        # hard copy initial weights
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Freeze target critics parameters
        for p in self.critic_1_target.parameters():
            p.requires_grad = False
        for p in self.critic_2_target.parameters():
            p.requires_grad = False
        
        #################################
        ### Temperature α (learnable) ###
        #################################
        # default target entropy = -|A|
        # for Dirichlet (simplex), we may start with different target
        self.target_entropy = (
            -float(n_actions) if target_entropy is None else float(target_entropy)
        )
        
        device = self.critic_1.device  # use the same device as your critics
        
        self.log_alpha = T.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=alpha_lr)
        

    ###############################################
    ### Action selection for single observation ###
    ###############################################
    def choose_action(self, observation):
        # Convert to a (1, obs_dim...) float tensor on the actor's device
        if isinstance(observation, np.ndarray):
            state = T.from_numpy(observation).to(self.actor.device, dtype=T.float32).unsqueeze(0)
        elif T.is_tensor(observation):
            state = observation.to(self.actor.device, dtype=T.float32)
            if state.dim() == 1:  # ensure batch dim
                state = state.unsqueeze(0)
        else:
            state = T.as_tensor(observation, dtype=T.float32, device=self.actor.device).unsqueeze(0)

        with T.no_grad():
            if self.policy_type == "dirichlet":
                actions, _ = self.actor.sample_dirichlet(state, reparameterize=False)
            else:  # "normal"
                actions, _ = self.actor.sample_normal(state, reparameterize=False)

        # Return as numpy (remove batch dimension)
        return actions.squeeze(0).cpu().numpy()

    ###########################################
    ### Store a Transition in Replay Buffer ###
    ###########################################
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    ############################
    ### SAC v2 Training Loop ###
    ############################
    ###   1) Critic update (to TD target using target critics)
    ###   2) Actor update (pathwise; minimize α logπ - min(Q))
    ###   3) Temperature α update (toward target_entropy)
    ###   4) Soft update target critics
    def learn(self):
        # Ensure enough samples
        if self.memory.mem_cntr < self.batch_size:
            return
          
        ###########################
        ### Sample a Mini Batch ###
        ###########################
        s_np, a_np, r_np, s2_np, d_np = self.memory.sample_buffer(self.batch_size)
        
        device = self.critic_1.device
        
        s   = T.as_tensor(s_np,  dtype=T.float32, device=device)
        a   = T.as_tensor(a_np,  dtype=T.float32, device=device)
        r   = T.as_tensor(r_np,  dtype=T.float32, device=device)
        d   = T.as_tensor(d_np,  dtype=T.bool,    device=device)
        s2  = T.as_tensor(s2_np, dtype=T.float32, device=device)

        # temperature hyperparameter
        alpha = self.log_alpha.exp()

        #########################
        ### CRITIC TARGET (y) ###
        #########################
        # y = r + γ (1 - done) [ min(Q_tgt(s', a')) - α logπ(a'|s') ]
        with T.no_grad():
            if self.policy_type == "dirichlet":
                a2, logp2 = self.actor.sample_dirichlet(s2, reparameterize=False)
            else:
                a2, logp2 = self.actor.sample_normal(s2, reparameterize=False)

            q1_t = self.critic_1_target(s2, a2).view(-1)
            q2_t = self.critic_2_target(s2, a2).view(-1)
            q_t_min = T.min(q1_t, q2_t)
            
            # (~d) is True for non-terminal transitions
            # if done=True → (~d)=False → future value = 0
            # if done=False → (~d)=True → include future value
            # q_t_min: Expected future reward
            # alpha * logp2: Entropy bonus (encourages exploration)
            y = self.scale * r + self.gamma * (~d) * (q_t_min - alpha * logp2.view(-1))
        
        #########################
        ### UPDATE CRITIC (Q) ###
        #########################
        # minimize MSE((Q1(s,a), y)) + MSE((Q2(s,a), y))
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1 = self.critic_1(s, a).view(-1)
        q2 = self.critic_2(s, a).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1, y)
        critic_2_loss = 0.5 * F.mse_loss(q2, y)
        critic_loss = critic_1_loss + critic_2_loss

        critic_loss.backward()
        #T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        #T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()      
        
        ########################
        ### UPDATE ACTOR (π) ###
        ########################
        # minimize E[ α logπ - min(Q1,Q2)(s, a_pi) ]
        self.actor.optimizer.zero_grad()

        if self.policy_type == "dirichlet":
            a_pi, logp_pi = self.actor.sample_dirichlet(s, reparameterize=True)
        else:
            a_pi, logp_pi = self.actor.sample_normal(s, reparameterize=True)

        q1_pi = self.critic_1(s, a_pi).view(-1)
        q2_pi = self.critic_2(s, a_pi).view(-1)
        q_pi_min = T.min(q1_pi, q2_pi)
        
        # Detach alpha in the actor loss; alpha is updated by its own objective
        actor_loss = (alpha.detach() * logp_pi.view(-1) - q_pi_min).mean()

        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor.optimizer.step()
        
        ##############################
        ### UPDATE TEMPERATURE (α) ###
        ##############################
        # minimize J(α) = E[ -α (logπ + target_entropy) ]
        # logp_pi = log probability of the sampled action
        # log_alpha = learnable log temperature parameter
        # Gradient flow (forward): Actor → logp_pi → alpha_loss → log_alpha
        # detach gradient from logp_pi, so when we do backward(), the gradient doesn't flow into Actor
        alpha_loss = -(self.log_alpha * (logp_pi.detach().view(-1) + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        
        #################################
        ### SOFT UPDATE CRITIC TARGET ###
        #################################
        self.soft_update_targets()
        
        #return scalars for logging
        return {
            "alpha": float(alpha.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "critic_loss": float(critic_loss.item())
        }

    def soft_update_targets(self, tau=None):
        if tau is None:
            tau = self.tau

        with T.no_grad():
            for p_t, p in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)
            for p_t, p in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    def save_models(self):
        print(".... saving models ....")
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        # save targets too (optional, but useful for exact restarts)
        T.save(self.critic_1_target.state_dict(), self.critic_1.checkpoint_file + "_target")
        T.save(self.critic_2_target.state_dict(), self.critic_2.checkpoint_file + "_target")
        # temperature
        T.save({"log_alpha": self.log_alpha.detach().cpu()}, "alpha_checkpoint.pt")

    def load_models(self):
        print(".... loading models ....")
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        # load targets if present; else hard-copy from online
        try:
            self.critic_1_target.load_state_dict(T.load(self.critic_1.checkpoint_file + "_target", map_location=self.critic_1.device))
            self.critic_2_target.load_state_dict(T.load(self.critic_2.checkpoint_file + "_target", map_location=self.critic_2.device))
        except Exception:
            self.critic_1_target.load_state_dict(self.critic_1.state_dict())
            self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        # load alpha
        try:
            payload = T.load("alpha_checkpoint.pt", map_location=self.critic_1.device)
            self.log_alpha = payload["log_alpha"].to(self.critic_1.device)
            self.log_alpha.requires_grad_(True)
        except Exception:
            pass