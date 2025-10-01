import torch
from torch.distributions import Beta

def test_ppo_network_compatibility():
    """
    Test if PPO networks work with selected device
    """
    print("🤖 PPO Network Compatibility Test:")
    
    # Get optimal device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    try:
        # Simulate PPO network components
        input_dims = (10,)  # State dimension
        n_actions = 3       # Action dimension
        
        # Test Actor Network components
        fc1 = torch.nn.Linear(input_dims[0], 128).to(device)
        fc2 = torch.nn.Linear(128, 128).to(device)
        alpha_head = torch.nn.Linear(128, n_actions).to(device)
        beta_head = torch.nn.Linear(128, n_actions).to(device)
        
        # Test forward pass
        state = torch.randn(1, input_dims[0], device=device)
        x = torch.tanh(fc1(state))
        x = torch.tanh(fc2(x))
        alpha = torch.nn.functional.relu(alpha_head(x)) + 1.0
        beta = torch.nn.functional.relu(beta_head(x)) + 1.0
        
        # Test Beta distribution
        dist = Beta(alpha, beta)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        print(f"  ✓ Actor network test passed")
        print(f"  ✓ Device: {device}")
        print(f"  ✓ Action shape: {action.shape}")
        print(f"  ✓ Log probability: {log_prob.shape}")
        
        # Test Critic Network
        critic = torch.nn.Sequential(
            torch.nn.Linear(input_dims[0], 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1)
        ).to(device)
        
        value = critic(state)
        print(f"  ✓ Critic network test passed")
        print(f"  ✓ Value shape: {value.shape}")
        
    except Exception as e:
        print(f"  ❌ PPO network test failed: {e}")
    
    print()