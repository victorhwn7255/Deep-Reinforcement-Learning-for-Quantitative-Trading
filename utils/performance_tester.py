import torch
import time

def test_tensor_operations(device, device_name):
    """
    Test basic tensor operations and measure performance
    """
    print("⚡ Performance Test:")
    print(f"  Testing on {device_name}...")
    
    try:
        # Create test tensors
        size = 1000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Test basic operations
        print(f"  ✓ Tensor creation successful")
        print(f"  ✓ Tensor shape: {a.shape}")
        print(f"  ✓ Tensor device: {a.device}")
        
        # Matrix multiplication benchmark
        start_time = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"  ✓ Matrix multiplication (10 runs): {avg_time:.4f}s avg")
        
        # Neural network layer test
        layer = torch.nn.Linear(1000, 500).to(device)
        output = layer(a)
        print(f"  ✓ Neural network layer test: {output.shape}")
        
        # Gradient computation test
        output.sum().backward()
        print(f"  ✓ Gradient computation successful")
        
    except Exception as e:
        print(f"  ❌ Error during testing: {e}")
    
    print()