import torch

def comprehensive_device_test():
    """
    Comprehensive test for GPU (CUDA), MPS (Apple Silicon), and CPU devices
    """
    print("=" * 60)
    print("PYTORCH DEVICE COMPATIBILITY TEST")
    print("=" * 60)
    
    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # Test CUDA (NVIDIA GPU)
    print("🔧 CUDA (NVIDIA GPU) Test:")
    print(f"  ✓ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA Version: {torch.version.cuda}")
        print(f"  ✓ GPU Count: {torch.cuda.device_count()}")
        print(f"  ✓ Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ❌ CUDA not available")
    print()
    
    # Test MPS (Apple Silicon)
    print("🍎 MPS (Apple Silicon) Test:")
    print(f"  ✓ MPS Available: {torch.backends.mps.is_available()}")
    print(f"  ✓ MPS Built: {torch.backends.mps.is_built()}")
    if torch.backends.mps.is_available():
        print("  ✓ MPS ready for acceleration!")
    else:
        print("  ❌ MPS not available")
    print()
    
    # Test CPU
    print("💻 CPU Test:")
    print(f"  ✓ CPU Threads: {torch.get_num_threads()}")
    print(f"  ✓ CPU Count: {torch.get_num_interop_threads()}")
    print()
    
    # Device Selection Logic
    print("🎯 Device Selection:")
    if torch.cuda.is_available():
        selected_device = torch.device('cuda:0')
        device_type = "CUDA GPU"
    elif torch.backends.mps.is_available():
        selected_device = torch.device('mps')
        device_type = "MPS (Apple Silicon)"
    else:
        selected_device = torch.device('cpu')
        device_type = "CPU"
    
    print(f"  ✓ Selected Device: {selected_device} ({device_type})")
    print()
    
    return selected_device, device_type