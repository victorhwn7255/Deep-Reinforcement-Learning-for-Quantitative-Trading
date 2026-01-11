from dataclasses import dataclass
from typing import List
import torch

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    # Market data
    tickers: List[str] = None
    start_date: str = '2010-01-01'
    end_date: str = '2024-12-31'
    train_split_ratio: float = 0.8

    # VIX data paths
    vix_path: str = '../../data/VIX_CLS_2010_2024.csv'
    vix3m_path: str = '../../data/VIX3M_CLS_2010_2024.csv'
    credit_spread_path: str = '../../data/Credit_Spread_2010_2024.csv'

    # Feature engineering parameters
    rsi_period: int = 14
    volatility_window: int = 20

    # VIX feature parameters
    vix_baseline: float = 20.0  # Used for normalization
    vix_regime_low: float = 15.0
    vix_regime_high: float = 25.0

    # Credit spread feature parameters
    credit_baseline: float = 2.0
    credit_lookback: int = 252
    credit_regime_low: float = 1.5
    credit_regime_high: float = 3.5

    def __post_init__(self):
        """Set default tickers if not provided."""
        if self.tickers is None:
            self.tickers = ['VNQ', 'SPY', 'TLT', 'GLD', 'BTC-USD']


@dataclass
class EnvironmentConfig:
    """Trading environment configuration."""

    # State construction
    lag: int = 5  # Lookback window for state
    include_position_in_state: bool = True

    # Transaction costs
    tc_rate: float = 0.0005  # 5 bps per unit turnover
    tc_fixed: float = 0.0  # Fixed cost per rebalance event
    turnover_threshold: float = 0.0  # Minimum turnover to trigger costs

    # Turnover calculation
    turnover_include_cash: bool = False
    turnover_use_half_factor: bool = True  # Avoid double-counting

    # Reward scaling (stability)
    reward_scale: float = 100.0  # Scale log returns for better SAC training


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""

    # Architecture
    n_hidden: int = 256  # Hidden layer size for all networks

    # Policy network (Dirichlet)
    alpha_min: float = 0.6  # Minimum concentration parameter (gradient stability)
    alpha_max: float = 100.0  # Maximum concentration parameter
    action_eps: float = 1e-8  # Small epsilon to avoid log(0)


@dataclass
class SACConfig:
    """SAC algorithm hyperparameters."""

    # Core SAC parameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update coefficient for target networks
    learning_rate: float = 0.001  # Learning rate for all networks

    # Entropy tuning
    alpha: float = 0.2  # Initial entropy coefficient
    auto_entropy_tuning: bool = True  # Automatically tune alpha
    target_entropy: float = None  # If None, computed from Dirichlet max entropy
    target_entropy_margin: float = 0.5  # Margin below max entropy (in nats)

    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256

    # Training schedule
    learning_starts: int = 3900  # Warmup steps before training
    update_frequency: int = 1  # How often to update networks (in steps)


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Training duration
    total_timesteps: int = 1_500_000

    # Logging and checkpointing
    log_interval: int = 10  # Log every N episodes
    save_interval: int = 50  # Save checkpoint every N episodes

    # Paths
    model_dir: str = "models"
    model_path_final: str = "models/sac_portfolio_final.pth"
    model_path_best: str = "models/sac_portfolio_best.pth"

    # Device preference (will be auto-detected if None)
    device: str = None  # 'cuda', 'cpu', 'mps', or None for auto


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    data: DataConfig = None
    env: EnvironmentConfig = None
    network: NetworkConfig = None
    sac: SACConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        """Initialize all sub-configs if not provided."""
        if self.data is None:
            self.data = DataConfig()
        if self.env is None:
            self.env = EnvironmentConfig()
        if self.network is None:
            self.network = NetworkConfig()
        if self.sac is None:
            self.sac = SACConfig()
        if self.training is None:
            self.training = TrainingConfig()

    def auto_detect_device(self) -> torch.device:
        """Auto-detect and return the best available device."""
        if self.training.device is not None:
            return torch.device(self.training.device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            # MPS has issues with Dirichlet gradients - use CPU for training
            print("⚠ WARNING: Apple Silicon (MPS) detected.")
            print("  Using CPU to avoid MPS-specific issues with Dirichlet.")
            return torch.device("cpu")
        else:
            return torch.device("cpu")

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)

        print("\nDATA CONFIG:")
        print(f"  Tickers: {self.data.tickers}")
        print(f"  Date range: {self.data.start_date} to {self.data.end_date}")
        print(f"  Train/test split: {self.data.train_split_ratio:.1%}")

        print("\nENVIRONMENT CONFIG:")
        print(f"  Lag: {self.env.lag}")
        print(f"  TC rate: {self.env.tc_rate:.6f} ({self.env.tc_rate * 10000:.1f} bps)")
        print(f"  Reward scale: {self.env.reward_scale:.1f}")
        print(f"  Include position in state: {self.env.include_position_in_state}")

        print("\nNETWORK CONFIG:")
        print(f"  Hidden size: {self.network.n_hidden}")
        print(f"  Alpha min/max: {self.network.alpha_min:.1f} / {self.network.alpha_max:.1f}")

        print("\nSAC CONFIG:")
        print(f"  Gamma: {self.sac.gamma:.3f}")
        print(f"  Tau: {self.sac.tau:.4f}")
        print(f"  Learning rate: {self.sac.learning_rate:.4f}")
        print(f"  Batch size: {self.sac.batch_size}")
        print(f"  Buffer size: {self.sac.buffer_size:,}")
        print(f"  Auto entropy tuning: {self.sac.auto_entropy_tuning}")

        print("\nTRAINING CONFIG:")
        print(f"  Total timesteps: {self.training.total_timesteps:,}")
        print(f"  Learning starts: {self.sac.learning_starts:,}")
        print(f"  Model path: {self.training.model_path_final}")

        print("=" * 80)


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

def get_default_config() -> Config:
    """Get default configuration for training."""
    return Config()


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

def get_low_cost_config() -> Config:
    """Configuration for low transaction cost environment."""
    config = get_default_config()
    config.env.tc_rate = 0.0001  # 1 bps
    return config


def get_high_cost_config() -> Config:
    """Configuration for high transaction cost environment."""
    config = get_default_config()
    config.env.tc_rate = 0.002  # 20 bps
    config.env.tc_fixed = 0.0001  # Fixed cost penalty
    return config


def get_conservative_config() -> Config:
    """Configuration for more conservative, stable policy."""
    config = get_default_config()
    config.network.alpha_min = 1.0  # Force more diversified portfolios
    config.sac.target_entropy_margin = 0.2  # Higher entropy target
    return config


def get_fast_training_config() -> Config:
    """Configuration for faster training (smaller buffer, higher LR)."""
    config = get_default_config()
    config.sac.buffer_size = 100_000
    config.sac.learning_rate = 0.003
    config.sac.batch_size = 512
    return config


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Default config
    config = get_default_config()
    config.print_summary()

    print("\n\nExample: Accessing specific parameters:")
    print(f"Learning rate: {config.sac.learning_rate}")
    print(f"Tickers: {config.data.tickers}")
    print(f"Reward scale: {config.env.reward_scale}")

    # Example 2: Custom config
    print("\n" + "=" * 80)
    print("CUSTOM CONFIGURATION")
    print("=" * 80)
    custom_config = Config(
        data=DataConfig(
            tickers=['SPY', 'TLT', 'GLD'],
            start_date='2015-01-01',
        ),
        sac=SACConfig(
            learning_rate=0.0005,
            batch_size=128,
        ),
    )
    custom_config.print_summary()
