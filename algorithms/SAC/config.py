from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import random

import numpy as np
import torch


# =============================================================================
# CONFIG STRUCTURE
# =============================================================================

@dataclass
class ExperimentConfig:
    """Experiment / reproducibility knobs."""
    seed: int = 42
    run_name: str = "sac_portfolio"
    output_dir: str = "runs"          # where to store run artifacts (relative path)
    save_resolved_config: bool = True # write config.json next to checkpoints
    verbose: bool = True


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    tickers: List[str] = field(default_factory=lambda: ["VNQ", "SPY", "TLT", "GLD", "BTC-USD"])
    start_date: str = "2010-01-01"
    end_date: str = "2024-12-31"
    train_split_ratio: float = 0.8

    # External / macro data (relative paths are intentionally allowed)
    vix_path: str = "../../data/VIX_CLS_2010_2024.csv"
    vix3m_path: str = "../../data/VIX3M_CLS_2010_2024.csv"
    credit_spread_path: str = "../../data/CREDIT_SPREAD_2010_2024.csv"
    yield_curve_path: str = "../../data/YIELD_CURVE_10Y3M_2010_2024.csv"
    dxy_path: str = "../../data/DOLLAR_INDEX_2010_2024.csv"

    # Column names (to tolerate different FRED / vendor exports)
    vix_col_candidates: List[str] = field(default_factory=lambda: ["VIXCLS", "VIX"])
    vix3m_col_candidates: List[str] = field(default_factory=lambda: ["VXVCLS", "VIX3M"])
    credit_col_candidates: List[str] = field(default_factory=lambda: ["Credit_Spread", "CREDIT_SPREAD", "credit_spread", "spread"])
    yieldcurve_col_candidates: List[str] = field(default_factory=lambda: [ "T10Y3M", "10Y3M", "10Y_3M", "YieldCurve", "yield_curve", "slope"])
    dxy_col_candidates: List[str] = field(default_factory=lambda: ["DTWEXBGS", "DXY", "Dollar_Index", "USD"])

    # How to align & fill macro series
    macro_join_how: str = "left"
    macro_ffill: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering knobs used to build environment state."""
    # Per-asset features (the environment will build columns as f"{ticker}_{name}")
    per_asset_feature_names: List[str] = field(default_factory=lambda: ["RSI", "volatility"])

    # Technical feature parameters
    rsi_period: int = 14
    volatility_window: int = 20

    # VIX features
    vix_baseline: float = 20.0
    vix_regime_low: float = 15.0
    vix_regime_high: float = 25.0
    vix_term_structure_clip: float = 1.0

    # Credit spread features (values are typically in decimals; e.g. 0.02 = 2%)
    credit_baseline: float = 0.02
    credit_regime_low: float = 0.02
    credit_regime_high: float = 0.04
    credit_momentum_window: int = 30
    credit_zscore_window: int = 252
    credit_divergence_window: int = 60
    credit_velocity_lag: int = 5
    credit_zscore_clip: float = 3.0
    credit_momentum_clip: float = 1.0
    credit_velocity_clip: float = 1.0
    credit_divergence_clip: float = 3.0

    # Macro feature column names (environment expects these exact names)
    macro_feature_columns: List[str] = field(default_factory=lambda: [
        # VIX
        "VIX_normalized",
        "VIX_regime",
        "VIX_term_structure",
        # Credit Spread
        "Credit_Spread_normalized",
        "Credit_Spread_regime",
        "Credit_Spread_momentum",
        "Credit_Spread_zscore",
        "Credit_Spread_velocity",
        "Credit_VIX_divergence",
        # Yield Curve
        "YieldCurve_10Y3M",
        "YieldCurve_10Y3M_change",
    ])
    
    # -------------------------
    # Regime HMM 
    # -------------------------
    use_regime_hmm: bool = False

    # Names of probability columns we will append to df
    regime_prob_columns: List[str] = field(default_factory=lambda: [
        "RegimeP_stable",
        "RegimeP_trans",
        "RegimeP_crisis",
    ])

    # HMM observation construction
    hmm_obs_ticker: str = "SPY"      # use SPY returns as the core
    hmm_include_rvol: bool = False   # include realized vol (redundant with VIX, disabled by default)
    hmm_rvol_window: int = 20        # realized vol window (only used if hmm_include_rvol=True)
    hmm_include_vix: bool = True
    hmm_include_credit_spread: bool = True
    hmm_include_vix_term: bool = True  # uses VIX term structure if available
    hmm_include_yc_change: bool = True  # include Δ(T10Y3M) over N days
    hmm_yc_change_lag: int = 5          # N days for yield curve change
    hmm_include_dxy: bool = True        # include DXY log return

    # HMM fit knobs
    hmm_n_states: int = 3
    hmm_n_iter: int = 50
    hmm_tol: float = 1e-4
    hmm_min_var: float = 1e-4
    
    # Yield curve feature parameters
    # Raw T10Y3M is in percentage points (e.g., 3.77). Typical range ~[-3, +4].
    yield_curve_slope_scale: float = 3.0
    yield_curve_change_lag: int = 5
    yield_curve_change_scale: float = 1.0


@dataclass
class EnvironmentConfig:
    """Trading environment configuration."""
    lag: int = 5
    include_position_in_state: bool = True

    # Transaction costs (one-way turnover cost)
    tc_rate: float = 0.0001          # 1 bps per unit one-way turnover
    tc_fixed: float = 0.0            # fixed cost per rebalance
    turnover_threshold: float = 0.0  # ignore tiny rebalances
    turnover_include_cash: bool = False
    turnover_use_half_factor: bool = True

    # Reward shaping
    reward_type: str = "linear"        # "log" (Risk-averse), 
                                    # "linear" (Risk-neutral), 
                                    # "sharpe", (Risk-adjusted)
                                    # "exp" (Risk-seeking)
    reward_scale: float = 10.0
    reward_clip_min: float = -0.999    # net_return clip BEFORE reward calculation
    reward_clip_max: float = 5.0       # allow agent have 300% gain

    # For exponential reward (risk-seeking)
    exp_risk_factor: float = 6.0       # higher = more risk-seeking (2, 3, 5, 6)

    # For Sharpe-style reward
    sharpe_window: int = 20            # rolling window for volatility calculation
    risk_free_rate: float = 0.0        # annualized risk-free rate (e.g., 0.02 = 2%)

    # Terminal handling (portfolio tasks are usually time-truncated, not terminal)
    treat_done_as_truncation: bool = True

    # Optional constraints (future-proofing)
    allow_short: bool = False
    allow_leverage: bool = False
    max_gross_exposure: float = 1.0  # ignored unless leverage enabled

    def build_feature_columns(self, tickers: List[str], features: FeatureConfig) -> List[str]:
        cols: List[str] = []
        for t in tickers:
            for name in features.per_asset_feature_names:
                cols.append(f"{t}_{name}")

        cols.extend(features.macro_feature_columns)

        if getattr(features, "use_regime_hmm", False):
            cols.extend(list(getattr(features, "regime_prob_columns", [])))

        return cols

@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    hidden_size: int = 128
    num_layers: int = 2
    activation: str = "relu"         
    layer_norm: bool = True
    dropout: float = 0.0
    weight_decay: float = 0.0

    # Dirichlet policy safety
    alpha_min: float = 0.1
    alpha_max: float = 60.0
    action_eps: float = 1e-8

@dataclass
class SACConfig:
    """SAC algorithm hyperparameters."""
    gamma: float = 0.99
    tau: float = 0.005

    # Optimizers
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    # Entropy / temperature
    init_alpha: float = 0.1
    auto_entropy_tuning: bool = True

    # If set, overrides auto target entropy
    target_entropy: Optional[float] = None

    # Dirichlet-native target entropy:
    # We define target_entropy = H(Dirichlet([c]*K)) - margin
    dirichlet_entropy_concentration: float = 1.0

    # subtract a small margin to encourage slightly less randomness (0.0 is fine too)
    target_entropy_margin: float = 1.25

    # Replay / updates
    buffer_size: int = 420_000
    batch_size: int = 256
    learning_starts: int = 5000
    update_frequency: int = 1
    updates_per_step: int = 1

    # Stability
    gradient_clip_norm: float = 1.0

@dataclass
class TrainingConfig:
    """Training loop configuration."""
    total_timesteps: int = 690_000
    log_interval_episodes: int = 10
    save_interval_episodes: int = 0  # 0 = disabled (only save best + final)

    # Checkpoint paths (relative paths)
    model_dir: str = "models"
    model_path_final: str = "models/sac_portfolio_final.pth"
    model_path_best: str = "models/sac_portfolio_best.pth"

    # Resume / warm start
    resume_from: Optional[str] = None  # path to checkpoint to resume from

    # Device preference
    device: Optional[str] = None  # "cuda" | "cpu" | "mps" | None


@dataclass
class EvaluationConfig:
    """Backtest / evaluation knobs."""
    model_path: str = "models/sac_portfolio_best.pth"
    deterministic: bool = True
    render_plots: bool = True
    save_plots: bool = True
    output_dir: str = "eval_outputs"


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # -------------------------
    # Convenience helpers
    # -------------------------
    def auto_detect_device(self) -> torch.device:
        """Pick the best device; avoid MPS for Dirichlet gradients."""
        if self.training.device is not None:
            return torch.device(self.training.device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            # Dirichlet rsample/log_prob gradients are often problematic on MPS
            if self.experiment.verbose:
                print("⚠ WARNING: Apple Silicon (MPS) detected.")
                print("  Using CPU to avoid MPS issues with Dirichlet gradients.")
            return torch.device("cpu")
        return torch.device("cpu")

    def compute_target_entropy(self, n_action: int) -> float:
        """Dirichlet-native target entropy (in the same units as Dirichlet.entropy()).

        If user provides sac.target_entropy, use it.
        Otherwise, set target to entropy of symmetric Dirichlet([c]*K) minus a margin.
        """
        if self.sac.target_entropy is not None:
            return float(self.sac.target_entropy)

        K = int(max(2, n_action))
        c = float(getattr(self.sac, "dirichlet_entropy_concentration", 1.0))
        margin = float(getattr(self.sac, "target_entropy_margin", 0.0))

        from torch.distributions import Dirichlet
        alpha = torch.full((K,), c, dtype=torch.float32)
        H = float(Dirichlet(alpha).entropy().item())
        return float(H - margin)


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @staticmethod
    def load_json(path: str) -> "Config":
        """
        Load config from JSON produced by Config.save_json().

        This reconstructs nested dataclasses properly:
        - experiment: ExperimentConfig
        - data: DataConfig
        - features: FeatureConfig
        - env: EnvironmentConfig
        - network: NetworkConfig
        - sac: SACConfig
        - training: TrainingConfig
        - evaluation: EvaluationConfig

        It also ignores unknown keys for forward/backward compatibility.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Config JSON must be an object/dict, got: {type(raw)}")

        def build_section(cls, key: str):
            section = raw.get(key, {})
            if section is None:
                section = {}
            if not isinstance(section, dict):
                # tolerate bad/old formats by falling back to defaults
                section = {}

            allowed = {fld.name for fld in fields(cls)}
            kwargs = {k: v for k, v in section.items() if k in allowed}
            return cls(**kwargs)

        return Config(
            experiment=build_section(ExperimentConfig, "experiment"),
            data=build_section(DataConfig, "data"),
            features=build_section(FeatureConfig, "features"),
            env=build_section(EnvironmentConfig, "env"),
            network=build_section(NetworkConfig, "network"),
            sac=build_section(SACConfig, "sac"),
            training=build_section(TrainingConfig, "training"),
            evaluation=build_section(EvaluationConfig, "evaluation"),
        )


    def set_global_seeds(self) -> None:
        """Seed Python, NumPy, and Torch."""
        seed = int(self.experiment.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def ensure_dirs(self) -> None:
        Path(self.training.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.experiment.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.evaluation.output_dir).mkdir(parents=True, exist_ok=True)

    def print_summary(self) -> None:
        if not self.experiment.verbose:
            return

        print("=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)

        print("\nEXPERIMENT:")
        print(f"  run_name: {self.experiment.run_name}")
        print(f"  seed: {self.experiment.seed}")

        print("\nDATA:")
        print(f"  tickers: {self.data.tickers}")
        print(f"  date range: {self.data.start_date} → {self.data.end_date}")
        print(f"  train split: {self.data.train_split_ratio:.1%}")
        print(f"  vix_path: {self.data.vix_path}")
        print(f"  vix3m_path: {self.data.vix3m_path}")
        print(f"  credit_spread_path: {self.data.credit_spread_path}")
        print(f"  yield_curve_path: {self.data.yield_curve_path}")

        print("\nFEATURES:")
        print(f"  rsi_period: {self.features.rsi_period}")
        print(f"  volatility_window: {self.features.volatility_window}")
        print(f"  vix_baseline / regimes: {self.features.vix_baseline} / "
              f"{self.features.vix_regime_low}, {self.features.vix_regime_high}")
        print(f"  credit baseline / regimes: {self.features.credit_baseline} / "
              f"{self.features.credit_regime_low}, {self.features.credit_regime_high}")
        # Regime HMM
        print(f"  use_regime_hmm: {self.features.use_regime_hmm}")
        if self.features.use_regime_hmm:
            print(f"    hmm_obs_ticker: {self.features.hmm_obs_ticker}")
            print(f"    hmm_n_states: {self.features.hmm_n_states}")
            print(f"    hmm_n_iter: {self.features.hmm_n_iter}")
            print(f"    hmm_include_rvol: {self.features.hmm_include_rvol}")
            if self.features.hmm_include_rvol:
                print(f"      hmm_rvol_window: {self.features.hmm_rvol_window}")
            print(f"    hmm_include_vix: {self.features.hmm_include_vix}")
            print(f"    hmm_include_credit_spread: {self.features.hmm_include_credit_spread}")
            print(f"    hmm_include_vix_term: {self.features.hmm_include_vix_term}")
            print(f"    hmm_include_yc_change: {self.features.hmm_include_yc_change}")
            if self.features.hmm_include_yc_change:
                print(f"      hmm_yc_change_lag: {self.features.hmm_yc_change_lag}")
            print(f"    hmm_include_dxy: {self.features.hmm_include_dxy}")
        # Yield Curve
        print(f"  yield_curve_slope_scale: {self.features.yield_curve_slope_scale}")
        print(f"  yield_curve_change_lag: {self.features.yield_curve_change_lag}")

        print("\nENVIRONMENT:")
        print(f"  lag: {self.env.lag}")
        print(f"  include_position_in_state: {self.env.include_position_in_state}")
        print(f"  tc_rate: {self.env.tc_rate:.6f} ({self.env.tc_rate * 10000:.1f} bps)")
        print(f"  reward_type: {self.env.reward_type}")
        if self.env.reward_type == "exp":
            print(f"    exp_risk_factor: {self.env.exp_risk_factor}")
        if self.env.reward_type == "sharpe":
            print(f"    sharpe_window: {self.env.sharpe_window}")
        print(f"  reward_scale: {self.env.reward_scale:.1f}")
        print(f"  reward clip: [{self.env.reward_clip_min}, {self.env.reward_clip_max}]")

        print("\nNETWORK:")
        print(f"  hidden_size: {self.network.hidden_size}")
        print(f"  num_layers: {self.network.num_layers}")
        print(f"  activation: {self.network.activation}")
        print(f"  layer_norm: {self.network.layer_norm}")
        print(f"  dropout: {self.network.dropout}")
        print(f"  alpha min/max: {self.network.alpha_min} / {self.network.alpha_max}")

        print("\nSAC:")
        print(f"  gamma: {self.sac.gamma}")
        print(f"  tau: {self.sac.tau}")
        print(f"  lr (actor/critic/alpha): {self.sac.actor_lr} / {self.sac.critic_lr} / {self.sac.alpha_lr}")
        print(f"  batch_size: {self.sac.batch_size}")
        print(f"  buffer_size: {self.sac.buffer_size:,}")
        print(f"  learning_starts: {self.sac.learning_starts}")
        print(f"  update_frequency: {self.sac.update_frequency}")
        print(f"  updates_per_step: {self.sac.updates_per_step}")
        print(f"  gradient_clip_norm: {self.sac.gradient_clip_norm}")
        print(f"  auto_entropy_tuning: {self.sac.auto_entropy_tuning}")
        if self.sac.target_entropy is not None:
            print(f"  target_entropy: {self.sac.target_entropy}")
        else:
            c = float(getattr(self.sac, "dirichlet_entropy_concentration", 1.0))
            m = float(getattr(self.sac, "target_entropy_margin", 0.0))
            print(f"  target_entropy: (auto Dirichlet; concentration={c}, margin={m})")

        print("\nTRAINING:")
        print(f"  total_timesteps: {self.training.total_timesteps:,}")
        print(f"  log_interval_episodes: {self.training.log_interval_episodes}")
        print(f"  save_interval_episodes: {self.training.save_interval_episodes}")
        print(f"  model_path_final: {self.training.model_path_final}")
        print(f"  model_path_best: {self.training.model_path_best}")
        if self.training.resume_from:
            print(f"  resume_from: {self.training.resume_from}")
        if self.training.device:
            print(f"  device: {self.training.device}")

        print("\nEVALUATION:")
        print(f"  model_path: {self.evaluation.model_path}")
        print(f"  deterministic: {self.evaluation.deterministic}")
        print(f"  render_plots: {self.evaluation.render_plots}")
        print(f"  save_plots: {self.evaluation.save_plots}")
        print(f"  output_dir: {self.evaluation.output_dir}")

        print("=" * 80)


def get_default_config() -> Config:
    return Config()
