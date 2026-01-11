import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import random
import os
from datetime import datetime

import torch

from environment import Env
from agent import Agent

print("="*60)
print("SAC PORTFOLIO EVALUATION SCRIPT")
print("="*60)

####################
### CONFIGURATION ###
####################
start = '2010-01-01'
end = '2024-12-31'
tickers = ['VNQ', 'SPY', 'TLT', 'GLD', 'BTC-USD']

# Hyperparameters (must match training)
learning_rate = 0.001
gamma = 0.99
tau = 0.005
alpha = 0.2
auto_entropy_tuning = True
buffer_size = 1000000
batch_size = 256
learning_starts = 10000
update_frequency = 1
n_hidden = 256

# Paths
model_path_best = "models/sac_portfolio_best.pth"
model_path_final = "models/sac_portfolio_final.pth"

# Device selection (skip MPS due to Dirichlet incompatibility)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using device: CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    print("⚠ WARNING: MPS (Apple Silicon) detected but not supported for SAC evaluation")
    print("  Reason: Dirichlet distribution requires CPU or CUDA")
    print("  Falling back to CPU (this is expected)")
    device = torch.device("cpu")
    print("✓ Using device: CPU")
else:
    device = torch.device("cpu")
    print("✓ Using device: CPU")

##########################
### Check Model Exists ###
##########################
print("\n" + "="*60)
print("CHECKING FOR TRAINED MODELS")
print("="*60)

if not os.path.exists(model_path_best) and not os.path.exists(model_path_final):
    print("\n✗ No trained model found!")
    print(f"  Expected: {model_path_best}")
    print(f"       or: {model_path_final}")
    print("\n  Please train a model first using:")
    print("    python train.py")
    print("="*60)
    exit(1)

if os.path.exists(model_path_best):
    print(f"✓ Best model found:  {model_path_best}")
if os.path.exists(model_path_final):
    print(f"✓ Final model found: {model_path_final}")

####################
### LOAD DATA ###
####################
print("\n" + "="*60)
print("LOADING MARKET DATA (one-time setup)")
print("="*60)

try:
    print(f"Downloading data for {tickers}...")
    df = yf.download(tickers, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("Downloaded data is empty.")
    print(f"✓ Downloaded {len(df)} days of data")
except Exception as e:
    print(f"✗ Error downloading data: {e}")
    exit(1)

df = df['Close']
df = df.dropna().copy()

####################
### VIX Features ###
####################
print("Loading VIX data...")

try:
    vix_df = pd.read_csv('../../data/VIX_CLS_2010_2024.csv')
    vix_df['observation_date'] = pd.to_datetime(vix_df['observation_date'])
    vix_df = vix_df.set_index('observation_date')
    vix_df = vix_df.rename(columns={'VIXCLS': 'VIX'})
    
    vix3m_df = pd.read_csv('../../data/VIX3M_CLS_2010_2024.csv')
    vix3m_df['observation_date'] = pd.to_datetime(vix3m_df['observation_date'])
    vix3m_df = vix3m_df.set_index('observation_date')
    vix3m_df = vix3m_df.rename(columns={'VXVCLS': 'VIX3M'})
    
    print("✓ VIX data loaded")
except Exception as e:
    print(f"✗ Error loading VIX data: {e}")
    exit(1)

vix_combined = vix_df.join(vix3m_df, how='outer')
df = df.join(vix_combined, how='left')
df['VIX'] = df['VIX'].ffill()
df['VIX3M'] = df['VIX3M'].ffill()

df['VIX_normalized'] = (df['VIX'] - 20) / 20

def get_vix_regime(vix):
    if vix < 15:
        return -1.0
    elif vix < 25:
        return 0.0
    else:
        return 1.0
df['VIX_regime'] = df['VIX'].apply(get_vix_regime)

df['VIX_term_structure'] = (df['VIX3M'] - df['VIX']) / df['VIX']
df['VIX_term_structure'] = np.clip(df['VIX_term_structure'], -1, 1)

##############################
### Credit Spread Features ###
##############################
print("Loading credit spread data...")

try:
    credit_df = pd.read_csv('../../data/CREDIT_SPREAD_2010_2024.csv')
    credit_df['observation_date'] = pd.to_datetime(credit_df['observation_date'])
    credit_df = credit_df.set_index('observation_date')
    print("✓ Credit spread data loaded")
except Exception as e:
    print(f"✗ Error loading credit spread data: {e}")
    exit(1)

df = df.join(credit_df, how='left')
df['BAMLC0A4CBBB'] = df['BAMLC0A4CBBB'].ffill()
df['Credit_Spread'] = df['BAMLC0A4CBBB'] / 100

df['Credit_Spread_normalized'] = (df['Credit_Spread'] - 0.02) / 0.02

def get_credit_regime(spread):
    if spread < 0.02:
        return -1.0
    elif spread < 0.04:
        return 0.0
    else:
        return 1.0
df['Credit_Spread_regime'] = df['Credit_Spread'].apply(get_credit_regime)

df['Credit_Spread_momentum'] = df['Credit_Spread'].pct_change(30)
df['Credit_Spread_momentum'] = np.clip(df['Credit_Spread_momentum'], -1, 1)

df['Credit_Spread_zscore'] = (
    (df['Credit_Spread'] - df['Credit_Spread'].rolling(252).mean()) / 
    (df['Credit_Spread'].rolling(252).std() + 1e-8)
)
df['Credit_Spread_zscore'] = np.clip(df['Credit_Spread_zscore'], -3, 3)

df['Credit_Spread_velocity'] = df['Credit_Spread_momentum'].diff(5)
df['Credit_Spread_velocity'] = np.clip(df['Credit_Spread_velocity'], -1, 1)

vix_normalized = (df['VIX'] - df['VIX'].rolling(60).mean()) / (df['VIX'].rolling(60).std() + 1e-8)
credit_normalized = (df['Credit_Spread'] - df['Credit_Spread'].rolling(60).mean()) / (df['Credit_Spread'].rolling(60).std() + 1e-8)
df['Credit_VIX_divergence'] = vix_normalized - credit_normalized
df['Credit_VIX_divergence'] = np.clip(df['Credit_VIX_divergence'], -3, 3)

####################
### RSI Features ###
####################
print("Creating RSI features...")
for ticker in tickers:
    df[ticker + '_RSI'] = talib.RSI(df[ticker], timeperiod=14) / 50 - 1

###########################
### Realized Volatility ###
###########################
print("Creating volatility features...")
for ticker in tickers:
    returns = df[ticker].pct_change()
    df[ticker + '_volatility'] = returns.rolling(20).std() * np.sqrt(252)
    df[ticker + '_volatility'] = (df[ticker + '_volatility'] - 0.25) / 0.25

df = df.dropna()
print(f"✓ All features created")

##########################
### Get Test Set Only ###
##########################
n_train = int(0.8 * len(df))
df_test = df.iloc[n_train:]

print(f"\n✓ Test data ready: {len(df_test)} days")
print(f"  Date range: {df_test.index[0]} to {df_test.index[-1]}")

###################
### ENVIRONMENT ###
###################
print("\n" + "="*60)
print("CREATING TEST ENVIRONMENT")
print("="*60)

# Environment configuration - MUST MATCH TRAINING!
include_position_in_state = True  # Same as training
tc_rate = 0.0005                 # Same as training: 5 bps per unit turnover
tc_fixed = 0.0                   # Same as training
turnover_threshold = 0.0         # Same as training
turnover_include_cash = False    # Same as training
turnover_use_half_factor = True  # Same as training

env_test = Env(
    df_test,
    tickers,
    lag=5,
    tc_rate=tc_rate,
    tc_fixed=tc_fixed,
    turnover_threshold=turnover_threshold,
    include_position_in_state=include_position_in_state,
    turnover_include_cash=turnover_include_cash,
    turnover_use_half_factor=turnover_use_half_factor,
)

print(f"✓ Test environment created")
print(f"  State dimension: {env_test.get_state_dim()}")
print(f"  Action dimension: {env_test.get_action_dim()}")
print(f"  Transaction costs: tc_rate={tc_rate} | tc_fixed={tc_fixed}")

#############
### AGENT ###
#############
print("\n" + "="*60)
print("INITIALIZING AGENT (one-time setup)")
print("="*60)

agent = Agent(
    n_input=env_test.get_state_dim(),
    n_action=env_test.get_action_dim(),
    learning_rate=learning_rate,
    gamma=gamma,
    tau=tau,
    alpha=alpha,
    auto_entropy_tuning=auto_entropy_tuning,
    buffer_size=buffer_size,
    batch_size=batch_size,
    learning_starts=learning_starts,
    update_frequency=update_frequency,
    n_hidden=n_hidden,
    device=device
)
print(f"✓ Agent initialized on device: {device}")

####################
### LOAD MODEL ###
####################
print("\n" + "="*60)
print("LOADING TRAINED MODEL (one-time setup)")
print("="*60)

print("\nWhich model do you want to evaluate?")
print("  1. Best model (recommended)")
print("  2. Final model")

model_choice = input("Enter choice (1 or 2): ").strip()

if model_choice == '1':
    if not os.path.exists(model_path_best):
        print(f"\n⚠ Best model not found: {model_path_best}")
        print(f"  Loading final model instead...")
        agent.load_model(model_path_final)
        best_model_state = None
        print(f"✓ Loaded final model from: {model_path_final}")
    else:
        best_model_state = torch.load(model_path_best, map_location=device, weights_only=False)
        agent.load_best_model(best_model_state)
        print(f"\n✓ Loaded best model from: {model_path_best}")
        print(f"  - Episode: {best_model_state.get('episode', 'N/A')}")
        print(f"  - Global step: {best_model_state.get('global_step', 'N/A')}")
        print(f"  - Avg return: {best_model_state.get('avg_return', 'N/A'):.4f}")
elif model_choice == '2':
    agent.load_model(model_path_final)
    print(f"\n✓ Loaded final model from: {model_path_final}")
    best_model_state = None
else:
    print("\n✗ Invalid choice. Exiting.")
    exit(1)

# Set to evaluation mode
agent.policy.eval()
agent.q1.eval()
agent.q2.eval()
agent.value.eval()

print("\n" + "="*60)
print("SETUP COMPLETE - MODEL LOADED IN MEMORY")
print("="*60)
print("You can now run multiple evaluations without reloading!")

###################################
### MAIN EVALUATION LOOP ###
###################################
while True:
    print("\n" + "="*60)
    print("EVALUATION MENU")
    print("="*60)
    print("\nWhat would you like to do?")
    print("  1. Run deterministic evaluation")
    print("  2. Run stochastic evaluation (with seed)")
    print("  3. Run multiple stochastic evaluations (statistical analysis)")
    print("  4. Switch to different model")
    print("  5. Exit")
    
    menu_choice = input("\nEnter choice (1-5): ").strip()
    
    if menu_choice == '5':
        print("\n" + "="*60)
        print("Exiting evaluation script. Goodbye!")
        print("="*60)
        break
    
    elif menu_choice == '4':
        print("\n" + "="*60)
        print("SWITCH MODEL")
        print("="*60)
        print("\nWhich model do you want to load?")
        print("  1. Best model")
        print("  2. Final model")
        
        new_model_choice = input("Enter choice (1 or 2): ").strip()
        
        if new_model_choice == '1':
            if not os.path.exists(model_path_best):
                print(f"\n⚠ Best model not found: {model_path_best}")
            else:
                best_model_state = torch.load(model_path_best, map_location=device, weights_only=False)
                agent.load_best_model(best_model_state)
                print(f"\n✓ Switched to best model")
        elif new_model_choice == '2':
            agent.load_model(model_path_final)
            print(f"\n✓ Switched to final model")
        else:
            print("\n✗ Invalid choice, keeping current model")
        
        agent.policy.eval()
        continue
    
    elif menu_choice not in ['1', '2', '3']:
        print("\n✗ Invalid choice. Please enter 1-5.")
        continue
    
    # Map menu choice to eval mode
    eval_mode = menu_choice
    
    # Initialize variables
    n_runs = 1
    show_statistics = False
    
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    
    ##############################
    ### MODE 1: DETERMINISTIC ###
    ##############################
    if eval_mode == '1':
        print("\n✓ Using deterministic evaluation (mean of Dirichlet)")
        print("  This gives reproducible results by using expected allocation")
        
        obs = env_test.reset()
        sac_rewards = []
        sac_actions = []
        done = False
        
        while not done:
            actions_np = agent.choose_action_deterministic(obs)
            sac_actions.append(actions_np.copy())
            obs, reward, done = env_test.step(actions_np)
            sac_rewards.append(reward)
        
        sac_rewards = np.array(sac_rewards)
        sac_actions = np.array(sac_actions)
        
        # Calculate metrics
        sac_cumulative = np.cumprod(np.exp(sac_rewards))
        sac_total_return = sac_cumulative[-1] - 1
        sac_sharpe = np.mean(sac_rewards) / (np.std(sac_rewards) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        sac_running_max = np.maximum.accumulate(sac_cumulative)
        sac_drawdown = (sac_cumulative - sac_running_max) / sac_running_max
        sac_max_drawdown = np.min(sac_drawdown)
        
        # For display
        display_mode = "Deterministic"
    
    ##################################
    ### MODE 2: STOCHASTIC + SEED ###
    ##################################
    elif eval_mode == '2':
        print("\n✓ Using stochastic evaluation with fixed seed")
        print("  This gives reproducible results with sampling randomness")
        
        # Set seed for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        
        obs = env_test.reset()
        sac_rewards = []
        sac_actions = []
        done = False
        
        while not done:
            actions_np = agent.choose_action_stochastic(obs)
            sac_actions.append(actions_np.copy())
            obs, reward, done = env_test.step(actions_np)
            sac_rewards.append(reward)
        
        sac_rewards = np.array(sac_rewards)
        sac_actions = np.array(sac_actions)
        
        # Calculate metrics
        sac_cumulative = np.cumprod(np.exp(sac_rewards))
        sac_total_return = sac_cumulative[-1] - 1
        sac_sharpe = np.mean(sac_rewards) / (np.std(sac_rewards) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        sac_running_max = np.maximum.accumulate(sac_cumulative)
        sac_drawdown = (sac_cumulative - sac_running_max) / sac_running_max
        sac_max_drawdown = np.min(sac_drawdown)
        
        # For display
        display_mode = "Stochastic (seed=42)"
    
    #################################
    ### MODE 3: MULTIPLE RUNS ###
    #################################
    elif eval_mode == '3':
        # Get number of runs with validation
        n_runs_input = input("\nHow many evaluation runs? (recommended: 10-20, default: 10): ").strip()
        try:
            n_runs = int(n_runs_input) if n_runs_input else 10
            n_runs = max(1, min(n_runs, 100))
            if n_runs != int(n_runs_input or 10):
                print(f"  Note: Adjusted to {n_runs} runs (valid range: 1-100)")
        except ValueError:
            print("  Invalid input. Using default of 10 runs.")
            n_runs = 10
        
        print(f"\n✓ Running {n_runs} stochastic evaluations...")
        print("  This may take a few minutes...")
        
        all_returns = []
        all_sharpes = []
        all_drawdowns = []
        all_volatilities = []
        all_actions_list = []
        all_rewards_list = []
        
        # Progress indicator
        print("  Progress: ", end="", flush=True)
        progress_interval = max(1, n_runs // 10)
        
        for run in range(n_runs):
            obs = env_test.reset()
            run_rewards = []
            run_actions = []
            done = False
            
            while not done:
                actions_np = agent.choose_action_stochastic(obs)
                run_actions.append(actions_np.copy())
                obs, reward, done = env_test.step(actions_np)
                run_rewards.append(reward)
            
            run_rewards = np.array(run_rewards)
            run_actions = np.array(run_actions)
            
            # Calculate metrics for this run
            run_cumulative = np.cumprod(np.exp(run_rewards))
            run_return = run_cumulative[-1] - 1
            run_sharpe = np.mean(run_rewards) / (np.std(run_rewards) + 1e-8) * np.sqrt(252)
            run_volatility = np.std(run_rewards) * np.sqrt(252)
            
            run_running_max = np.maximum.accumulate(run_cumulative)
            run_drawdown = (run_cumulative - run_running_max) / run_running_max
            run_max_drawdown = np.min(run_drawdown)
            
            all_returns.append(run_return)
            all_sharpes.append(run_sharpe)
            all_drawdowns.append(run_max_drawdown)
            all_volatilities.append(run_volatility)
            all_actions_list.append(run_actions)
            all_rewards_list.append(run_rewards)
            
            # Progress indicator
            if (run + 1) % progress_interval == 0 or run == n_runs - 1:
                progress_pct = int((run + 1) * 100 / n_runs)
                print(f"{progress_pct}%", end=" ", flush=True)
        
        print("\n  Complete!\n")
        
        # Print individual run results
        print("  Individual Run Results:")
        for run in range(n_runs):
            print(f"    Run {run+1:2d}/{n_runs}: Return = {all_returns[run]*100:7.2f}%, "
                  f"Sharpe = {all_sharpes[run]:.4f}, Max DD = {all_drawdowns[run]*100:.2f}%")
        
        # Convert to arrays
        all_returns = np.array(all_returns)
        all_sharpes = np.array(all_sharpes)
        all_drawdowns = np.array(all_drawdowns)
        all_volatilities = np.array(all_volatilities)
        
        # Use median run for visualization
        median_idx = np.argsort(all_returns)[len(all_returns)//2]
        sac_rewards = all_rewards_list[median_idx]
        sac_actions = all_actions_list[median_idx]
        
        # Recalculate for median run (for visualization)
        sac_cumulative = np.cumprod(np.exp(sac_rewards))
        sac_total_return = sac_cumulative[-1] - 1
        sac_sharpe = np.mean(sac_rewards) / (np.std(sac_rewards) + 1e-8) * np.sqrt(252)
        sac_running_max = np.maximum.accumulate(sac_cumulative)
        sac_drawdown = (sac_cumulative - sac_running_max) / sac_running_max
        sac_max_drawdown = np.min(sac_drawdown)
        
        # For display
        display_mode = f"Multiple Runs (n={n_runs})"
        show_statistics = True
    
    ###################################
    ### EVALUATE BENCHMARK PORTFOLIO ###
    ###################################
    print("\nEvaluating Benchmark...")
    benchmark_weights = np.array([0.1, 0.5, 0.3, 0.05, 0.05, 0])
    obs = env_test.reset()
    benchmark_rewards = []
    done = False
    
    while not done:
        obs, reward, done = env_test.step(benchmark_weights)
        benchmark_rewards.append(reward)
    
    benchmark_rewards = np.array(benchmark_rewards)
    benchmark_cumulative = np.cumprod(np.exp(benchmark_rewards))
    benchmark_total_return = benchmark_cumulative[-1] - 1
    benchmark_sharpe = np.mean(benchmark_rewards) / (np.std(benchmark_rewards) + 1e-8) * np.sqrt(252)
    
    # Maximum drawdown
    benchmark_running_max = np.maximum.accumulate(benchmark_cumulative)
    benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
    benchmark_max_drawdown = np.min(benchmark_drawdown)
    
    print("✓ Benchmark evaluation complete")
    
    ########################
    ### PRINT RESULTS ###
    ########################
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE")
    print("="*60)
    print(f"Evaluation Mode: {display_mode}")
    
    if show_statistics:
        # Show aggregated statistics
        print("\n" + "-"*60)
        print("SAC AGENT - STATISTICAL SUMMARY")
        print("-"*60)
        print(f"  Total Return:")
        print(f"    Mean:        {np.mean(all_returns)*100:>8.2f}%")
        print(f"    Std Dev:     {np.std(all_returns)*100:>8.2f}%")
        print(f"    Min:         {np.min(all_returns)*100:>8.2f}%")
        print(f"    Max:         {np.max(all_returns)*100:>8.2f}%")
        print(f"    Median:      {np.median(all_returns)*100:>8.2f}%")
        print(f"    95% CI:      [{np.percentile(all_returns, 2.5)*100:.2f}%, {np.percentile(all_returns, 97.5)*100:.2f}%]")
        
        print(f"\n  Sharpe Ratio:")
        print(f"    Mean:        {np.mean(all_sharpes):>8.4f}")
        print(f"    Std Dev:     {np.std(all_sharpes):>8.4f}")
        print(f"    Min:         {np.min(all_sharpes):>8.4f}")
        print(f"    Max:         {np.max(all_sharpes):>8.4f}")
        
        print(f"\n  Max Drawdown:")
        print(f"    Mean:        {np.mean(all_drawdowns)*100:>8.2f}%")
        print(f"    Std Dev:     {np.std(all_drawdowns)*100:>8.2f}%")
        print(f"    Best (min):  {np.max(all_drawdowns)*100:>8.2f}%")
        print(f"    Worst (max): {np.min(all_drawdowns)*100:>8.2f}%")
        
        print(f"\n  Ann. Volatility:")
        print(f"    Mean:        {np.mean(all_volatilities)*100:>8.2f}%")
        print(f"    Std Dev:     {np.std(all_volatilities)*100:>8.2f}%")
        
        # Probability metrics
        prob_beat_benchmark = np.mean(all_returns > benchmark_total_return) * 100
        prob_positive = np.mean(all_returns > 0) * 100
        prob_2x_benchmark = np.mean(all_returns > 2 * benchmark_total_return) * 100
        
        print(f"\n  Probability Metrics:")
        print(f"    P(beat benchmark):   {prob_beat_benchmark:>6.1f}%")
        print(f"    P(positive return):  {prob_positive:>6.1f}%")
        print(f"    P(>2x benchmark):    {prob_2x_benchmark:>6.1f}%")
        
        # Coefficient of variation
        cv_return = np.std(all_returns) / np.mean(all_returns) if np.mean(all_returns) != 0 else 0
        print(f"\n  Stability Metric:")
        print(f"    Coefficient of Variation: {cv_return:.4f}")
        if cv_return < 0.05:
            stability = "VERY STABLE ✓✓✓"
        elif cv_return < 0.10:
            stability = "STABLE ✓✓"
        elif cv_return < 0.15:
            stability = "MODERATE ✓"
        else:
            stability = "UNSTABLE ⚠"
        print(f"    Assessment: {stability}")
        
        # Option to save results to CSV
        save_csv = input("\n  Would you like to save detailed results to CSV? (y/n): ").strip().lower()
        if save_csv == 'y':
            os.makedirs("results", exist_ok=True)
            results_df = pd.DataFrame({
                'run': range(1, n_runs + 1),
                'total_return_pct': all_returns * 100,
                'sharpe_ratio': all_sharpes,
                'max_drawdown_pct': all_drawdowns * 100,
                'ann_volatility_pct': all_volatilities * 100
            })
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/evaluation_results_{timestamp}.csv"
            try:
                results_df.to_csv(filename, index=False)
                print(f"  ✓ Results saved to: {filename}")
            except Exception as e:
                print(f"  ✗ Failed to save results: {e}")
    
    else:
        # Show single run results
        print("\n" + "-"*60)
        print("SAC AGENT")
        print("-"*60)
        print(f"  Total Return:      {sac_total_return*100:>8.2f}%")
        print(f"  Sharpe Ratio:      {sac_sharpe:>8.4f}")
        print(f"  Max Drawdown:      {sac_max_drawdown*100:>8.2f}%")
        print(f"  Ann. Volatility:   {np.std(sac_rewards)*np.sqrt(252)*100:>8.2f}%")
    
    print("\n" + "-"*60)
    print("BENCHMARK (VNQ:10% SPY:50% TLT:30% GLD:5% BTC:5%)")
    print("-"*60)
    print(f"  Total Return:      {benchmark_total_return*100:>8.2f}%")
    print(f"  Sharpe Ratio:      {benchmark_sharpe:>8.4f}")
    print(f"  Max Drawdown:      {benchmark_max_drawdown*100:>8.2f}%")
    print(f"  Ann. Volatility:   {np.std(benchmark_rewards)*np.sqrt(252)*100:>8.2f}%")
    
    print("\n" + "-"*60)
    print("COMPARISON (SAC vs Benchmark)")
    print("-"*60)
    
    if show_statistics:
        mean_return_diff = (np.mean(all_returns) - benchmark_total_return) * 100
        mean_sharpe_diff = np.mean(all_sharpes) - benchmark_sharpe
        print(f"  Return Difference:   {mean_return_diff:>+8.2f}% (mean)")
        print(f"  Sharpe Difference:   {mean_sharpe_diff:>+8.4f} (mean)")
        
        if np.mean(all_returns) > benchmark_total_return:
            print(f"\n  ✓ SAC outperformed benchmark in {prob_beat_benchmark:.1f}% of runs")
            print(f"  ✓ Average outperformance: {mean_return_diff:.2f}%")
        else:
            print(f"\n  ⚠ SAC underperformed benchmark on average")
    else:
        return_diff = (sac_total_return - benchmark_total_return) * 100
        sharpe_diff = sac_sharpe - benchmark_sharpe
        print(f"  Return Difference:   {return_diff:>+8.2f}%")
        print(f"  Sharpe Difference:   {sharpe_diff:>+8.4f}")
        
        if sac_total_return > benchmark_total_return:
            print(f"\n  ✓ SAC outperformed benchmark by {return_diff:.2f}%")
        else:
            print(f"\n  ⚠ SAC underperformed benchmark by {abs(return_diff):.2f}%")
    
    #####################
    ### VISUALIZATION ###
    #####################
    visualize = input("\nWould you like to visualize the results? (y/n): ").strip().lower()
    
    if visualize == 'y':
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        ax1.plot(sac_cumulative, label='SAC Agent', linewidth=2, color='blue')
        ax1.plot(benchmark_cumulative, label='Benchmark', linewidth=2, linestyle='--', color='orange')
        if show_statistics:
            ax1.set_title(f'Cumulative Returns (Median of {n_runs} runs)', fontsize=12, fontweight='bold')
        else:
            ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        ax2.fill_between(range(len(sac_drawdown)), sac_drawdown*100, 0, 
                          alpha=0.4, label='SAC Drawdown', color='blue')
        ax2.fill_between(range(len(benchmark_drawdown)), benchmark_drawdown*100, 0, 
                          alpha=0.4, label='Benchmark Drawdown', color='orange')
        ax2.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Daily Returns Distribution (or Return Distribution if multiple runs)
        ax3 = axes[1, 0]
        if show_statistics:
            ax3.hist(all_returns*100, bins=min(30, n_runs), alpha=0.7, label=f'SAC (n={n_runs})', 
                    density=True, color='blue', edgecolor='black')
            ax3.axvline(np.mean(all_returns)*100, color='blue', linestyle='--', 
                       linewidth=2, label='SAC Mean')
            ax3.axvline(benchmark_total_return*100, color='orange', linestyle='--', 
                       linewidth=2, label='Benchmark')
            ax3.set_title('Total Return Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Total Return (%)')
        else:
            ax3.hist(sac_rewards*100, bins=50, alpha=0.5, label='SAC', density=True, color='blue')
            ax3.hist(benchmark_rewards*100, bins=50, alpha=0.5, label='Benchmark', 
                    density=True, color='orange')
            ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Portfolio Weights Over Time
        ax4 = axes[1, 1]
        asset_names = ['VNQ', 'SPY', 'TLT', 'GLD', 'BTC', 'Cash']
        colors = ['brown', 'green', 'blue', 'gold', 'orange', 'gray']
        for i, (name, color) in enumerate(zip(asset_names, colors)):
            ax4.plot(sac_actions[:, i], label=name, alpha=0.7, linewidth=1.5, color=color)
        if show_statistics:
            ax4.set_title(f'Portfolio Weights Over Time (Median run)', fontsize=12, fontweight='bold')
        else:
            ax4.set_title('Portfolio Weights Over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Trading Days')
        ax4.set_ylabel('Weight')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        plt.suptitle(f'SAC Evaluation Results: {df_test.index[0].strftime("%Y-%m-%d")} to {df_test.index[-1].strftime("%Y-%m-%d")}', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save plot
        os.makedirs("plots", exist_ok=True)
        plot_filename = f"plots/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=150)
        print(f"\n✓ Plot saved to: {plot_filename}")
        
        plt.show()
        print("✓ Visualization displayed!")
    
    # End of this evaluation run - loop back to menu

# Exit message (only reached when user chooses option 5)
print("\nThank you for using the SAC Portfolio Evaluation System!")
