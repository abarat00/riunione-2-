"""
Benchmark Test Script for Portfolio Agent

Usage:
    python benchmark_test.py --model_path results/baseline_full/weights/portfolio_actor_50.pth
                           --output_dir results/baseline_full

Description:
    This script provides a standardized way to test a trained portfolio agent against
    a known test dataset and calculates consistent performance metrics. It's designed
    to be used after training to generate reliable metrics that can be compared across
    different experiments.

    It uses the same environment setup as the main training script but focuses specifically
    on testing and metrics calculation, avoiding the discrepancies between training and test
    performance reporting.
"""

import os
import argparse
import numpy as np
import pandas as pd
import json
import torch
from datetime import datetime
import matplotlib.pyplot as plt

# Import our modules (ensure they're in your PYTHONPATH)
from portfolio_agent import PortfolioAgent
from portfolio_env import PortfolioEnvironment
from financial_calendar import FinancialCalendar

# Base configuration 
BASE_PATH = 'C:\\Users\\Administrator\\Desktop\\DRL PORTFOLIO\\NAS Results\\Multi_Ticker\\Normalized_RL_INPUT\\'
NORM_PARAMS_PATH_BASE = f'{BASE_PATH}json\\'
CSV_PATH_BASE = f'{BASE_PATH}'

# Default list of tickers
DEFAULT_TICKERS = ["XLF", "XLE", "XLK", "IHI", "XLY"]

# Features to use
norm_columns = [
    "open", "volume", "change", "day", "week", "adjCloseGold", "adjCloseSpy",
    "Credit_Spread", "m_plus", "m_minus", "drawdown", "drawup",
    "s_plus", "s_minus", "upper_bound", "lower_bound", "avg_duration", "avg_depth",
    "cdar_95", "VIX_Close", "MACD", "MACD_Signal", "MACD_Histogram", "SMA5",
    "SMA10", "SMA15", "SMA20", "SMA25", "SMA30", "SMA36", "RSI5", "RSI14", "RSI20",
    "RSI25", "ADX5", "ADX10", "ADX15", "ADX20", "ADX25", "ADX30", "ADX35",
    "BollingerLower", "BollingerUpper", "WR5", "WR14", "WR20", "WR25",
    "SMA5_SMA20", "SMA5_SMA36", "SMA20_SMA36", "SMA5_Above_SMA20",
    "Golden_Cross", "Death_Cross", "BB_Position", "BB_Width",
    "BB_Upper_Distance", "BB_Lower_Distance", "Volume_SMA20", "Volume_Change_Pct",
    "Volume_1d_Change_Pct", "Volume_Spike", "Volume_Collapse", "GARCH_Vol",
    "pred_lstm", "pred_gru", "pred_blstm", "pred_lstm_direction",
    "pred_gru_direction", "pred_blstm_direction"
]

def check_file_exists(file_path):
    """Verify if a file exists and print an appropriate message."""
    if not os.path.exists(file_path):
        print(f"WARNING: File not found: {file_path}")
        return False
    return True

def load_data_for_tickers(tickers, train_fraction=0.8):
    """
    Load and prepare data for all tickers.
    
    Parameters:
    - tickers: list of tickers to load
    - train_fraction: fraction of data to use for training (0.8 = 80%)
    
    Returns:
    - dfs_train: dict of DataFrames for training
    - dfs_test: dict of DataFrames for testing
    - norm_params_paths: dict of paths to normalization parameters
    - valid_tickers: list of tickers that were successfully loaded
    """
    dfs_train = {}
    dfs_test = {}
    norm_params_paths = {}
    valid_tickers = []
    
    for ticker in tickers:
        norm_params_path = f'{NORM_PARAMS_PATH_BASE}{ticker}_norm_params.json'
        csv_path = f'{CSV_PATH_BASE}{ticker}\\{ticker}_normalized.csv'
        
        # Verify file existence
        if not (check_file_exists(norm_params_path) and check_file_exists(csv_path)):
            print(f"Skipping ticker {ticker} due to missing files")
            continue
        
        # Load dataset
        print(f"Loading data for {ticker}...")
        df = pd.read_csv(csv_path)
        
        # Check for all required columns
        missing_cols = [col for col in norm_columns if col not in df.columns]
        if missing_cols:
            print(f"Skipping ticker {ticker}. Missing columns: {missing_cols}")
            continue
        
        # Sort dataset by date (if present)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        # Split into training and test
        train_size = int(len(df) * train_fraction)
        dfs_train[ticker] = df.iloc[:train_size]
        dfs_test[ticker] = df.iloc[train_size:]
        norm_params_paths[ticker] = norm_params_path
        
        valid_tickers.append(ticker)
        print(f"Dataset for {ticker} loaded: {len(df)} rows")
    
    return dfs_train, dfs_test, norm_params_paths, valid_tickers

def align_dataframes(dfs):
    """
    Align DataFrames to have the same date range and number of rows.
    """
    aligned_dfs = {}
    
    # Find common date range
    if all('date' in df.columns for df in dfs.values()):
        # Find most recent start date
        start_date = max(df['date'].min() for df in dfs.values())
        # Find earliest end date
        end_date = min(df['date'].max() for df in dfs.values())
        
        print(f"Common date range: {start_date} - {end_date}")
        
        # Filter and align each DataFrame
        for ticker, df in dfs.items():
            aligned_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
            # Make sure dates are sorted
            aligned_df = aligned_df.sort_values('date')
            aligned_dfs[ticker] = aligned_df
        
        # Check that all aligned DataFrames have the same number of rows
        lengths = [len(df) for df in aligned_dfs.values()]
        if len(set(lengths)) > 1:
            print(f"WARNING: Aligned DataFrames have different lengths: {lengths}")
            # Find minimum length
            min_length = min(lengths)
            print(f"Truncating to {min_length} rows...")
            # Truncate all DataFrames to the same length
            for ticker in aligned_dfs:
                aligned_dfs[ticker] = aligned_dfs[ticker].iloc[:min_length].copy()
    else:
        # If no 'date' columns, use the minimum number of rows
        min_rows = min(len(df) for df in dfs.values())
        for ticker, df in dfs.items():
            aligned_dfs[ticker] = df.iloc[:min_rows].copy()
    
    # Final length check
    lengths = [len(df) for df in aligned_dfs.values()]
    print(f"Aligned DataFrame lengths: {lengths}")
    
    return aligned_dfs

def load_environment_config(config_path):
    """Load environment configuration from JSON file."""
    if not os.path.exists(config_path):
        print(f"WARNING: Configuration file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}

def create_test_environment(tickers, aligned_dfs_test, norm_params_paths, env_config=None):
    """Create a test environment based on the given configuration."""
    # Default configuration
    default_config = {
        'tickers': tickers,
        'sigma': 0.1,
        'theta': 0.1,
        'lambd': 0.05,
        'psi': 0.2,
        'cost': "trade_l1",
        'max_pos_per_asset': 2.0,
        'max_portfolio_pos': 6.0,
        'squared_risk': False,
        'penalty': "tanh",
        'alpha': 3,
        'beta': 3,
        'clip': True,
        'scale_reward': 5,
        'dfs': aligned_dfs_test,
        'max_step': min(len(df) for df in aligned_dfs_test.values()),
        'norm_params_paths': norm_params_paths,
        'norm_columns': norm_columns,
        'free_trades_per_month': 10,
        'commission_rate': 0.0025,
        'min_commission': 1.0,
        'trading_frequency_penalty_factor': 0.05,
        'position_stability_bonus_factor': 0.2,
        'correlation_penalty_factor': 0.15,
        'diversification_bonus_factor': 0.25,
        'initial_capital': 100000,
        'risk_free_rate': 0.02,
        'use_sortino': True,
        'target_return': 0.05
    }
    
    # Update with provided configuration if available
    if env_config:
        for key, value in env_config.items():
            if key != 'dfs' and key != 'norm_params_paths' and key != 'max_step':
                try:
                    # Convert string values to appropriate types
                    if isinstance(value, str):
                        if value.lower() == 'true':
                            default_config[key] = True
                        elif value.lower() == 'false':
                            default_config[key] = False
                        elif value.replace('.', '', 1).isdigit():
                            if '.' in value:
                                default_config[key] = float(value)
                            else:
                                default_config[key] = int(value)
                        else:
                            default_config[key] = value
                    else:
                        default_config[key] = value
                except Exception as e:
                    print(f"Error setting config value {key}={value}: {e}")
    
    # Always use the provided aligned data and max_step
    default_config['dfs'] = aligned_dfs_test
    default_config['max_step'] = min(len(df) for df in aligned_dfs_test.values())
    default_config['norm_params_paths'] = norm_params_paths
    
    # Create and return the environment
    return PortfolioEnvironment(**default_config)

def create_agent(tickers, model_path, features_per_asset):
    """Create and initialize an agent with the model."""
    num_assets = len(tickers)
    
    # Crea l'agente
    agent = PortfolioAgent(
        num_assets=num_assets,
        memory_type="prioritized",
        batch_size=256,
        max_step=1000,
        theta=0.1,
        sigma=0.2,
        use_enhanced_actor=True,
        use_batch_norm=True
    )
    
    # AGGIUNGI QUESTE RIGHE:
    # Inizializza esplicitamente l'actor prima di caricare il modello
    from portfolio_models import EnhancedPortfolioActor
    state_size = 5 * features_per_asset + 5 + 5 + 6  # Calcolo dello state_size come in PortfolioEnvironment
    agent.actor_local = EnhancedPortfolioActor(
        state_size=state_size, 
        action_size=num_assets, 
        features_per_asset=features_per_asset,
        encoding_size=32
    ).to(agent.device)
    agent.actor_target = EnhancedPortfolioActor(
        state_size=state_size, 
        action_size=num_assets, 
        features_per_asset=features_per_asset,
        encoding_size=32
    ).to(agent.device)
    
    # Carica il modello
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        agent.load_models(actor_path=model_path)
    else:
        print(f"ERROR: Model not found at {model_path}")
    
    return agent

def run_test(env, agent):
    """Run a test episode and collect detailed metrics."""
    env.reset()
    state = env.get_state()
    done = env.done
    
    # Create structures for logging
    action_log = []
    position_log = []
    portfolio_value_log = []
    rewards_log = []
    
    step_counter = 0
    
    while not done:
        with torch.no_grad():
            actions = agent.act(state, noise=False)
        
        reward = env.step(actions)
        state = env.get_state()
        done = env.done
        
        # Save logs
        action_log.append(actions)
        position_log.append(env.positions.copy())
        portfolio_value_log.append(env.get_portfolio_value())
        rewards_log.append(reward)
        
        step_counter += 1
    
    # Get final metrics
    metrics = env.get_real_portfolio_metrics()
    
    # Add additional metrics
    metrics['total_steps'] = step_counter
    metrics['cumulative_reward'] = sum(rewards_log)
    metrics['avg_reward'] = np.mean(rewards_log)
    metrics['initial_portfolio_value'] = portfolio_value_log[0]
    
    # Calculate some position statistics
    position_array = np.array(position_log)
    metrics['avg_positions'] = np.mean(position_array, axis=0).tolist()
    metrics['max_positions'] = np.max(position_array, axis=0).tolist()
    metrics['min_positions'] = np.min(position_array, axis=0).tolist()
    
    # Calculate average trade size
    action_array = np.array(action_log)
    metrics['avg_trade_size'] = np.mean(np.abs(action_array))
    metrics['max_trade_size'] = np.max(np.abs(action_array))
    
    # Return both metrics and detailed logs
    logs = {
        'action_log': action_log,
        'position_log': position_log,
        'portfolio_value_log': portfolio_value_log,
        'rewards_log': rewards_log
    }
    
    return metrics, logs

def save_test_results(metrics, logs, tickers, output_dir):
    """Save test results and logs to the specified directory."""
    test_dir = os.path.join(output_dir, 'benchmark_test')
    os.makedirs(test_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = os.path.join(test_dir, 'test_metrics_benchmark.json')
    with open(metrics_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, list)):
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                else:
                    serializable_metrics[key] = value
            elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        json.dump(serializable_metrics, f, indent=4)
    
    # Save logs as numpy arrays
    action_log_array = np.array([a for a in logs['action_log']])
    position_log_array = np.array([p for p in logs['position_log']])
    portfolio_value_array = np.array(logs['portfolio_value_log'])
    rewards_array = np.array(logs['rewards_log'])
    
    np.save(os.path.join(test_dir, 'action_log.npy'), action_log_array)
    np.save(os.path.join(test_dir, 'position_log.npy'), position_log_array)
    np.save(os.path.join(test_dir, 'portfolio_value_log.npy'), portfolio_value_array)
    np.save(os.path.join(test_dir, 'rewards_log.npy'), rewards_array)
    
    # Create a CSV with the main data
    df = pd.DataFrame({
        'step': range(len(portfolio_value_array)),
        'portfolio_value': portfolio_value_array,
        'reward': rewards_array
    })
    
    # Add positions for each asset
    for i, ticker in enumerate(tickers):
        if i < position_log_array.shape[1]:
            df[f'position_{ticker}'] = position_log_array[:, i]
    
    df.to_csv(os.path.join(test_dir, 'test_details.csv'), index=False)
    
    # Also copy the metrics to the main test directory for compatibility
    standard_test_dir = os.path.join(output_dir, 'test')
    os.makedirs(standard_test_dir, exist_ok=True)
    
    with open(os.path.join(standard_test_dir, 'test_metrics_benchmark.json'), 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    df.to_csv(os.path.join(standard_test_dir, 'test_details.csv'), index=False)
    
    print(f"Test results saved in: {test_dir}")

def plot_benchmark_results(metrics, logs, tickers, output_dir):
    """Create a simple visualization of key benchmark results."""
    test_dir = os.path.join(output_dir, 'benchmark_test')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # 1. Portfolio Value
    plt.subplot(3, 1, 1)
    plt.plot(logs['portfolio_value_log'], linewidth=2)
    plt.axhline(y=logs['portfolio_value_log'][0], color='r', linestyle='--', alpha=0.5)
    plt.title('Portfolio Value During Test')
    plt.xlabel('Step')
    plt.ylabel('Value ($)')
    plt.grid(True, alpha=0.3)
    
    # Annotate with key metrics
    plt.text(0.02, 0.90, f"Initial Value: ${metrics['initial_portfolio_value']:,.2f}", 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.02, 0.85, f"Final Value: ${metrics['final_portfolio_value']:,.2f}", 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.02, 0.80, f"Total Return: {metrics['total_return']:.2f}%", 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.02, 0.75, f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}", 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 2. Asset Positions
    plt.subplot(3, 1, 2)
    position_log_array = np.array([p for p in logs['position_log']])
    for i in range(position_log_array.shape[1]):
        if i < len(tickers):
            plt.plot(position_log_array[:, i], label=tickers[i])
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Asset Positions Over Time')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 3. Trading Activity (Cumulative Actions)
    plt.subplot(3, 1, 3)
    action_log_array = np.array([a for a in logs['action_log']])
    cumulative_actions = np.cumsum(np.abs(action_log_array), axis=0)
    
    for i in range(cumulative_actions.shape[1]):
        if i < len(tickers):
            plt.plot(cumulative_actions[:, i], label=f"{tickers[i]} Cumulative")
    
    plt.title('Cumulative Trading Activity')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Action Magnitude')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add overall title and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(f'Benchmark Test Results - {datetime.now().strftime("%Y-%m-%d")}', fontsize=16)
    
    plt.savefig(os.path.join(test_dir, 'benchmark_results.png'), dpi=150)
    plt.close()
    
    print(f"Benchmark results plot saved to: {os.path.join(test_dir, 'benchmark_results.png')}")

def main(args):
    """Main function for benchmark testing."""
    model_path = args.model_path
    output_dir = args.output_dir
    tickers = args.tickers.split(',') if args.tickers else DEFAULT_TICKERS
    
    print(f"Running benchmark test with model: {model_path}")
    print(f"Tickers: {tickers}")
    
    # 1. Load and prepare test data
    print("Loading test data...")
    _, dfs_test, norm_params_paths, valid_tickers = load_data_for_tickers(
        tickers, train_fraction=1-args.test_ratio)
    
    if not valid_tickers:
        print("ERROR: No valid tickers found. Exiting.")
        return
    
    print(f"Valid tickers for testing: {valid_tickers}")
    
    # 2. Align test DataFrames
    print("Aligning test DataFrames...")
    aligned_dfs_test = align_dataframes(dfs_test)
    
    # 3. Load environment configuration if available
    env_config = {}
    env_config_path = os.path.join(output_dir, 'env_config.json')
    if os.path.exists(env_config_path):
        env_config = load_environment_config(env_config_path)
        print(f"Loaded environment configuration from: {env_config_path}")
    
    # 4. Create test environment
    print("Creating test environment...")
    env = create_test_environment(valid_tickers, aligned_dfs_test, norm_params_paths, env_config)
    
    # 5. Create agent and load model
    print("Creating agent and loading model...")
    agent = create_agent(valid_tickers, model_path, len(norm_columns))
    
    # 6. Run benchmark test
    print("Running benchmark test...")
    metrics, logs = run_test(env, agent)
    
    # 7. Save and visualize results
    print("Saving and visualizing results...")
    save_test_results(metrics, logs, valid_tickers, output_dir)
    plot_benchmark_results(metrics, logs, valid_tickers, output_dir)
    
    # 8. Print summary of results
    print("\n===== BENCHMARK TEST RESULTS =====")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
    print(f"Initial Portfolio Value: ${metrics['initial_portfolio_value']:,.2f}")
    print(f"Total Steps: {metrics['total_steps']}")
    print(f"Average Reward: {metrics['avg_reward']:.4f}")
    print("===================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Test for Portfolio Agent")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model file (.pth)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save benchmark results")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated list of tickers to use (default: XLF,XLE,XLK,IHI,XLY)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Ratio of data to use for testing (default: 0.2)")
    
    args = parser.parse_args()
    main(args)