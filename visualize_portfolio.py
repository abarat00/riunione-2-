"""
Enhanced Portfolio Visualization Script

Usage:
    python visualize_portfolio.py --results_dir results/baseline_full

Description:
    This script creates improved visualizations for portfolio performance analysis.
    It loads data from test logs and training results to create comprehensive reports.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import glob
from datetime import datetime

def load_numpy_safely(file_path):
    """Load numpy file safely, returning None if file doesn't exist or can't be loaded."""
    if not os.path.exists(file_path):
        return None
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_test_data(results_dir):
    """Load test data from the specified directory."""
    test_dir = os.path.join(results_dir, 'test')
    data = {}
    
    # Load numpy arrays
    data['action_log'] = load_numpy_safely(os.path.join(test_dir, 'action_log.npy'))
    data['position_log'] = load_numpy_safely(os.path.join(test_dir, 'position_log.npy'))
    data['portfolio_value_log'] = load_numpy_safely(os.path.join(test_dir, 'portfolio_value_log.npy'))
    data['rewards_log'] = load_numpy_safely(os.path.join(test_dir, 'rewards_log.npy'))
    
    # Load test metrics
    test_metrics_path = os.path.join(test_dir, 'test_metrics.json')
    if os.path.exists(test_metrics_path):
        try:
            with open(test_metrics_path, 'r') as f:
                data['test_metrics'] = json.load(f)
        except Exception as e:
            print(f"Error loading test metrics: {e}")
            data['test_metrics'] = {}
    else:
        data['test_metrics'] = {}
    
    # Load test details CSV if available
    test_details_path = os.path.join(test_dir, 'test_details.csv')
    if os.path.exists(test_details_path):
        try:
            data['test_details'] = pd.read_csv(test_details_path)
        except Exception as e:
            print(f"Error loading test details: {e}")
            data['test_details'] = None
    else:
        data['test_details'] = None
    
    return data

def load_training_data(results_dir):
    """Load training data from the specified directory."""
    data = {}
    
    # Load training results CSV
    training_results_path = os.path.join(results_dir, 'training_results.csv')
    if os.path.exists(training_results_path):
        try:
            data['training_results'] = pd.read_csv(training_results_path)
        except Exception as e:
            print(f"Error loading training results: {e}")
            data['training_results'] = None
    else:
        data['training_results'] = None
    
    # Find latest checkpoint
    weights_dir = os.path.join(results_dir, 'weights')
    if os.path.exists(weights_dir):
        checkpoint_files = glob.glob(os.path.join(weights_dir, 'checkpoint_ep*.pt'))
        if checkpoint_files:
            latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(os.path.basename(x).split('ep')[1].split('.')[0]))[-1]
            try:
                # Load with torch.load, but only extract metrics
                import torch
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                if 'metrics' in checkpoint:
                    data['checkpoint_metrics'] = checkpoint['metrics']
                else:
                    data['checkpoint_metrics'] = {}
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                data['checkpoint_metrics'] = {}
        else:
            data['checkpoint_metrics'] = {}
    else:
        data['checkpoint_metrics'] = {}
    
    # Find raw_results.pkl if available
    raw_results_path = os.path.join(results_dir, 'raw_results.pkl')
    if os.path.exists(raw_results_path):
        try:
            import pickle
            with open(raw_results_path, 'rb') as f:
                data['raw_results'] = pickle.load(f)
        except Exception as e:
            print(f"Error loading raw results: {e}")
            data['raw_results'] = {}
    else:
        data['raw_results'] = {}
    
    return data

def load_configuration(results_dir):
    """Load configuration data from the specified directory."""
    config = {}
    
    # Load environment configuration
    env_config_path = os.path.join(results_dir, 'env_config.json')
    if os.path.exists(env_config_path):
        try:
            with open(env_config_path, 'r') as f:
                config['env_config'] = json.load(f)
        except Exception as e:
            print(f"Error loading environment config: {e}")
            config['env_config'] = {}
    else:
        config['env_config'] = {}
    
    # Load agent configuration
    agent_config_path = os.path.join(results_dir, 'agent_config.json')
    if os.path.exists(agent_config_path):
        try:
            with open(agent_config_path, 'r') as f:
                config['agent_config'] = json.load(f)
        except Exception as e:
            print(f"Error loading agent config: {e}")
            config['agent_config'] = {}
    else:
        config['agent_config'] = {}
    
    # Load profile configuration
    profile_config_path = os.path.join(results_dir, 'profile_config.json')
    if os.path.exists(profile_config_path):
        try:
            with open(profile_config_path, 'r') as f:
                config['profile_config'] = json.load(f)
        except Exception as e:
            print(f"Error loading profile config: {e}")
            config['profile_config'] = {}
    else:
        config['profile_config'] = {}
    
    return config

def plot_portfolio_performance(test_data, training_data, config, results_dir):
    """Create comprehensive portfolio performance visualization."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Set up figure
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig)
    
    # Extract tickers
    tickers = None
    if test_data.get('test_details') is not None:
        position_columns = [col for col in test_data['test_details'].columns if col.startswith('position_')]
        if position_columns:
            tickers = [col.split('_')[1] for col in position_columns]
    
    if tickers is None and 'env_config' in config:
        tickers = config['env_config'].get('tickers', ['Unknown'])
    
    if tickers is None:
        tickers = ['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
    
    # 1. Portfolio Value Over Time (Test)
    ax1 = fig.add_subplot(gs[0, :2])
    if test_data.get('portfolio_value_log') is not None:
        portfolio_values = test_data['portfolio_value_log']
        ax1.plot(portfolio_values, color='darkblue', linewidth=2)
        ax1.axhline(y=portfolio_values[0], color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Portfolio Value During Test', fontsize=14)
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Value ($)', fontsize=12)
        
        # Add annotations for key metrics
        if len(portfolio_values) > 0:
            final_value = portfolio_values[-1]
            initial_value = portfolio_values[0]
            max_value = np.max(portfolio_values)
            min_value = np.min(portfolio_values)
            
            ax1.annotate(f'Final: ${final_value:,.2f}', 
                        xy=(len(portfolio_values)-1, final_value),
                        xytext=(len(portfolio_values)-len(portfolio_values)//4, final_value+1000),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10)
            ax1.annotate(f'Initial: ${initial_value:,.2f}',
                        xy=(0, initial_value),
                        xytext=(len(portfolio_values)//8, initial_value-5000),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10)
            
            # Add returns text
            returns_pct = (final_value / initial_value - 1) * 100
            ax1.text(0.02, 0.02, f'Total Return: {returns_pct:.2f}%', 
                     transform=ax1.transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.8))
    else:
        ax1.text(0.5, 0.5, "No portfolio value data available", 
                 ha='center', va='center', fontsize=12)
    
    # 2. Positions Over Time
    ax2 = fig.add_subplot(gs[1, :2])
    if test_data.get('position_log') is not None:
        position_log = test_data['position_log']
        for i in range(position_log.shape[1]):
            ticker_idx = min(i, len(tickers)-1)  # Avoid index error
            ax2.plot(position_log[:, i], label=tickers[ticker_idx])
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Asset Positions Over Time', fontsize=14)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Position', fontsize=12)
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, "No position data available", 
                 ha='center', va='center', fontsize=12)
    
    # 3. Trading Activity (Actions)
    ax3 = fig.add_subplot(gs[2, :2])
    if test_data.get('action_log') is not None:
        action_log = test_data['action_log']
        
        # Calculate absolute action size at each step
        action_magnitudes = np.abs(action_log)
        total_action_magnitude = np.sum(action_magnitudes, axis=1)
        
        # Create a heatmap-like visualization
        cmap = plt.cm.viridis
        for i in range(action_log.shape[1]):
            ticker_idx = min(i, len(tickers)-1)  # Avoid index error
            actions = action_log[:, i]
            # Use color for direction (buy/sell) and alpha for magnitude
            color = 'green'  # For buys
            ax3.fill_between(range(len(actions)), 0, actions, 
                           where=(actions > 0), alpha=0.5,
                           color=color, label=f"{tickers[ticker_idx]} (buy)" if i == 0 else "")
            
            color = 'red'  # For sells
            ax3.fill_between(range(len(actions)), 0, actions, 
                           where=(actions < 0), alpha=0.5,
                           color=color, label=f"{tickers[ticker_idx]} (sell)" if i == 0 else "")
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Trading Actions Over Time', fontsize=14)
        ax3.set_xlabel('Step', fontsize=12)
        ax3.set_ylabel('Action Magnitude', fontsize=12)
        
        # Create a simplified legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.5, label='Buy Actions'),
            Patch(facecolor='red', alpha=0.5, label='Sell Actions')
        ]
        ax3.legend(handles=legend_elements, loc='upper right')
    else:
        ax3.text(0.5, 0.5, "No action data available", 
                 ha='center', va='center', fontsize=12)
    
    # 4. Cumulative Rewards
    ax4 = fig.add_subplot(gs[3, :2])
    if test_data.get('rewards_log') is not None:
        rewards = test_data['rewards_log']
        cumulative_rewards = np.cumsum(rewards)
        ax4.plot(cumulative_rewards, color='purple', linewidth=2)
        ax4.set_title('Cumulative Rewards', fontsize=14)
        ax4.set_xlabel('Step', fontsize=12)
        ax4.set_ylabel('Cumulative Reward', fontsize=12)
        
        # Add final reward info
        if len(cumulative_rewards) > 0:
            final_reward = cumulative_rewards[-1]
            ax4.text(0.02, 0.02, f'Final Cumulative Reward: {final_reward:.2f}', 
                     transform=ax4.transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, "No reward data available", 
                 ha='center', va='center', fontsize=12)
    
    # 5. Position Correlation Matrix
    ax5 = fig.add_subplot(gs[0, 2])
    if test_data.get('position_log') is not None and test_data['position_log'].shape[1] > 1:
        position_log = test_data['position_log']
        corr_matrix = np.corrcoef(position_log.T)
        
        # Create a correlation heatmap
        im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                text = ax5.text(j, i, f"{corr_matrix[i, j]:.2f}",
                               ha="center", va="center", 
                               color="black" if abs(corr_matrix[i, j]) < 0.7 else "white",
                               fontsize=10)
        
        # Set ticks and labels
        ax5.set_xticks(np.arange(len(tickers[:position_log.shape[1]])))
        ax5.set_yticks(np.arange(len(tickers[:position_log.shape[1]])))
        ax5.set_xticklabels(tickers[:position_log.shape[1]])
        ax5.set_yticklabels(tickers[:position_log.shape[1]])
        plt.setp(ax5.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax5.set_title('Position Correlation Matrix', fontsize=14)
        fig.colorbar(im, ax=ax5)
    else:
        ax5.text(0.5, 0.5, "Position correlation cannot be calculated", 
                 ha='center', va='center', fontsize=12)
    
    # 6. Training vs Test Performance
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Extract training and test metrics
    train_portfolio_value = None
    test_portfolio_value = None
    train_sharpe = None
    test_sharpe = None
    
    # From checkpoint metrics
    if 'checkpoint_metrics' in training_data and training_data['checkpoint_metrics']:
        if 'final_portfolio_values' in training_data['checkpoint_metrics']:
            if isinstance(training_data['checkpoint_metrics']['final_portfolio_values'], list):
                train_portfolio_value = training_data['checkpoint_metrics']['final_portfolio_values'][-1]
            else:
                train_portfolio_value = list(training_data['checkpoint_metrics']['final_portfolio_values'])[-1]
        
        if 'final_sharpe_ratios' in training_data['checkpoint_metrics']:
            if isinstance(training_data['checkpoint_metrics']['final_sharpe_ratios'], list):
                train_sharpe = training_data['checkpoint_metrics']['final_sharpe_ratios'][-1]
            else:
                train_sharpe = list(training_data['checkpoint_metrics']['final_sharpe_ratios'])[-1]
    
    # From test metrics
    if 'test_metrics' in test_data and test_data['test_metrics']:
        test_portfolio_value = test_data['test_metrics'].get('final_portfolio_value')
        test_sharpe = test_data['test_metrics'].get('sharpe_ratio')
    
    # Calculate returns
    initial_value = 100000  # Default initial capital
    if 'env_config' in config and 'initial_capital' in config['env_config']:
        initial_value = float(config['env_config']['initial_capital'])
    
    train_return = ((train_portfolio_value / initial_value) - 1) * 100 if train_portfolio_value else None
    test_return = ((test_portfolio_value / initial_value) - 1) * 100 if test_portfolio_value else None
    
    # Plot the comparison
    metrics = ['Portfolio Value', 'Return (%)', 'Sharpe Ratio']
    if train_portfolio_value is not None and test_portfolio_value is not None:
        train_values = [train_portfolio_value, train_return, train_sharpe]
        test_values = [test_portfolio_value, test_return, test_sharpe]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax6.bar(x - width/2, train_values, width, label='Training')
        ax6.bar(x + width/2, test_values, width, label='Test')
        
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics)
        ax6.set_title('Training vs Test Performance', fontsize=14)
        ax6.legend()
        
        # Add data labels
        for i, v in enumerate(train_values):
            ax6.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)
        for i, v in enumerate(test_values):
            ax6.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)
    else:
        ax6.text(0.5, 0.5, "Comparison data not available", 
                 ha='center', va='center', fontsize=12)
    
    # 7. Position Distribution
    ax7 = fig.add_subplot(gs[2, 2])
    if test_data.get('position_log') is not None:
        position_log = test_data['position_log']
        
        # Create violin plots for position distributions
        violin_parts = ax7.violinplot([position_log[:, i] for i in range(position_log.shape[1])],
                                    showmeans=True, showmedians=True)
        
        # Set colors for violin plots
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(plt.cm.tab10(i % 10))
            pc.set_alpha(0.7)
        
        ax7.set_xticks(np.arange(1, position_log.shape[1] + 1))
        ax7.set_xticklabels(tickers[:position_log.shape[1]])
        ax7.set_ylabel('Position Value')
        ax7.set_title('Position Distribution', fontsize=14)
        
        # Add grid for readability
        ax7.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax7.text(0.5, 0.5, "No position data available", 
                 ha='center', va='center', fontsize=12)
    
    # 8. Key Metrics / Summary
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.axis('off')
    
    # Collect all relevant metrics
    summary_text = "Portfolio Performance Summary\n" + "="*30 + "\n\n"
    
    # Test metrics
    if 'test_metrics' in test_data and test_data['test_metrics']:
        summary_text += "TEST METRICS:\n"
        metrics = test_data['test_metrics']
        summary_text += f"Total Return: {metrics.get('total_return', 'N/A'):.2f}%\n"
        summary_text += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}\n"
        summary_text += f"Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2f}%\n"
        summary_text += f"Volatility: {metrics.get('volatility', 'N/A'):.2f}%\n"
        summary_text += f"Final Value: ${metrics.get('final_portfolio_value', 'N/A'):,.2f}\n\n"
    
    # Trading statistics
    if test_data.get('action_log') is not None:
        action_log = test_data['action_log']
        significant_trades = np.sum(np.abs(action_log) > 0.001)
        total_steps = action_log.shape[0]
        
        summary_text += "TRADING STATISTICS:\n"
        summary_text += f"Total Steps: {total_steps}\n"
        summary_text += f"Significant Trades: {significant_trades}\n"
        summary_text += f"Trading Frequency: {significant_trades/total_steps*100:.2f}%\n\n"
    
    # Position statistics
    if test_data.get('position_log') is not None:
        position_log = test_data['position_log']
        
        summary_text += "POSITION STATISTICS:\n"
        for i in range(min(position_log.shape[1], len(tickers))):
            ticker = tickers[i]
            pos = position_log[:, i]
            summary_text += f"{ticker}: Avg={np.mean(pos):.4f}, Max={np.max(pos):.4f}, Min={np.min(pos):.4f}\n"
    
    # Configuration
    summary_text += "\nCONFIGURATION:\n"
    if 'profile_config' in config and 'env' in config['profile_config']:
        env_config = config['profile_config']['env']
        summary_text += f"Commission Rate: {env_config.get('commission_rate', 'N/A')}\n"
        summary_text += f"Max Position per Asset: {env_config.get('max_pos_per_asset', 'N/A')}\n"
        summary_text += f"Profile: {config.get('profile', 'custom')}\n"
    
    ax8.text(0, 1, summary_text, fontsize=10, va='top', 
             family='monospace', transform=ax8.transAxes)
    
    # Add overall title and save figure
    plt.tight_layout()
    fig.suptitle('Enhanced Portfolio Performance Analysis', fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.94)
    
    output_path = os.path.join(results_dir, 'comprehensive_analysis.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Comprehensive analysis saved to: {output_path}")
    
    return fig

def create_detailed_report(test_data, training_data, config, results_dir):
    """Create a detailed HTML report with all performance metrics."""
    import jinja2
    
    # Prepare the template
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Portfolio Performance Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #2c3e50; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .metric-card { background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }
            .metric-title { font-weight: bold; color: #2c3e50; }
            .metric-value { font-size: 24px; color: #3498db; margin: 10px 0; }
            .metric-subtitle { color: #7f8c8d; font-size: 14px; }
            .grid-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }
            @media (max-width: 768px) { .grid-container { grid-template-columns: 1fr; } }
            .warning { background-color: #ffe6e6; border-left: 4px solid #e74c3c; }
            .success { background-color: #e6ffe6; border-left: 4px solid #2ecc71; }
        </style>
    </head>
    <body>
        <h1>Portfolio Performance Report</h1>
        <p>Generated on {{ generation_date }}</p>
        
        <h2>Performance Summary</h2>
        <div class="grid-container">
            {% if test_metrics %}
            <div class="metric-card {% if test_metrics.total_return > 0 %}success{% else %}warning{% endif %}">
                <div class="metric-title">Total Return</div>
                <div class="metric-value">{{ test_metrics.total_return|round(2) }}%</div>
                <div class="metric-subtitle">Performance relative to initial investment</div>
            </div>
            <div class="metric-card {% if test_metrics.sharpe_ratio > 1 %}success{% elif test_metrics.sharpe_ratio > 0 %}neutral{% else %}warning{% endif %}">
                <div class="metric-title">Sharpe Ratio</div>
                <div class="metric-value">{{ test_metrics.sharpe_ratio|round(4) }}</div>
                <div class="metric-subtitle">Risk-adjusted performance metric</div>
            </div>
            <div class="metric-card {% if test_metrics.max_drawdown > -20 %}neutral{% else %}warning{% endif %}">
                <div class="metric-title">Maximum Drawdown</div>
                <div class="metric-value">{{ test_metrics.max_drawdown|round(2) }}%</div>
                <div class="metric-subtitle">Largest portfolio decline from peak to trough</div>
            </div>
            {% else %}
            <div class="metric-card warning">
                <div class="metric-title">No Test Metrics Available</div>
                <div class="metric-subtitle">Data not found or could not be loaded</div>
            </div>
            {% endif %}
        </div>
        
        <h2>Trading Activity Analysis</h2>
        {% if action_stats %}
        <div class="grid-container">
            <div class="metric-card">
                <div class="metric-title">Total Steps</div>
                <div class="metric-value">{{ action_stats.total_steps }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Significant Trades</div>
                <div class="metric-value">{{ action_stats.significant_trades }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Trading Frequency</div>
                <div class="metric-value">{{ action_stats.trading_frequency|round(2) }}%</div>
            </div>
        </div>
        {% else %}
        <p>No trading activity data available.</p>
        {% endif %}
        
        <h2>Position Analysis</h2>
        {% if position_stats %}
        <table>
            <tr>
                <th>Asset</th>
                <th>Average Position</th>
                <th>Maximum Position</th>
                <th>Minimum Position</th>
                <th>Position Variance</th>
            </tr>
            {% for asset in position_stats %}
            <tr>
                <td>{{ asset.ticker }}</td>
                <td>{{ asset.avg|round(4) }}</td>
                <td>{{ asset.max|round(4) }}</td>
                <td>{{ asset.min|round(4) }}</td>
                <td>{{ asset.var|round(4) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p>No position data available.</p>
        {% endif %}
        
        <h2>Position Correlations</h2>
        {% if position_correlations %}
        <table>
            <tr>
                <th>Asset Pair</th>
                <th>Correlation</th>
                <th>Interpretation</th>
            </tr>
            {% for corr in position_correlations %}
            <tr>
                <td>{{ corr.pair }}</td>
                <td>{{ corr.value|round(4) }}</td>
                <td>
                    {% if corr.value > 0.7 %}
                    Strong positive correlation
                    {% elif corr.value > 0.3 %}
                    Moderate positive correlation
                    {% elif corr.value > -0.3 %}
                    Weak or no correlation
                    {% elif corr.value > -0.7 %}
                    Moderate negative correlation
                    {% else %}
                    Strong negative correlation
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p>No correlation data available.</p>
        {% endif %}
        
        <h2>Configuration</h2>
        {% if config_info %}
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            {% for param, value in config_info.items() %}
            <tr>
                <td>{{ param }}</td>
                <td>{{ value }}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p>No configuration data available.</p>
        {% endif %}
        
        <h2>Training-Test Comparison</h2>
        {% if comparison %}
        <table>
            <tr>
                <th>Metric</th>
                <th>Training</th>
                <th>Test</th>
                <th>Difference</th>
            </tr>
            {% for comp in comparison %}
            <tr>
                <td>{{ comp.metric }}</td>
                <td>{{ comp.train|round(4) }}</td>
                <td>{{ comp.test|round(4) }}</td>
                <td>{{ comp.diff|round(4) }} ({{ comp.pct_diff|round(2) }}%)</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p>No comparison data available.</p>
        {% endif %}
    </body>
    </html>
    """
    
    # Prepare the data for the template
    template_data = {
        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'test_metrics': test_data.get('test_metrics', {}),
        'config_info': {}
    }
    
    # Extract config info
    if 'profile_config' in config and 'env' in config['profile_config']:
        template_data['config_info'] = {
            'Profile': config.get('profile', 'custom'),
            'Commission Rate': config['profile_config']['env'].get('commission_rate', 'N/A'),
            'Max Position per Asset': config['profile_config']['env'].get('max_pos_per_asset', 'N/A'),
            'Free Trades': config['profile_config']['env'].get('free_trades', 'N/A'),
            'Reward Return Weight': config['profile_config']['env'].get('reward_return_weight', 'N/A'),
            'Position Stability Factor': config['profile_config']['env'].get('position_stability_factor', 'N/A'),
            'Diversification Factor': config['profile_config']['env'].get('diversification_factor', 'N/A')
        }
    
    # Calculate action stats
    if test_data.get('action_log') is not None:
        action_log = test_data['action_log']
        significant_trades = np.sum(np.abs(action_log) > 0.001)
        total_steps = action_log.shape[0]
        
        template_data['action_stats'] = {
            'total_steps': total_steps,
            'significant_trades': significant_trades,
            'trading_frequency': significant_trades/total_steps*100
        }
    
    # Calculate position stats
    if test_data.get('position_log') is not None and 'test_metrics' in test_data:
        position_log = test_data['position_log']
        tickers = []
        
        if 'env_config' in config and 'tickers' in config['env_config']:
            tickers = config['env_config']['tickers']
        else:
            # Create generic ticker names
            tickers = [f'Asset{i+1}' for i in range(position_log.shape[1])]
        
        position_stats = []
        for i in range(min(position_log.shape[1], len(tickers))):
            ticker = tickers[i]
            pos = position_log[:, i]
            position_stats.append({
                'ticker': ticker,
                'avg': np.mean(pos),
                'max': np.max(pos),
                'min': np.min(pos),
                'var': np.var(pos)
            })
        
        template_data['position_stats'] = position_stats
        
        # Calculate correlations
        if position_log.shape[1] > 1:
            corr_matrix = np.corrcoef(position_log.T)
            correlations = []
            
            for i in range(position_log.shape[1]):
                for j in range(i+1, position_log.shape[1]):
                    if i < len(tickers) and j < len(tickers):
                        correlations.append({
                            'pair': f"{tickers[i]}-{tickers[j]}",
                            'value': corr_matrix[i, j]
                        })
            
            template_data['position_correlations'] = correlations
    
    # Calculate training-test comparison
    if 'checkpoint_metrics' in training_data and training_data['checkpoint_metrics'] and 'test_metrics' in test_data:
        train_data = training_data['checkpoint_metrics']
        test_data_metrics = test_data['test_metrics']
        
        comparison = []
        
        # Portfolio value
        if 'final_portfolio_values' in train_data and 'final_portfolio_value' in test_data_metrics:
            if isinstance(train_data['final_portfolio_values'], list):
                train_value = train_data['final_portfolio_values'][-1]
            else:
                train_value = list(train_data['final_portfolio_values'])[-1]
            
            test_value = test_data_metrics['final_portfolio_value']
            diff = test_value - train_value
            pct_diff = (diff / abs(train_value)) * 100 if train_value != 0 else float('inf')
            
            comparison.append({
                'metric': 'Portfolio Value',
                'train': train_value,
                'test': test_value,
                'diff': diff,
                'pct_diff': pct_diff
            })
        
        # Sharpe ratio
        if 'final_sharpe_ratios' in train_data and 'sharpe_ratio' in test_data_metrics:
            if isinstance(train_data['final_sharpe_ratios'], list):
                train_sharpe = train_data['final_sharpe_ratios'][-1]
            else:
                train_sharpe = list(train_data['final_sharpe_ratios'])[-1]
                
            test_sharpe = test_data_metrics['sharpe_ratio']
            diff = test_sharpe - train_sharpe
            pct_diff = (diff / abs(train_sharpe)) * 100 if train_sharpe != 0 else float('inf')
            
            comparison.append({
                'metric': 'Sharpe Ratio',
                'train': train_sharpe,
                'test': test_sharpe,
                'diff': diff,
                'pct_diff': pct_diff
            })
        
        template_data['comparison'] = comparison
    
    # Render the template
    template = jinja2.Template(template_str)
    html_report = template.render(**template_data)
    
    # Save the report
    report_path = os.path.join(results_dir, 'performance_report.html')
    with open(report_path, 'w') as f:
        f.write(html_report)
    
    print(f"Detailed HTML report saved to: {report_path}")

def main(args):
    """Main function to create visualizations and reports."""
    results_dir = args.results_dir
    
    print(f"Analyzing results in directory: {results_dir}")
    
    # Load all data
    test_data = load_test_data(results_dir)
    training_data = load_training_data(results_dir)
    config = load_configuration(results_dir)
    
    # Create visualizations
    plot_portfolio_performance(test_data, training_data, config, results_dir)
    
    # Create HTML report if specified
    if args.html_report:
        try:
            create_detailed_report(test_data, training_data, config, results_dir)
        except ImportError:
            print("WARNING: Jinja2 not installed. HTML report generation skipped.")
            print("To create HTML reports, install Jinja2: pip install jinja2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Portfolio Visualization Script")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Directory containing results to visualize")
    parser.add_argument("--html_report", action="store_true",
                        help="Generate detailed HTML report (requires jinja2)")
    
    args = parser.parse_args()
    main(args)