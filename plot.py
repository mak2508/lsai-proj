import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import glob

def load_metrics(file_paths):
    """Load metrics from multiple jsonl files into a list of dataframes."""
    dfs = []
    for pattern in file_paths:
        # Expand wildcard pattern
        matching_files = glob.glob(pattern)
        if not matching_files:
            print(f"Warning: No files found matching pattern: {pattern}")
            continue
            
        for path in matching_files:
            # Extract run name from filename (e.g., 'baseline_20250125_020643.jsonl' -> 'baseline')
            run_name = Path(path).stem.split('_')[0]
            
            # Read jsonl file
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['run'] = run_name
            dfs.append(df)
    
    return dfs

def plot_training_curves(dfs, output_dir):
    """Plot training metrics over time."""
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Set style - use a built-in matplotlib style instead of seaborn
    plt.style.use('default')  # or 'classic', 'bmh', 'ggplot'
    sns.set_theme()  # This will set seaborn's default styling
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics Comparison', fontsize=16)
    
    # 1. Loss over time
    for df in dfs:
        axes[0,0].plot(df['step'], df['loss'], label=df['run'].iloc[0])
    axes[0,0].set_xlabel('Step')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 2. MFU over time
    for df in dfs:
        axes[0,1].plot(df['step'], df['mfu'], label=df['run'].iloc[0])
    axes[0,1].set_xlabel('Step')
    axes[0,1].set_ylabel('MFU (%)')
    axes[0,1].set_title('Model FLOPs Utilization')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # 3. Tokens per second over time
    for df in dfs:
        axes[1,0].plot(df['step'], df['tokens_per_second'], label=df['run'].iloc[0])
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('Tokens/Second')
    axes[1,0].set_title('Training Speed')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 4. Training tokens percentage over time
    for df in dfs:
        axes[1,1].plot(df['step'], df['training_tokens_percentage'], label=df['run'].iloc[0])
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('Training Tokens (%)')
    axes[1,1].set_title('Training Tokens Percentage')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png')
    plt.close()

def plot_performance_comparison(dfs, output_dir):
    """Create box plots comparing performance metrics across runs."""
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate average metrics for each run
    avg_metrics = combined_df.groupby('run').agg({
        'tokens_per_second': 'mean',
        'mfu': 'mean',
        'tflops': 'mean',
        'training_tokens_percentage': 'mean'
    }).reset_index()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Metrics Comparison', fontsize=16)
    
    # Get colors from matplotlib's default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_dict = dict(zip(avg_metrics['run'].unique(), colors))
    
    # 1. Average Tokens per Second
    sns.barplot(data=avg_metrics, x='run', y='tokens_per_second', ax=axes[0,0], palette=color_dict)
    axes[0,0].set_title('Average Training Speed')
    axes[0,0].set_ylabel('Tokens/Second')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Average MFU
    sns.barplot(data=avg_metrics, x='run', y='mfu', ax=axes[0,1], palette=color_dict)
    axes[0,1].set_title('Average Model FLOPs Utilization')
    axes[0,1].set_ylabel('MFU (%)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Average TFLOPs
    sns.barplot(data=avg_metrics, x='run', y='tflops', ax=axes[1,0], palette=color_dict)
    axes[1,0].set_title('Average TFLOPs')
    axes[1,0].set_ylabel('TFLOPs')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Average Training Tokens Percentage
    sns.barplot(data=avg_metrics, x='run', y='training_tokens_percentage', ax=axes[1,1], palette=color_dict)
    axes[1,1].set_title('Average Training Tokens Percentage')
    axes[1,1].set_ylabel('Training Tokens (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png')
    plt.close()

def plot_time_analysis(dfs, output_dir):
    """Analyze and plot timing information."""
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Time Analysis', fontsize=16)
    
    # 1. Time per step distribution
    sns.boxplot(data=combined_df, x='run', y='time_delta', ax=axes[0])
    axes[0].set_title('Time per Step Distribution')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Cumulative time
    for df in dfs:
        df['cumulative_time'] = df['time_delta'].cumsum()
        axes[1].plot(df['step'], df['cumulative_time'], label=df['run'].iloc[0])
    axes[1].set_title('Cumulative Training Time')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_analysis.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate training plots from jsonl files')
    parser.add_argument('--input_files', nargs='+', required=True, help='List of jsonl files to plot')
    parser.add_argument('--output_dir', required=True, help='Directory to save plots')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    dfs = load_metrics(args.input_files)
    
    # Generate plots
    plot_training_curves(dfs, args.output_dir)
    plot_performance_comparison(dfs, args.output_dir)
    plot_time_analysis(dfs, args.output_dir)
    
    print(f"Plots have been saved to {args.output_dir}")

if __name__ == "__main__":
    main()
