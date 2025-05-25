import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import yaml

def load_metrics(file_mappings):
    """Load metrics from jsonl files into a list of dataframes."""
    dfs = []
    for file_path, display_name in file_mappings.items():
        try:
            # Read jsonl file
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['run'] = display_name
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
            continue
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in file: {file_path}")
            continue
    
    return dfs

def plot_training_curves(dfs, output_dir):
    """Plot training metrics over time as separate figures."""
    # Create subfolder for training metrics
    training_dir = output_dir / 'training_metrics'
    training_dir.mkdir(exist_ok=True)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Set style
    plt.style.use('default')
    sns.set_theme()
    
    # 1. Loss over time
    plt.figure(figsize=(10, 6))
    for df in dfs:
        plt.plot(df['step'], df['loss'], label=df['run'].iloc[0])
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(training_dir / 'loss.png')
    plt.close()
    
    # 2. MFU over time
    plt.figure(figsize=(10, 6))
    for df in dfs:
        plt.plot(df['step'], df['mfu'], label=df['run'].iloc[0])
    plt.xlabel('Step')
    plt.ylabel('MFU (%)')
    plt.title('Model FLOPs Utilization')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(training_dir / 'mfu.png')
    plt.close()
    
    # 3. Tokens per second over time
    plt.figure(figsize=(10, 6))
    for df in dfs:
        plt.plot(df['step'], df['tokens_per_second'], label=df['run'].iloc[0])
    plt.xlabel('Step')
    plt.ylabel('Tokens/Second')
    plt.title('Training Speed')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(training_dir / 'tokens_per_second.png')
    plt.close()
    
    # 4. Training tokens percentage over time
    plt.figure(figsize=(10, 6))
    for df in dfs:
        plt.plot(df['step'], df['training_tokens_percentage'], label=df['run'].iloc[0])
    plt.xlabel('Step')
    plt.ylabel('Training Tokens (%)')
    plt.title('Training Tokens Percentage')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(training_dir / 'training_tokens_percentage.png')
    plt.close()

def plot_performance_comparison(dfs, output_dir):
    """Create separate bar plots for each performance metric."""
    # Create subfolder for performance metrics
    perf_dir = output_dir / 'performance_metrics'
    perf_dir.mkdir(exist_ok=True)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate average metrics for each run
    avg_metrics = combined_df.groupby('run').agg({
        'tokens_per_second': 'mean',
        'mfu': 'mean',
        'tflops': 'mean',
        'training_tokens_percentage': 'mean'
    }).reset_index()
    
    # Get colors from matplotlib's default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_dict = dict(zip(avg_metrics['run'].unique(), colors))
    
    # 1. Average Tokens per Second
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_metrics, x='run', y='tokens_per_second', palette=color_dict)
    plt.title('Average Training Speed')
    plt.ylabel('Tokens/Second')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(perf_dir / 'avg_tokens_per_second.png')
    plt.close()
    
    # 2. Average MFU
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_metrics, x='run', y='mfu', palette=color_dict)
    plt.title('Average Model FLOPs Utilization')
    plt.ylabel('MFU (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(perf_dir / 'avg_mfu.png')
    plt.close()
    
    # 3. Average TFLOPs
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_metrics, x='run', y='tflops', palette=color_dict)
    plt.title('Average TFLOPs')
    plt.ylabel('TFLOPs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(perf_dir / 'avg_tflops.png')
    plt.close()
    
    # 4. Average Training Tokens Percentage
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_metrics, x='run', y='training_tokens_percentage', palette=color_dict)
    plt.title('Average Training Tokens Percentage')
    plt.ylabel('Training Tokens (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(perf_dir / 'avg_training_tokens_percentage.png')
    plt.close()

def plot_time_analysis(dfs, output_dir):
    """Analyze and plot timing information as separate figures."""
    # Create subfolder for time analysis
    time_dir = output_dir / 'time_analysis'
    time_dir.mkdir(exist_ok=True)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 1. Time per step distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_df, x='run', y='time_delta')
    plt.title('Time per Step Distribution')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(time_dir / 'time_per_step.png')
    plt.close()
    
    # 2. Cumulative time
    plt.figure(figsize=(10, 6))
    for df in dfs:
        df['cumulative_time'] = df['time_delta'].cumsum()
        plt.plot(df['step'], df['cumulative_time'], label=df['run'].iloc[0])
    plt.title('Cumulative Training Time')
    plt.xlabel('Step')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(time_dir / 'cumulative_time.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate training plots from jsonl files using a config file')
    parser.add_argument('--config', required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory structure
    output_dir = Path(config['output']['base_dir']) / config['job_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    dfs = load_metrics(config['files'])
    
    if not dfs:
        print("Error: No data loaded from any of the specified files")
        return
    
    # Generate plots
    plot_training_curves(dfs, output_dir)
    plot_performance_comparison(dfs, output_dir)
    plot_time_analysis(dfs, output_dir)
    
    print(f"Plots have been saved to {output_dir}")
    print("Organized in subfolders:")
    print(f"  - {output_dir}/training_metrics/")
    print(f"  - {output_dir}/performance_metrics/")
    print(f"  - {output_dir}/time_analysis/")

if __name__ == "__main__":
    main()
