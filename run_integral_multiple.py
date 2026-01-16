"""
Run Rossler_Integral_nn.py multiple times and save results
Usage: python run_integral_multiple.py [n_runs]
Example: python run_integral_multiple.py 20
"""
import subprocess
import os
from datetime import datetime
import numpy as np
import sys

def parse_auc_from_file(filepath):
    """Parse AUC scores from result file"""
    auc_scores = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '-edges:' in line:
                parts = line.strip().split(':')
                order_label = parts[0].strip()
                auc_val = parts[1].strip()
                if auc_val != 'N/A':
                    auc_scores[order_label] = float(auc_val)
    return auc_scores

def run_multiple_experiments(n_runs=10):
    """Run Rossler_Integral_nn.py multiple times"""
    
    print("="*80)
    print(f"Running Rossler_Integral_nn.py {n_runs} times")
    print("="*80)
    
    # Create main results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_results_dir = f"results_integral_multiple/{timestamp}"
    os.makedirs(main_results_dir, exist_ok=True)
    
    all_results = []
    
    for run_idx in range(1, n_runs + 1):
        print(f"\n{'='*80}")
        print(f"Run {run_idx}/{n_runs}")
        print(f"{'='*80}")
        
        # Run the script
        result = subprocess.run(
            ['python', 'Rossler_Integral_nn.py'],
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Find the most recent results directory
        results_dirs = sorted([d for d in os.listdir('results') if d.startswith('integral_')])
        if results_dirs:
            latest_result = results_dirs[-1]
            auc_file = f"results/{latest_result}/auc_scores.txt"
            
            if os.path.exists(auc_file):
                auc_scores = parse_auc_from_file(auc_file)
                all_results.append(auc_scores)
                
                # Save this run's result
                with open(f"{main_results_dir}/run_{run_idx}.txt", 'w') as f:
                    f.write(f"Run {run_idx}\n")
                    f.write(f"Source: results/{latest_result}/\n\n")
                    for order_label, auc_val in auc_scores.items():
                        f.write(f"  {order_label}: {auc_val:.4f}\n")
                
                print(f"\nRun {run_idx} AUC scores:")
                for order_label, auc_val in auc_scores.items():
                    print(f"  {order_label}: {auc_val:.4f}")
            else:
                print(f"Warning: AUC file not found for run {run_idx}")
        else:
            print(f"Warning: No results directory found for run {run_idx}")
    
    # Compute statistics
    print(f"\n{'='*80}")
    print("Computing Statistics")
    print(f"{'='*80}")
    
    # Organize results by order
    orders = ['2-edges', '3-edges', '4-edges', '5-edges', '6-edges', '7-edges']
    stats = {}
    
    for order in orders:
        values = [r[order] for r in all_results if order in r]
        if values:
            stats[order] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n': len(values)
            }
    
    # Save statistics
    stats_file = f"{main_results_dir}/statistics.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Rossler_Integral_nn.py - Multiple Runs Statistics\n")
        f.write(f"Number of runs: {n_runs}\n\n")
        f.write("Order    | Mean    | Std     | Min     | Max     | n\n")
        f.write("---------+---------+---------+---------+---------+----\n")
        for order in orders:
            if order in stats:
                s = stats[order]
                f.write(f"{order:8s} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['n']:2d}\n")
    
    # Print statistics
    print("\nStatistics:")
    print("Order    | Mean    | Std     | Min     | Max     | n")
    print("---------+---------+---------+---------+---------+----")
    for order in orders:
        if order in stats:
            s = stats[order]
            print(f"{order:8s} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['n']:2d}")
    
    # Save formatted results (mean ± std format)
    formatted_file = f"{main_results_dir}/formatted_results.txt"
    with open(formatted_file, 'w') as f:
        f.write("Rossler_Integral_nn - Multiple Runs\n")
        f.write(", ".join(orders) + "\n")
        
        stats_strs = []
        for order in orders:
            if order in stats:
                s = stats[order]
                stats_strs.append(f"{s['mean']:.4f} (±{s['std']:.4f})")
            else:
                stats_strs.append("N/A")
        f.write(", ".join(stats_strs) + "\n")
    
    # Print formatted results
    print("\nFormatted Results (mean ± std):")
    print(", ".join(orders))
    print(", ".join(stats_strs))
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {main_results_dir}/")
    print(f"{'='*80}")
    
    return all_results, stats

if __name__ == "__main__":
    # Get n_runs from command line argument or default to 10
    n_runs = 10
    if len(sys.argv) > 1:
        try:
            n_runs = int(sys.argv[1])
            print(f"Running {n_runs} experiments (from command line argument)")
        except ValueError:
            print(f"Invalid argument '{sys.argv[1]}', using default n_runs=10")
    else:
        print(f"Using default n_runs=10 (specify as: python run_integral_multiple.py N)")
    
    all_results, stats = run_multiple_experiments(n_runs=n_runs)
