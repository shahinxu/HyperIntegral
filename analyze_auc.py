import re
import glob
import os
from collections import defaultdict
import numpy as np

def extract_auc_from_log(log_file, target_epoch=13500):
    runs_auc = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Split by run markers
    run_pattern = r'=== Run (\d+) started at.*?(?==== Run \d+|$)'
    runs = re.findall(run_pattern, content, re.DOTALL)
    
    # Alternative: split by explicit run markers
    run_sections = re.split(r'=== Run (\d+) started at', content)
    
    current_run = None
    for i in range(1, len(run_sections), 2):
        run_num = int(run_sections[i])
        run_content = run_sections[i+1] if i+1 < len(run_sections) else ""
        
        # Find the last occurrence of the target epoch in this run
        epoch_pattern = rf'Epoch {target_epoch},.*?\n.*?AUC \(2-edges\): ([\d.]+)(?:, AUC \(3-edges\): ([\d.]+))?(?:, AUC \(4-edges\): ([\d.]+))?(?:, AUC \(5-edges\): ([\d.]+))?(?:, AUC \(6-edges\): ([\d.]+))?(?:, AUC \(7-edges\): ([\d.]+))?'
        
        matches = list(re.finditer(epoch_pattern, run_content, re.DOTALL))
        
        if matches:
            # Take the last match for this epoch (in case it appears multiple times)
            last_match = matches[-1]
            auc_values = {}
            
            for order_idx, order in enumerate([2, 3, 4, 5, 6, 7], start=1):
                value = last_match.group(order_idx)
                if value:
                    auc_values[f'AUC_{order}edges'] = float(value)
            
            if auc_values:
                runs_auc[run_num] = auc_values
    
    return runs_auc

def analyze_logs_directory(log_dir, target_epoch=13500):
    """Analyze a single logs directory."""
    log_file = os.path.join(log_dir, 'all_runs.log')
    
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found")
        return {}
    
    print(f"\nAnalyzing: {log_dir}")
    runs_auc = extract_auc_from_log(log_file, target_epoch)
    
    if runs_auc:
        print(f"  Found {len(runs_auc)} runs with Epoch {target_epoch} data")
    else:
        print(f"  No data found for Epoch {target_epoch}")
    
    return runs_auc

def main():
    # Find all logs directories
    logs_dirs = sorted(glob.glob('/playpen-shared/zhenx/HyperPINN/logs_*'))
    
    if not logs_dirs:
        print("No logs_* directories found!")
        return
    
    print(f"Found {len(logs_dirs)} logs directories")
    
    # Collect all AUC values across all directories and runs
    all_auc_values = defaultdict(list)
    total_runs = 0
    
    for log_dir in logs_dirs:
        runs_auc = analyze_logs_directory(log_dir)
        
        for run_num, auc_values in runs_auc.items():
            total_runs += 1
            for key, value in auc_values.items():
                all_auc_values[key].append(value)
    
    # Calculate and display averages
    print("\n" + "="*80)
    print(f"SUMMARY: Analyzed {total_runs} total runs across {len(logs_dirs)} log directories")
    print("="*80)
    
    if not all_auc_values:
        print("No AUC data found!")
        return
    
    print("\nAverage AUC values across all runs:")
    print("-" * 80)
    
    for order in [2, 3, 4, 5, 6, 7]:
        key = f'AUC_{order}edges'
        if key in all_auc_values:
            values = all_auc_values[key]
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            count = len(values)
            
            print(f"AUC ({order}-edges): {mean:.4f} ± {std:.4f}  "
                  f"(min={min_val:.4f}, max={max_val:.4f}, n={count})")
    
    # Detailed breakdown by directory
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN BY DIRECTORY")
    print("="*80)
    
    for log_dir in logs_dirs:
        runs_auc = extract_auc_from_log(os.path.join(log_dir, 'all_runs.log'))
        if not runs_auc:
            continue
        
        print(f"\n{os.path.basename(log_dir)}:")
        
        # Aggregate by AUC type for this directory
        dir_auc_values = defaultdict(list)
        for run_num, auc_values in runs_auc.items():
            for key, value in auc_values.items():
                dir_auc_values[key].append(value)
        
        for order in [2, 3, 4, 5, 6, 7]:
            key = f'AUC_{order}edges'
            if key in dir_auc_values:
                values = dir_auc_values[key]
                mean = np.mean(values)
                std = np.std(values)
                print(f"  AUC ({order}-edges): {mean:.4f} ± {std:.4f} (n={len(values)})")

if __name__ == "__main__":
    main()
