import os
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
from collections import defaultdict, Counter
import warnings

# Import Seaborn first 
import seaborn as sns
sns.set_theme(style='whitegrid')

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc_file_defaults()

# SciPy
import scipy
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# Concurrent processing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# Set up the wandb API
api = wandb.Api(timeout=60)

plt.rcParams['figure.dpi'] = 300

fontdict = {'family': 'serif',
             'weight': 'normal',
             'size': 12,}

plt.rc('font', **fontdict)








def download_metrics(run, metrics, suppress_warnings=False):
    history = run.scan_history(page_size=100000)
    # Create a dictionary to store metrics data
    metrics_data = {}
  
    # Initialize metrics_data with empty lists for each available metric
    missing_metrics = []
    for metric in metrics:
        try:
            metrics_data[metric] = np.array([(row[metric],row['_step']) for row in history])
        except Exception as e:
            missing_metrics.append(metric)

    if missing_metrics != []:
        if not suppress_warnings:
            warnings.warn(f"Warning: Metrics {missing_metrics} not found in history for run of {run.group} with run id: {run.id}")
        
    return metrics_data



# multi processed downloading
def download_metrics_multithread(runs, metrics, threads_per_process=5, suppress_warnings=False):
    """Download the runs for a group using threading."""
    metric_list = []

    with ThreadPoolExecutor(max_workers=threads_per_process) as executor:
        futures = {executor.submit(download_metrics, run, metrics, suppress_warnings): run for run in runs}
        for i,future in enumerate(concurrent.futures.as_completed(futures)):
            losses = future.result()
            if losses is not None:
                metric_list.append(losses)

    return metric_list

def download_helper(group, wandb_project_name, metrics, threads_per_process, suppress_warnings=False):
    api = wandb.Api(timeout=120) # Create a new API instance within each process
    runs = api.runs(wandb_project_name, filters={"group": group})
    return group, download_metrics_multithread(runs, metrics, threads_per_process=threads_per_process, suppress_warnings=suppress_warnings)

def download_from_wandb(groups, 
                        wandb_project_name, 
                        metrics=["train loss"],
                        num_processes=3, 
                        threads_per_process=5,
                        suppress_warnings=False):

    grouped_losses = {}
    
    #in parallel download and parse
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        
        futures = {executor.submit(download_helper, 
                                   group, 
                                   wandb_project_name, 
                                   metrics, 
                                   threads_per_process,
                                   suppress_warnings): group 
                                   for group in groups}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(groups)):
            group, losses = future.result()
            grouped_losses[group] = losses
    
    #filter empty list
    final = {}
    for k,v in grouped_losses.items():
        if v == []:
            print(f"No runs found for group: \"{k}\", removing it from the final results.")
        else:
            final[k] = v

    return final
    


def merge_metrics(data):
    """
    Merge metrics from different runs for each group.
    
    Args:
        data: Dictionary where keys are group names and values are lists of dictionaries,
              each containing metrics for a run.
    
    Returns:
        Dictionary where keys are group names and values are dictionaries of merged metrics.
        Each metric contains arrays for mean, steps, std, and stderr.
    """
    merged_data = {}
    
    for group_name, runs in data.items():
        merged_data[group_name] = {}
        
        # Find all metrics across all runs
        all_metrics = set()
        for run in runs:
            all_metrics.update(run.keys())
        
        # Process each metric
        for metric in all_metrics:
            # Collect all runs that have this metric
            valid_runs = [run[metric] for run in runs if metric in run and run[metric].size > 0]
            
            if not valid_runs:
                # No valid data for this metric
                merged_data[group_name][metric] = {}
                continue
            
            # Extract steps and values from each run
            all_steps = []
            all_values = []
            
            for run_data in valid_runs:
                # Each run_data is a 2D array with shape [n_steps, 2]
                # The first column is the value, the second is the step
                if run_data.shape[1] == 2:
                    values = run_data[:, 0]
                    steps = run_data[:, 1]
                    all_steps.append(steps)
                    all_values.append(values)
            
            if not all_steps:
                merged_data[group_name][metric] = {}
                continue
                
            # Find common steps across all runs
            common_steps = set(all_steps[0])
            for steps in all_steps[1:]:
                common_steps = common_steps.intersection(set(steps))
            common_steps = sorted(list(common_steps))
            
            if not common_steps:
                merged_data[group_name][metric] = {}
                continue
            
            # Collect values for common steps
            values_at_common_steps = []
            for steps, values in zip(all_steps, all_values):
                steps_array = np.array(steps)
                values_for_run = []
                for step in common_steps:
                    idx = np.where(steps_array == step)[0]
                    if idx.size > 0:
                        values_for_run.append(values[idx[0]])
                    else:
                        values_for_run.append(np.nan)
                values_at_common_steps.append(values_for_run)
            
            # Convert to numpy array for calculations
            values_array = np.array(values_at_common_steps)
            
            # Calculate statistics
            mean_values = np.nanmean(values_array, axis=0)
            std_values = np.nanstd(values_array, axis=0)
            stderr_values = std_values / np.sqrt(np.sum(~np.isnan(values_array), axis=0))
            
            # Store results
            merged_data[group_name][metric] = {
                'mean': mean_values,
                'steps': np.array(common_steps),
                'std': std_values,
                'stderr': stderr_values
            }
    
    return merged_data

# Example usage
merged_data = merge_metrics(data)




def dowload_and_parse_runs(groups, 
                          wandb_project_name, 
                          metrics=["train loss"], 
                          num_processes=3, 
                          threads_per_process=3, 
                          suppress_warnings=False):

    final = download_from_wandb(
            groups=groups,
            wandb_project_name=wandb_project_name,
            metrics=metrics, 
            num_processes=num_processes, 
            threads_per_process=threads_per_process,
            suppress_warnings=suppress_warnings)

    return merge_metrics(final)
