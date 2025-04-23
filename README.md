# Welcome to Wandb Matlplotlib plotting!


# Example Usage
"""
# Download and parse runs from Weights & Biases
# ---------------------------------------------
# This function downloads experiment data from W&B and merges metrics across runs

# Basic usage:
merged_data = dowload_and_parse_runs(
    groups=["experiment1", "experiment2"],  # List of experiment group names to download
    wandb_project_name="my-project",        # Your W&B project name
)

# Advanced usage:
merged_data = dowload_and_parse_runs(
    groups=["experiment1", "experiment2"],
    wandb_project_name="my-project",
    metrics=["train loss", "test loss", "accuracy"],  # Metrics to download
    num_processes=4,                                  # Parallel processing settings
    threads_per_process=2,
    suppress_warnings=True                            # Hide warnings about missing metrics
)

# The returned data structure contains merged statistics across runs:
# {
#   'experiment1': {
#     'train loss': {
#       'mean': array([...]),     # Mean values at each step
#       'steps': array([...]),    # Common steps across all runs
#       'std': array([...]),      # Standard deviation
#       'stderr': array([...])    # Standard error
#     },
#     'test loss': { ... }
#   },
#   'experiment2': { ... }
# }
"""
