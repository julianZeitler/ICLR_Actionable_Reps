import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
from datetime import datetime
import jax.numpy as jnp
import jax
from typing import Dict, List, Optional, Tuple, Any
import itertools

from NRT_functions import losses
from NRT_functions import helper_functions
from NRT_functions import plotters
from PlaneSequential import run_plane_sequential_optimization
from NRT_functions.analysis import quantitative_analysis_seq, compute_g_at_positions
from DataHandling import TrajectoryDataset, TrajectoryGenerator

def generate_2d_plots(g0, om, S, parameters, savepath, counter):
    figures = {}

    fig_freq, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(om[:, 0], om[:, 1], s=100, alpha=0.6)
    ax.set_xlabel('$\omega_x$', fontsize=12)
    ax.set_ylabel('$\omega_y$', fontsize=12)
    ax.set_title('Frequency vectors', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    fig_freq.tight_layout()

    figures['freq_plot'] = fig_freq
    fig_freq.savefig(os.path.join(savepath, f"freq_plot_{counter}.png"))


    fig_S, axes = plt.subplots(1, 3, figsize=(12, 3))

    im1 = axes[0].imshow(S, cmap='RdBu_r', aspect='equal')
    axes[0].set_title('S')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0])

    im1 = axes[1].imshow(jnp.linalg.inv(S), cmap='RdBu_r', aspect='equal')
    axes[1].set_title('S^-1')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[1])

    im1 = axes[2].imshow(S @ jnp.linalg.inv(S), cmap='RdBu_r', aspect='equal')
    axes[2].set_title('S @ S^-1')
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[2])
    fig_S.tight_layout()

    figures['S'] = fig_S
    fig_S.savefig(os.path.join(savepath, f"S_analysis_{counter}.png"))


    # Create grid of positions
    res = 70
    phi_small = np.linspace(-np.pi, np.pi, res)/6
    phi_small = np.meshgrid(phi_small, phi_small)
    phi_small = np.hstack([np.ndarray.flatten(phi_small[0])[:,None],
                                np.ndarray.flatten(phi_small[1])[:,None]])

    phi_medium = np.linspace(-np.pi, np.pi, res)/2
    phi_medium = np.meshgrid(phi_medium, phi_medium)
    phi_medium = np.hstack([np.ndarray.flatten(phi_medium[0])[:,None], 
                           np.ndarray.flatten(phi_medium[1])[:,None]])
    
    phi_large = np.linspace(-np.pi, np.pi, res)*parameters.get("pos_lengthscale", 2)
    phi_large = np.meshgrid(phi_large, phi_large)
    phi_large = np.hstack([np.ndarray.flatten(phi_large[0])[:,None], 
                           np.ndarray.flatten(phi_large[1])[:,None]])
    
    V_small = np.array(compute_g_at_positions(g0, om, S, phi_small))
    V_medium = np.array(compute_g_at_positions(g0, om, S, phi_medium))
    V_large = np.array(compute_g_at_positions(g0, om, S, phi_large))
        
    # Normalize by large room norms
    large_norms = np.linalg.norm(V_large, axis=0, keepdims=True)
    V_small = V_small / large_norms
    V_medium = V_medium / large_norms
    V_large = V_large / large_norms
    
    # Calculate losses per neuron
    if parameters["sampling_choice"] == 1:
        phi_calc = np.random.normal(0, 1, [res*res, 2])
    elif parameters['sampling_choice'] == 0 or parameters['sampling_choice'] == 2:
        r = np.sqrt(np.random.uniform(size=[res*res, 1]))
        theta = np.random.uniform(size=[res*res, 1]) * 2 * np.pi
        phi_calc = np.hstack([r * np.cos(theta), r * np.sin(theta)])
    else:
        phi_calc = phi_medium
    
    phi_calc = np.expand_dims(phi_calc, axis=1)
    neur_losses = np.zeros(parameters["D"])
    sigma_sq = parameters.get("sigma_sq", 0.04)
    sigma_theta = parameters.get("sigma_theta", 0.5)
    f = parameters.get("f", 1)
    chi = helper_functions.calc_chi_plane(phi_calc, sigma_theta, f)
    
    # Overall loss
    overall_loss = losses.sep_plane_KernChi_seq(g0, om, S, phi_calc, sigma_sq, chi)
    
    Vs = [V_small, V_medium, V_large]
    
    # Plot neurons
    RowsD = int(np.ceil(np.sqrt(parameters["D"])))
    ColumnsD = int(np.ceil(parameters["D"]/RowsD))
    
    for (plot_counter, V_plot) in enumerate(Vs):
        fig = plt.figure(figsize=(20, 16))
        for neuron in range(parameters["D"]):
            plt.subplot(RowsD, ColumnsD, neuron + 1)
            plt.axis('off')
            plt.imshow(np.reshape(V_plot[neuron, :], [res, res]), vmin=V_plot.min(), vmax=V_plot.max())
            plt.colorbar()
            # plt.title(f"{neur_losses[neuron]:.3f}")
        fig.tight_layout()
        plt.suptitle(f'Overall Loss: {overall_loss:.5f}', y=1.0)
        figures[f'neurons_plot_{plot_counter + 1}'] = fig
        fig.savefig(os.path.join(savepath, f"Neurons_Plot_{counter}_{plot_counter+1}.png"))
    
    fig_score, _ = quantitative_analysis_seq(g0, S, om, parameters, counter, savepath=savepath)
    figures['grid_scores'] = fig_score
    
    return figures

def generate_loss_plots(L, min_L, lambda_pos=None, lambda_norm=None, val_L=None):
    """Generate loss evolution plots.

    Args:
        L: Loss array of shape [4, n_iters] containing [total, separation, positivity, norm]
        min_L: Minimum loss information
        lambda_pos: Optional array of lambda_pos values over iterations
        lambda_norm: Optional array of lambda_norm values over iterations
        val_L: Optional validation loss array of shape [4, n_val_iters], can be more sparsely sampled
    """
    titles = ['Loss', 'Separation', 'Positivity', 'Norm']
    fig = plt.figure(figsize=(20, 8))

    # Calculate x-axis scaling for validation losses if provided
    if val_L is not None:
        n_train = L.shape[1]
        n_val = val_L.shape[1]
        # Scale validation x-coordinates to align with training iterations
        val_x = np.linspace(0, n_train - 1, n_val)

    for counter in range(4):
        ax1 = plt.subplot(1, 4, counter + 1)
        color1 = 'tab:blue'
        ax1.plot(L[counter, :], color=color1, label='Train')

        # Plot validation loss if provided
        if val_L is not None:
            ax1.plot(val_x, val_L[counter, :], color='tab:green', linestyle='--',
                    alpha=0.8, label='Validation')

        ax1.set_xlabel('Iteration (x save_iters)')
        ax1.set_ylabel(titles[counter], color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_title(titles[counter])

        # Add legend if validation losses are shown
        if val_L is not None and counter == 0:
            ax1.legend(loc='upper right')

        # Add second y-axis for lambda values on Positivity (counter=2) and Norm (counter=3)
        if counter == 2 and lambda_pos is not None:
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.plot(lambda_pos, color=color2, alpha=0.7, linestyle='--')
            ax2.set_ylabel('lambda_pos', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            # Ensure lambda plot is visually above by setting higher zorder
            ax2.set_zorder(ax1.get_zorder() + 1)
            ax1.set_frame_on(False)

        elif counter == 3 and lambda_norm is not None:
            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.plot(lambda_norm, color=color2, alpha=0.7, linestyle='--')
            ax2.set_ylabel('lambda_norm', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            # Ensure lambda plot is visually above by setting higher zorder
            ax2.set_zorder(ax1.get_zorder() + 1)
            ax1.set_frame_on(False)

    plt.suptitle(f'Min Loss: {min_L[1]:.3f}')
    plt.tight_layout()

    return fig

def compute_validation_losses(
    run_path: str,
    validation_path: str = "dataset/validation",
    use_best_checkpoints: bool = False
) -> Dict[str, Any]:
    """
    Compute validation losses for checkpoints saved during training.

    Args:
        run_path: Path to the run directory containing checkpoints/ and parameters.json
        validation_path: Path to directory containing validation datasets
        use_best_checkpoints: If True, use g0_best_step_*.pkl etc., otherwise use g0_step_*.pkl

    Returns:
        Dictionary with structure:
        {
            'steps': [200, 400, ...],
            'per_k': {
                0: {
                    'per_dataset': {'dataset_name': np.array([4, n_steps]), ...},
                    'mean_across_datasets': np.array([4, n_steps]),
                    'std_across_datasets': np.array([4, n_steps]),
                },
                ...
            },
            'mean_across_k': {
                'per_dataset': {'dataset_name': {'mean': ..., 'std': ...}, ...},
                'mean_across_datasets': np.array([4, n_steps]),
                'std_across_datasets': np.array([4, n_steps]),
            },
            'dataset_metadata': {'dataset_name': {...}, ...},
            'summary': {
                'final_mean_loss': float,
                'best_step': int,
                'best_mean_loss': float
            }
        }
    """
    # Load parameters
    with open(os.path.join(run_path, 'parameters.json'), 'r') as f:
        parameters = json.load(f)

    sigma_sq = parameters.get("sigma_sq", 0.04)
    sigma_theta = parameters.get("sigma_theta", 0.5)
    f_param = parameters.get("f", 1)
    N_shift = parameters.get("N_shift", 15)
    Shift_std = parameters.get("Shift_std", 3)

    # Discover k directories
    checkpoints_path = os.path.join(run_path, "checkpoints")
    k_dirs = sorted([d for d in os.listdir(checkpoints_path) if d.startswith('k')])
    k_indices = [int(d[1:]) for d in k_dirs]

    # Discover checkpoint steps from first k directory
    first_k_path = os.path.join(checkpoints_path, k_dirs[0])
    prefix = "g0_best_step_" if use_best_checkpoints else "g0_step_"
    step_files = [f for f in os.listdir(first_k_path) if f.startswith(prefix) and f.endswith('.pkl')]
    steps = sorted([int(f.replace(prefix, '').replace('.pkl', '')) for f in step_files])

    # Discover validation datasets
    val_datasets = sorted([d for d in os.listdir(validation_path)
                          if os.path.isdir(os.path.join(validation_path, d))])

    # Load validation dataset metadata
    dataset_metadata = {}
    for ds_name in val_datasets:
        with open(os.path.join(validation_path, ds_name, 'metadata.json'), 'r') as f:
            dataset_metadata[ds_name] = json.load(f)

    # Note: We'll load validation batches on-demand to avoid memory issues
    print(f"Found {len(val_datasets)} validation datasets")

    # JIT compile loss functions for GPU acceleration
    sep_loss_jit = jax.jit(losses.sep_plane_KernChi_seq)
    pos_loss_jit = jax.jit(losses.pos_plane_seq)
    norm_loss_jit = jax.jit(losses.norm_plane_seq)

    # Initialize results structure
    n_steps = len(steps)
    per_k = {}

    print(f"Computing validation losses for {len(k_indices)} k values, {n_steps} steps, {len(val_datasets)} datasets...")

    for k_idx in k_indices:
        k_path = os.path.join(checkpoints_path, f"k{k_idx}")
        per_k[k_idx] = {'per_dataset': {}}

        for ds_name in val_datasets:
            per_k[k_idx]['per_dataset'][ds_name] = np.zeros((4, n_steps))

        for step_idx, step in enumerate(steps):
            print(f"    Step {step} ({step_idx + 1}/{n_steps})...", end=" ", flush=True)

            # Load checkpoint
            prefix = "best_" if use_best_checkpoints else ""
            with open(os.path.join(k_path, f'g0_{prefix}step_{step}.pkl'), 'rb') as f:
                g0 = pickle.load(f)
            with open(os.path.join(k_path, f'om_{prefix}step_{step}.pkl'), 'rb') as f:
                om = pickle.load(f)
            with open(os.path.join(k_path, f'S_{prefix}step_{step}.pkl'), 'rb') as f:
                S = pickle.load(f)

            # Convert to JAX arrays for GPU computation
            g0 = jnp.array(g0)
            om = jnp.array(om)
            S = jnp.array(S)

            # Compute loss for each validation dataset (process batch-by-batch)
            for ds_name in val_datasets:
                ds_path = os.path.join(validation_path, ds_name)
                metadata = dataset_metadata[ds_name]
                num_batches = metadata['num_batches']

                # Accumulate losses over batches
                sep_losses, pos_losses, norm_losses = [], [], []

                for batch_idx in range(num_batches):
                    with open(os.path.join(ds_path, f'batch_{batch_idx:05d}.pkl'), 'rb') as f_batch:
                        phi = pickle.load(f_batch)  # [batch_size, seq_len, 2]

                    B, L = phi.shape[0], phi.shape[1]

                    # Convert to JAX arrays for GPU computation
                    phi_jax = jnp.array(phi)
                    chi = helper_functions.calc_chi_plane(phi_jax, sigma_theta, f_param)

                    # Compute individual loss components using JIT-compiled functions
                    sep_losses.append(float(sep_loss_jit(g0, om, S, phi_jax, sigma_sq, chi)))
                    pos_losses.append(float(pos_loss_jit(g0, om, S, phi_jax)))

                    # Compute norm loss: create shifted versions of trajectories
                    phi_shift = np.random.normal(0, Shift_std, [N_shift, 2])
                    phi_norm = jnp.array(np.reshape(
                        phi[:, None, :, :] + phi_shift[None, :, None, :],
                        [B * N_shift, L, 2]
                    ))
                    norm_losses.append(float(norm_loss_jit(g0, om, S, phi_jax, phi_norm)))

                # Average over batches
                sep_loss = np.mean(sep_losses)
                pos_loss = np.mean(pos_losses)
                norm_loss = np.mean(norm_losses)

                # Store: [total, separation, positivity, norm]
                per_k[k_idx]['per_dataset'][ds_name][0, step_idx] = sep_loss   # Total = separation
                per_k[k_idx]['per_dataset'][ds_name][1, step_idx] = sep_loss   # Separation
                per_k[k_idx]['per_dataset'][ds_name][2, step_idx] = pos_loss   # Positivity
                per_k[k_idx]['per_dataset'][ds_name][3, step_idx] = norm_loss  # Norm

            print("done", flush=True)

        # Compute mean/std across datasets for this k
        all_losses = np.stack([per_k[k_idx]['per_dataset'][ds] for ds in val_datasets], axis=0)
        per_k[k_idx]['mean_across_datasets'] = np.mean(all_losses, axis=0)
        per_k[k_idx]['std_across_datasets'] = np.std(all_losses, axis=0)

        print(f"  k={k_idx} done")

    # Aggregate across k values
    mean_across_k = {'per_dataset': {}}

    for ds_name in val_datasets:
        ds_losses = np.stack([per_k[k]['per_dataset'][ds_name] for k in k_indices], axis=0)
        mean_across_k['per_dataset'][ds_name] = {
            'mean': np.mean(ds_losses, axis=0),
            'std': np.std(ds_losses, axis=0)
        }

    # Mean across k and datasets
    all_k_dataset_losses = np.stack([per_k[k]['mean_across_datasets'] for k in k_indices], axis=0)
    mean_across_k['mean_across_datasets'] = np.mean(all_k_dataset_losses, axis=0)
    mean_across_k['std_across_datasets'] = np.std(all_k_dataset_losses, axis=0)

    # Summary statistics
    mean_total_losses = mean_across_k['mean_across_datasets'][0, :]  # Total loss across steps
    best_step_idx = np.argmin(mean_total_losses)
    summary = {
        'final_mean_loss': float(mean_total_losses[-1]),
        'best_step': steps[best_step_idx],
        'best_mean_loss': float(mean_total_losses[best_step_idx])
    }

    print(f"Done. Best step: {summary['best_step']} with loss {summary['best_mean_loss']:.6f}")

    return {
        'steps': steps,
        'per_k': per_k,
        'mean_across_k': mean_across_k,
        'dataset_metadata': dataset_metadata,
        'summary': summary
    }


def generate_analysis_plots(savepath: str, counter: int = 0) -> Dict[str, plt.Figure]:
    """
    Generate all analysis plots for a completed optimization run.

    Args:
        savepath: Path to the directory containing optimization results
        counter: Which iteration to analyze (default: 0)
        save_fig: Whether to save figures to disk (default: True)
        plot_scale: Scaling factor for position lengthscale in plots (default: 1)

    Returns:
        Dictionary mapping plot names to figure objects
    """

    figures = {}

    if counter == '':
        adding_string = ''
    else:
        adding_string = '_'

    # Load parameters from JSON
    with open(os.path.join(savepath, 'parameters.json'), 'r') as json_file:
        parameters = json.load(json_file)

    # Load optimization results
    with open(os.path.join(savepath, f'g0{adding_string}{counter}.pkl'), 'rb') as f:
        g0 = pickle.load(f)

    with open(os.path.join(savepath, f'om{adding_string}{counter}.pkl'), 'rb') as f:
        om = pickle.load(f)

    with open(os.path.join(savepath, f'S{adding_string}{counter}.pkl'), 'rb') as f:
        S = pickle.load(f)

    with open(os.path.join(savepath, f'min_L{adding_string}{counter}.pkl'), 'rb') as f:
        min_L = pickle.load(f)

    with open(os.path.join(savepath, f'L{adding_string}{counter}.pkl'), 'rb') as f:
        L = pickle.load(f)

    with open(os.path.join(savepath, f'g0_init{adding_string}{counter}.pkl'), 'rb') as f:
        g0_init = pickle.load(f)
    
    with open(os.path.join(savepath, f'S_init{adding_string}{counter}.pkl'), 'rb') as f:
        S_init = pickle.load(f)

    try:
        with open(os.path.join(savepath, f'g0_final{adding_string}{counter}.pkl'), 'rb') as pickle_file:
            g0_final = pickle.load(pickle_file)
    except:
        g0_final = g0

    try:
        with open(os.path.join(savepath, f'S_final{adding_string}{counter}.pkl'), 'rb') as pickle_file:
            S_final = pickle.load(pickle_file)
    except:
        S_final = S

    # Load lambda arrays if they exist
    try:
        with open(os.path.join(savepath, f'lambda_pos{adding_string}{counter}.pkl'), 'rb') as f:
            lambda_pos = pickle.load(f)
    except:
        lambda_pos = None

    try:
        with open(os.path.join(savepath, f'lambda_norm{adding_string}{counter}.pkl'), 'rb') as f:
            lambda_norm = pickle.load(f)
    except:
        lambda_norm = None

    # Use final weights if min_L indicates
    if min_L[0] < 1: # min_L[0] is counter
        S = S_final
        g0 = g0_final

    # Generate dimension-specific plots
    if parameters["dim"] == 1:
        pass
    elif parameters["dim"] == 2:
        figures.update(generate_2d_plots(g0, om, S, parameters, savepath, counter))

    # Generate loss plots (common to all dimensions)
    fig_loss = generate_loss_plots(L, min_L, lambda_pos, lambda_norm)
    figures['losses'] = fig_loss
    fig_loss.savefig(os.path.join(savepath, f"Losses_{counter}.png"))

    return figures

def generate_sweep_score_distributions(sweep_path: str, savepath: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Generate distribution plots of grid scores across all runs in a sweep.

    Creates boxplots and median/IQR plots showing the distribution of mean and max
    grid scores over all runs for each scale/angle combination.

    Args:
        sweep_path: Path to the sweep directory containing run subdirectories
        savepath: Path to save figures. If None, saves to sweep_path.

    Returns:
        Dictionary mapping plot names to figure objects
    """
    import math

    if savepath is None:
        savepath = sweep_path

    figures = {}

    run_dirs = sorted([
        d for d in os.listdir(sweep_path)
        if d.startswith('run') and os.path.isdir(os.path.join(sweep_path, d))
    ])

    if not run_dirs:
        print(f"No run directories found in {sweep_path}")
        return figures

    seq_lens = []

    # Get stored keys for grid score metrics
    try:
        with open(os.path.join(sweep_path, run_dirs[0], f"grid_scores_0.json")) as f:
            grid_scores = json.load(f)
            keys = list(grid_scores.keys())
    except FileNotFoundError:
        print(f"No grid_scores_0.json found in {os.path.join(sweep_path, run_dirs[0])}")
        return figures

    with open(os.path.join(sweep_path, run_dirs[0], f"parameters.json")) as f:
        run_parameters = json.load(f)

    excluded_keys = ["sm_60", "md_60", "lg_60", "sm_90", "md_90", "lg_90"]
    keys = [key for key in keys if key not in excluded_keys]
    scores = {key: list(np.zeros((len(run_dirs), run_parameters["K"]))) for key in keys}

    for dir_idx, run_dir in enumerate(run_dirs):
        run_savepath = os.path.join(sweep_path, run_dir)
        with open(os.path.join(run_savepath, 'parameters.json'), 'r') as f:
            run_params = json.load(f)

        seq_lens = seq_lens + [run_params["seq_len"]]

        for k in range(run_params["K"]):
            try:
                with open(os.path.join(run_savepath, f"grid_scores_{k}.json"), 'r') as f:
                    grid_scores = json.load(f)
                for key in scores:
                    scores[key][dir_idx][k] = grid_scores[key]
            except FileNotFoundError:
                print(f"Warning: grid_scores_{k}.json not found in {run_savepath}")

    # Parse key to create descriptive title
    size_map = {"sm": "Small", "md": "Medium", "lg": "Large"}

    def get_base_key(key):
        """Extract size and angle from key (e.g., 'sm_60' from 'sm_60_mean')"""
        parts = key.split("_")
        return f"{parts[0]}_{parts[1]}"

    def parse_base_key(base_key):
        """Create descriptive title from base key"""
        parts = base_key.split("_")
        size = size_map.get(parts[0], parts[0])
        angle = parts[1] + "\u00b0"
        return f"{size} ratemap, {angle} angle"

    # Group keys by base (size_angle)
    base_keys = sorted(set(get_base_key(key) for key in keys))

    for base_key in base_keys:
        mean_key = f"{base_key}_mean"
        max_key = f"{base_key}_max"

        if mean_key not in scores or max_key not in scores:
            continue

        base_title = parse_base_key(base_key)

        # Boxplot arrangement
        fig_box, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig_box.suptitle(f"Distribution of grid scores over runs\n{base_title}")

        for ax, key, subtitle in [(axes[0], mean_key, "Mean (over neuron scores)"),
                                   (axes[1], max_key, "Max (over neuron scores)")]:
            ax.boxplot(scores[key], positions=seq_lens, widths=[s * 0.4 for s in seq_lens])
            ax.set_xscale('log')
            ax.set_title(subtitle)
            ax.set_xlabel("Sequence length")
            ax.set_ylabel("Grid score")

        fig_box.tight_layout()
        fig_box.savefig(os.path.join(savepath, f"score_distribution_boxplot_{base_key}.png"))
        figures[f'boxplot_{base_key}'] = fig_box

        # Median with IQR arrangement
        fig_iqr, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig_iqr.suptitle(f"Distribution of grid scores over runs\n{base_title}")

        for ax, key, subtitle in [(axes[0], mean_key, "Mean (over neuron scores)"),
                                   (axes[1], max_key, "Max (over neuron scores)")]:
            medians = np.median(np.array(scores[key]), axis=1)
            q1 = np.quantile(np.array(scores[key]), 0.25, axis=1)
            q3 = np.quantile(np.array(scores[key]), 0.75, axis=1)

            ax.fill_between(seq_lens, q1, q3, alpha=.5, linewidth=0, label="IQR (Q1-Q3)")
            ax.plot(seq_lens, medians, linewidth=2, label="Median")
            ax.set_xscale('log')
            ax.set_title(subtitle)
            ax.set_xlabel("Sequence length")
            ax.set_ylabel("Grid score")
            ax.legend()

        fig_iqr.tight_layout()
        fig_iqr.savefig(os.path.join(savepath, f"score_distribution_iqr_{base_key}.png"))
        figures[f'iqr_{base_key}'] = fig_iqr

    print(f"Score distribution plots saved to {savepath}")
    return figures


def run_seq_parameter_sweep(
    base_parameters: Dict[str, Any],
    sweep_params: Dict[str, List[Any]] | List[Dict[str, Any]],
    base_savepath: Optional[str] = None,
    key_seed: int = 0,
    generate_plots: bool = True,
    g0_init: Optional[Any] = None,
    om_init: Optional[Any] = None,
    S_init: Optional[Any] = None,
    same_init_across_K: bool = False,
    use_dataset_variants: bool = False
) -> List[Dict[str, Any]]:
    """
    Run a parameter sweep over specified parameter combinations.

    Args:
        base_parameters: Base parameter dictionary (will be copied and modified for each run)
        sweep_params: Either:
                      - Dictionary mapping parameter names to lists of values (outer product mode)
                        Example: {'lambda_pos_init': [0.05, 0.1, 0.15], 'k_p': [-8, -9, -10]}
                        This creates all combinations (3 x 3 = 9 runs)
                      - List of dictionaries with specific parameter combinations (explicit mode)
                        Example: [{'lambda_pos_init': 0.05, 'k_p': -8},
                                  {'lambda_pos_init': 0.1, 'k_p': -9}]
                        This creates only the specified combinations (2 runs)
        base_savepath: Base directory for saving results (subdirs created for each run)
        key_seed: Starting random seed (incremented for each run)
        generate_plots: Whether to generate analysis plots for each run
        g0_init: Optional initial values for g0 parameter (if None, random initialization)
        om_init: Optional initial values for om parameter (if None, random initialization)
        S_init: Optional initial values for S parameter (if None, random initialization)
        same_init_across_K: If True, use same random initialization for all K runs within each parameter combination.
                           If False (default), each K run gets different random initialization.
        use_dataset_variants: If True, create K different dataset variants (one per K iteration).
                             Each variant has different random trajectories but same seq_len/batch parameters.
                             If False (default), all K runs use the same dataset.

    Returns:
        List of dictionaries containing results and metadata for each run
    """

    # Create all parameter combinations based on input type
    if isinstance(sweep_params, list):
        # Explicit mode: use specified combinations directly
        combinations = [tuple(combo.values()) for combo in sweep_params]
        param_names = list(sweep_params[0].keys()) if sweep_params else []
        sweep_mode = "explicit"
    else:
        # Outer product mode: create all combinations
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        combinations = list(itertools.product(*param_values))
        sweep_mode = "outer_product"

    print(f"\n{'='*80}")
    print(f"Starting parameter sweep with {len(combinations)} combinations")
    print(f"Sweep mode: {sweep_mode}")
    print(f"Sweeping over: {param_names}")
    print(f"{'='*80}\n")

    results = []

    if base_savepath is None:
        today = datetime.strftime(datetime.now(), '%y%m%d')
        now = datetime.strftime(datetime.now(), '%H%M%S')
        base_savepath = f"data/{today}/sweep_{now}/"

    for idx, combo in enumerate(combinations):
        print(f"\n{'-'*80}")
        print(f"Run {idx + 1}/{len(combinations)}")
        print(f"Parameters: {dict(zip(param_names, combo))}")
        print(f"{'-'*80}\n")    

        # Create parameter dict for this run
        run_parameters = base_parameters.copy()

        # Update with sweep parameters
        for param_name, param_value in zip(param_names, combo):
            run_parameters[param_name] = param_value

        run_savepath = os.path.join(base_savepath, f"run{idx:03d}/")

        # Ensure directory exists
        os.makedirs(run_savepath, exist_ok=True)

        # Generate dataset(s) and create dataloader(s)
        if use_dataset_variants:
            # Create K different dataset variants, one for each K iteration
            dataloaders = []
            for variant_idx in range(run_parameters["K"]):
                dataset = f"dataset/seq_len_{run_parameters['seq_len']}_batch_{run_parameters['batch']}_variant_{variant_idx}/"
                if not os.path.exists(dataset):
                    print(f"Generating dataset variant {variant_idx}...")
                    generator = TrajectoryGenerator()
                    generator.generate_dataset(dataset, 50000, batch_size=run_parameters["batch"], sequence_length=run_parameters["seq_len"])

                dataloader_variant = TrajectoryDataset(
                    dataset,
                    num_workers=6,
                    prefetch_batches=10
                )
                dataloaders.append(dataloader_variant)
            dataloader = dataloaders  # Pass list of dataloaders
        else:
            # Use single dataset for all K iterations
            dataset = f"dataset/seq_len_{run_parameters['seq_len']}_batch_{run_parameters['batch']}/"
            if not os.path.exists(dataset):
                generator = TrajectoryGenerator()
                generator.generate_dataset(dataset, 50000, batch_size=run_parameters["batch"], sequence_length=run_parameters["seq_len"])

            dataloader = TrajectoryDataset(
                dataset,
                num_workers=6,
                prefetch_batches=10
            )

        # Save sweep configuration
        sweep_config = {
            'run_index': idx,
            'total_runs': len(combinations),
            'sweep_parameters': dict(zip(param_names, combo)),
            'all_parameters': run_parameters
        }
        with open(os.path.join(run_savepath, 'sweep_config.json'), 'w') as f:
            json.dump(sweep_config, f, indent=2)

        # Run optimization
        try:
            opt_results = run_plane_sequential_optimization(
                parameters=run_parameters,
                dataloader=dataloader,
                savepath=run_savepath,
                key_seed=key_seed + idx,
                g0_init=g0_init,
                om_init=om_init,
                S_init=S_init,
                same_init_across_K=same_init_across_K
            )

            # Generate analysis plots
            if generate_plots:
                print(f"\nGenerating analysis plots...")
                for k in range(run_parameters["K"]):
                    try:
                        figures = generate_analysis_plots(run_savepath, counter=k)
                        plt.close('all')  # Close all figures to free memory
                        print(f"Analysis plots for {k} saved to {run_savepath}")
                    except Exception as e:
                        print(f"Warning: Failed to generate plots for run {idx} {k}: {e}")

            # Store results
            run_result = {
                'run_index': idx,
                'sweep_parameters': dict(zip(param_names, combo)),
                'savepath': run_savepath,
                'min_losses': opt_results['min_L_list'],
                'success': True
            }
            results.append(run_result)

            print(f"\n✓ Run {idx + 1} completed successfully")
            print(f"  Best loss: {opt_results['min_L_list'][0][1]:.5f}")

        except Exception as e:
            print(f"\n✗ Run {idx + 1} failed with error: {e}")
            run_result = {
                'run_index': idx,
                'sweep_parameters': dict(zip(param_names, combo)),
                'savepath': run_savepath,
                'success': False,
                'error': str(e)
            }
            results.append(run_result)

    # Save summary of all runs
    summary_path = os.path.join(os.path.dirname(run_savepath), 'sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Parameter sweep completed")
    print(f"Successful runs: {sum(r['success'] for r in results)}/{len(results)}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")

    # Generate score distribution plots across all runs
    if generate_plots:
        print("Generating sweep-level score distribution plots...")
        try:
            generate_sweep_score_distributions(base_savepath)
            plt.close('all')
        except Exception as e:
            print(f"Warning: Failed to generate sweep score distributions: {e}")

    return results

if __name__ == "__main__":
    # Example: Generate plots for an existing run
    # savepath = "data/251120/081340/"
    # figures = generate_analysis_plots(savepath, counter=0, save_fig=True)
    # plt.show()

    # Example: Run a parameter sweep
    base_params = {
        "D": 65, "T": 50000, "K": 1, "N_rand": 150, "phi_std": 1,
        "N_shift": 15, "resample_iters": 5, "save_iters": 5, "print_iters": 1000,
        "lambda_pos_init": 0.1, "k_p": -9, "alpha_p": 0.9, "gamma_p": 0.0001,
        "lambda_norm_init": 0.005, "k_norm": 4, "alpha_norm": 0.9,
        "gamma_norm": 0.0001, "beta1": 0.9, "beta2": 0.9, "eta": 1e-8,
        "epsilon_g0": 0.1, "epsilon_om": 0.1, "epsilon_s": 0.1, "dim": 2,
        "sampling_choice": 1, "norm_size": 1,
        "Shift_std": 3, "shift_points_sep": 0,
        "om_init_scheme": 0, "om_init_scale": 2,
        "sigma_sq": 0.04, "sigma_theta": 0.5, "f": 1
    }

    sweep_params = {
        "lambda_pos_init": [0.05, 0.1, 0.15],
        "k_p": [-8, -9, -10]
    }
