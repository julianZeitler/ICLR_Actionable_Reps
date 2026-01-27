import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
from datetime import datetime
import jax.numpy as jnp
from typing import Dict, List, Optional, Any
import itertools

from nrt import losses
from nrt import helpers
from sequential.plane import run_plane_sequential_optimization
from nrt.analysis import quantitative_analysis_seq
from nrt.data import TrajectoryDataset, TrajectoryGenerator
from nrt.plotting import loss_plots, neuron_plotter_2d

def generate_2d_plots(g0, om, S, savepath, counter):
    figures = {}

    ### Frequencies
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

    ### S analysis
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

    ### Grid Scores
    res = 70
    widths = (1, 2, 4)
    V_small, V_medium, V_large = helpers.get_ratemaps_seq(g0, om, S, res, widths)
    Vs = [V_small, V_medium, V_large]

    fig_score, scores = quantitative_analysis_seq(Vs, widths, counter, res=res, savepath=savepath)
    figures['grid_scores'] = fig_score

    ### Ratemap plots        
    # Normalize by large room norms
    large_norms = np.linalg.norm(V_large, axis=0, keepdims=True)
    V_small = V_small / large_norms
    V_medium = V_medium / large_norms
    V_large = V_large / large_norms
    
    neuron_sm_fig = neuron_plotter_2d(V_small, res, scores["sm_60"])
    neuron_md_fig = neuron_plotter_2d(V_medium, res, scores["md_60"])
    neuron_lg_fig = neuron_plotter_2d(V_large, res, scores["lg_60"])
    neuron_figs = [neuron_sm_fig, neuron_md_fig, neuron_lg_fig]

    for plot_counter, fig in enumerate(neuron_figs):
        figures[f'neurons_plot_{plot_counter + 1}'] = fig
        fig.savefig(os.path.join(savepath, f"Neurons_Plot_{counter}_{plot_counter+1}.png"))
    
    return figures

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
        figures.update(generate_2d_plots(g0, om, S, savepath, counter))

    # Generate loss plots (common to all dimensions)
    fig_loss = loss_plots(L, min_L, lambda_pos, lambda_norm)
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


def generate_sweep_validation_loss_distributions(
    sweep_path: str,
    savepath: Optional[str] = None,
    use_final: bool = True,
    loss_component: int = 0
) -> Dict[str, Any]:
    """
    Generate distribution plots comparing validation losses across environments.

    Creates boxplots comparing random vs snake environments side by side,
    with separate figures for different environment sizes.

    Args:
        sweep_path: Path to the sweep directory containing run subdirectories
        savepath: Path to save figures. If None, saves to sweep_path.
        use_final: If True, use final step losses. If False, use best (minimum) losses.
        loss_component: Which loss component to plot (0=total, 1=separation, 2=positivity, 3=norm)

    Returns:
        Dictionary mapping plot names to figure objects
    """
    if savepath is None:
        savepath = sweep_path

    figures = {}
    loss_names = ["Total", "Separation", "Positivity", "Norm"]
    loss_name = loss_names[loss_component]

    run_dirs = sorted([
        d for d in os.listdir(sweep_path)
        if d.startswith('run') and os.path.isdir(os.path.join(sweep_path, d))
    ])

    if not run_dirs:
        print(f"No run directories found in {sweep_path}")
        return figures

    # Get dataset keys from first run's validation_losses.json
    try:
        with open(os.path.join(sweep_path, run_dirs[0], "validation_losses.json")) as f:
            val_losses = json.load(f)
            first_k = list(val_losses['per_k'].keys())[0]
            dataset_keys = list(val_losses['per_k'][first_k]['per_dataset'].keys())
    except FileNotFoundError:
        print(f"No validation_losses.json found in {os.path.join(sweep_path, run_dirs[0])}")
        return figures

    # Parse dataset keys to extract environment types and sizes
    # Expected format: "random_box2x2", "snake_grid10_box5x5", etc.
    sizes = set()
    env_types = set()
    for key in dataset_keys:
        if "box" in key:
            # Extract size (e.g., "2x2" from "random_box2x2")
            size = key.split("box")[-1]
            sizes.add(size)
            # Extract environment type (e.g., "random" or "snake_grid10")
            env_type = key.split("_box")[0]
            env_types.add(env_type)

    sizes = sorted(sizes)
    env_types = sorted(env_types)

    # Collect validation losses for each dataset across all runs
    # Structure: {dataset_key: [[losses for k=0, k=1, ...] for each run]}
    all_losses = {key: [] for key in dataset_keys}

    seq_lens = []
    for run_dir in run_dirs:
        run_path = os.path.join(sweep_path, run_dir)

        # Load parameters
        with open(os.path.join(run_path, 'parameters.json'), 'r') as f:
            run_params = json.load(f)
        seq_lens.append(run_params.get("seq_len", 0))

        # Load validation losses
        val_losses_path = os.path.join(run_path, 'validation_losses.json')
        if not os.path.exists(val_losses_path):
            # Append NaN for missing data
            for key in dataset_keys:
                all_losses[key].append([np.nan] * run_params["K"])
            continue

        with open(val_losses_path, 'r') as f:
            val_losses = json.load(f)

        run_losses = {key: [] for key in dataset_keys}
        for k in range(run_params["K"]):
            k_str = str(k)
            if k_str not in val_losses['per_k']:
                for key in dataset_keys:
                    run_losses[key].append(np.nan)
                continue

            for key in dataset_keys:
                if key not in val_losses['per_k'][k_str]['per_dataset']:
                    run_losses[key].append(np.nan)
                    continue

                losses = np.array(val_losses['per_k'][k_str]['per_dataset'][key])
                # losses has shape [4, n_steps]
                if use_final:
                    loss_val = losses[loss_component, -1]
                else:
                    loss_val = np.min(losses[loss_component])
                run_losses[key].append(loss_val)

        for key in dataset_keys:
            all_losses[key].append(run_losses[key])

    # Convert to arrays: shape [n_runs, K]
    for key in all_losses:
        all_losses[key] = np.array(all_losses[key])

    # Create figures for each size, comparing environment types
    unique_seq_lens = sorted(set(seq_lens))

    for size in sizes:
        # Find dataset keys for this size
        size_keys = [k for k in dataset_keys if f"box{size}" in k]
        if len(size_keys) < 1:
            continue

        # Separate by environment type
        env_data = {}
        for key in size_keys:
            env_type = key.split("_box")[0]
            env_data[env_type] = all_losses[key]

        loss_type_str = "Final" if use_final else "Best"

        # Organize data by seq_len for each environment type
        # Structure: {env_type: {seq_len: [losses]}}
        env_seq_losses = {}
        for env_type, losses in env_data.items():
            env_seq_losses[env_type] = {sl: [] for sl in unique_seq_lens}
            for idx, sl in enumerate(seq_lens):
                # losses[idx] has shape [K], flatten and filter NaN
                run_losses = losses[idx].flatten().tolist()
                env_seq_losses[env_type][sl].extend([l for l in run_losses if not np.isnan(l)])

        # Create boxplot figure with subplots for each env type
        n_envs = len(env_seq_losses)
        fig_box, axes_box = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5), squeeze=False)
        axes_box = axes_box.flatten()

        fig_box.suptitle(f"{loss_type_str} Validation {loss_name} Loss Distribution\nEnvironment size: {size}")

        for ax, (env_type, seq_losses) in zip(axes_box, sorted(env_seq_losses.items())):
            # Prepare data for boxplot: list of loss arrays, one per seq_len
            box_data = [seq_losses[sl] for sl in unique_seq_lens]

            # Only plot if we have data
            if any(len(d) > 0 for d in box_data):
                ax.boxplot(box_data, positions=unique_seq_lens, widths=[s * 0.4 for s in unique_seq_lens])
                ax.set_xscale('log')
                ax.set_xlabel("Sequence length")
                ax.set_ylabel(f"{loss_name} Loss")

            # Format environment type for display
            display_name = env_type.replace("_", " ").title()
            ax.set_title(display_name)

        fig_box.tight_layout()
        fig_box.savefig(os.path.join(savepath, f"val_loss_dist_boxplot_{size}_{loss_name.lower()}.png"))
        figures[f'boxplot_{size}_{loss_name.lower()}'] = fig_box

        # Create IQR plot over seq_len
        if len(unique_seq_lens) > 1:
            fig_iqr, axes_iqr = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5), squeeze=False)
            axes_iqr = axes_iqr.flatten()

            fig_iqr.suptitle(f"{loss_type_str} Validation {loss_name} Loss vs Sequence Length\nEnvironment size: {size}")

            for ax, (env_type, seq_losses) in zip(axes_iqr, sorted(env_seq_losses.items())):
                medians = []
                q1s = []
                q3s = []
                valid_seq_lens = []

                for sl in unique_seq_lens:
                    sl_losses = seq_losses[sl]
                    if sl_losses:
                        medians.append(np.median(sl_losses))
                        q1s.append(np.quantile(sl_losses, 0.25))
                        q3s.append(np.quantile(sl_losses, 0.75))
                        valid_seq_lens.append(sl)

                if valid_seq_lens:
                    ax.fill_between(valid_seq_lens, q1s, q3s, alpha=0.5, linewidth=0, label="IQR (Q1-Q3)")
                    ax.plot(valid_seq_lens, medians, linewidth=2, marker='o', label="Median")
                    ax.set_xscale('log')
                    ax.set_xlabel("Sequence Length")
                    ax.set_ylabel(f"{loss_name} Loss")
                    ax.legend()

                display_name = env_type.replace("_", " ").title()
                ax.set_title(display_name)

            fig_iqr.tight_layout()
            fig_iqr.savefig(os.path.join(savepath, f"val_loss_iqr_{size}_{loss_name.lower()}.png"))
            figures[f'iqr_{size}_{loss_name.lower()}'] = fig_iqr

    print(f"Validation loss distribution plots saved to {savepath}")
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
