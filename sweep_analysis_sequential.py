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

def compute_g_at_positions(g0, om, S1, S2, phi):
    """
    Compute g(phi) = T(phi) @ g0 where T(phi) = S @ T_irrep(phi) @ S^(-1)
    
    Args:
        g0: activity at origin, shape [D, 1]
        om: frequencies, shape [M, 2]
        S: change of basis matrix, shape [D, D]
        phi: positions, shape [N, 2]
    
    Returns:
        g: activity at positions, shape [D, N]
    """
    # Apply softplus and normalize (same as in loss functions)
    g0_processed = jax.nn.softplus(g0)
    g0_processed = g0_processed / jnp.linalg.norm(g0_processed)
    
    # Get transformation matrices
    T = helper_functions.get_T_2D(om, phi, S1, S2)
    
    # Apply transformation
    g = jnp.einsum('nij,j->in', T, g0_processed)
    
    return g

def _generate_2d_plots(g0, om, S1, S2, parameters, savepath, counter, plot_scale):
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

    im1 = axes[0].imshow(S1, cmap='RdBu_r', aspect='equal')
    axes[0].set_title('S_1')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0])

    im1 = axes[1].imshow(S2, cmap='RdBu_r', aspect='equal')
    axes[1].set_title('S_2')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[1])

    im1 = axes[2].imshow(S1 @ S2, cmap='RdBu_r', aspect='equal')
    axes[2].set_title('S_1 @ S_2')
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[2])
    fig_S.tight_layout()

    figures['S'] = fig_S
    fig_S.savefig(os.path.join(savepath, f"S_analysis_{counter}.png"))


    # Create grid of positions
    N_plot = 70
    phi_plot_small = np.linspace(-np.pi, np.pi, N_plot)/2
    phi_plot_small = np.meshgrid(phi_plot_small, phi_plot_small)
    phi_plot_small = np.hstack([np.ndarray.flatten(phi_plot_small[0])[:,None], 
                                 np.ndarray.flatten(phi_plot_small[1])[:,None]])
    
    phi_plot_large = np.linspace(-np.pi, np.pi, N_plot)*parameters.get("pos_lengthscale", 2)*plot_scale
    phi_plot_large = np.meshgrid(phi_plot_large, phi_plot_large)
    phi_plot_large = np.hstack([np.ndarray.flatten(phi_plot_large[0])[:,None], 
                                 np.ndarray.flatten(phi_plot_large[1])[:,None]])
    
    # Compute activities using transformation
    V_small = np.array(compute_g_at_positions(g0, om, S1, S2, phi_plot_small))
    V_large = np.array(compute_g_at_positions(g0, om, S1, S2, phi_plot_large))
    
    # Normalize by large room norms
    large_norms = np.linalg.norm(V_large, axis=0, keepdims=True)
    V_small = V_small / large_norms
    V_large = V_large / large_norms
    
    # Calculate losses per neuron
    if parameters["sampling_choice"] == 1:
        phi_calc = np.random.normal(0, 1, [N_plot*N_plot, 2])
    elif parameters['sampling_choice'] == 0 or parameters['sampling_choice'] == 2:
        r = np.sqrt(np.random.uniform(size=[N_plot*N_plot, 1]))
        theta = np.random.uniform(size=[N_plot*N_plot, 1]) * 2 * np.pi
        phi_calc = np.hstack([r * np.cos(theta), r * np.sin(theta)])
    else:
        phi_calc = phi_plot_small
    
    neur_losses = np.zeros(parameters["D"])
    sigma_sq = parameters.get("sigma_sq", 0.04)
    sigma_theta = parameters.get("sigma_theta", 0.5)
    f = parameters.get("f", 1)
    chi = helper_functions.calc_chi_plane(phi_calc, sigma_theta, f)
    
    # Calculate loss for each neuron
    for neuron in range(parameters["D"]):
        g0_neuron = g0[neuron:neuron+1]
        neur_losses[neuron] = losses.sep_plane_KernChi_seq(g0_neuron, om, S1, S2, phi_calc, sigma_sq, chi)
    
    # Overall loss
    overall_loss = losses.sep_plane_KernChi_seq(g0, om, S1, S2, phi_calc, sigma_sq, chi)
    
    Vs = [V_small, V_large]
    phi_plots = [phi_plot_small, phi_plot_large]
    
    # Plot neurons
    RowsD = int(np.ceil(np.sqrt(parameters["D"])))
    ColumnsD = int(np.ceil(parameters["D"]/RowsD))
    
    for (plot_counter, V_plot) in enumerate(Vs):
        fig = plt.figure(figsize=(20, 16))
        for neuron in range(parameters["D"]):
            plt.subplot(RowsD, ColumnsD, neuron + 1)
            plt.axis('off')
            plt.imshow(np.reshape(V_plot[neuron, :], [N_plot, N_plot]), vmin=V_plot.min(), vmax=V_plot.max())
            plt.colorbar()
            plt.title(f"{neur_losses[neuron]:.3f}")
        fig.tight_layout()
        plt.suptitle(f'Overall Loss: {overall_loss:.5f}', y=1.0)
        figures[f'neurons_plot_{plot_counter + 1}'] = fig
        fig.savefig(os.path.join(savepath, f"Neurons_Plot_{counter}_{plot_counter+1}.png"))
    
    return figures

def _generate_loss_plots(L, min_L):
    """Generate loss evolution plots."""
    titles = ['Loss', 'Separation', 'Positivity', 'Norm']
    fig = plt.figure(figsize=(12, 8))

    for counter in range(4):
        plt.subplot(1, 4, counter + 1)
        plt.plot(L[counter, :])
        plt.title(titles[counter])

    plt.suptitle(f'Min Loss: {min_L[1]:.3f}')
    plt.tight_layout()

    return fig

def generate_analysis_plots(savepath: str, counter: int = 0, plot_scale: float = 1) -> Dict[str, plt.Figure]:
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

    with open(os.path.join(savepath, f'S1{adding_string}{counter}.pkl'), 'rb') as f:
        S1 = pickle.load(f)
    
    with open(os.path.join(savepath, f'S2{adding_string}{counter}.pkl'), 'rb') as f:
        S2 = pickle.load(f)

    with open(os.path.join(savepath, f'min_L{adding_string}{counter}.pkl'), 'rb') as f:
        min_L = pickle.load(f)

    with open(os.path.join(savepath, f'L{adding_string}{counter}.pkl'), 'rb') as f:
        L = pickle.load(f)

    with open(os.path.join(savepath, f'g0_init{adding_string}{counter}.pkl'), 'rb') as f:
        g0_init = pickle.load(f)
    
    with open(os.path.join(savepath, f'S1_init{adding_string}{counter}.pkl'), 'rb') as f:
        S1_init = pickle.load(f)
    
    with open(os.path.join(savepath, f'S2_init{adding_string}{counter}.pkl'), 'rb') as f:
        S2_init = pickle.load(f)

    try:
        with open(os.path.join(savepath, f'g0_final{adding_string}{counter}.pkl'), 'rb') as pickle_file:
            g0_final = pickle.load(pickle_file)
    except:
        g0_final = g0

    try:
        with open(os.path.join(savepath, f'S1_final{adding_string}{counter}.pkl'), 'rb') as pickle_file:
            S1_final = pickle.load(pickle_file)
        with open(os.path.join(savepath, f'S2_final{adding_string}{counter}.pkl'), 'rb') as pickle_file:
            S2_final = pickle.load(pickle_file)
    except:
        S1_final = S1
        S2_final = S2

    # Use final weights if min_L indicates
    if min_L[0] < 1: # min_L[0] is counter
        S1 = S1_final
        S2 = S2_final
        g0 = g0_final

    # Generate dimension-specific plots
    if parameters["dim"] == 1:
        pass
    elif parameters["dim"] == 2:
        figures.update(_generate_2d_plots(g0, om, S1, S2, parameters, savepath, counter, plot_scale))

    # Generate loss plots (common to all dimensions)
    fig_loss = _generate_loss_plots(L, min_L)
    figures['losses'] = fig_loss
    fig_loss.savefig(os.path.join(savepath, f"Losses_{counter}.png"))

    return figures

def run_seq_parameter_sweep(
    base_parameters: Dict[str, Any],
    sweep_params: Dict[str, List[Any]],
    base_savepath: Optional[str] = None,
    key_seed: int = 0,
    generate_plots: bool = True
) -> List[Dict[str, Any]]:
    """
    Run a parameter sweep over specified parameter combinations.

    Args:
        base_parameters: Base parameter dictionary (will be copied and modified for each run)
        sweep_params: Dictionary mapping parameter names to lists of values to sweep over
                      Example: {'lambda_pos_init': [0.05, 0.1, 0.15], 'k_p': [-8, -9, -10]}
        om_init_scheme: Frequency initialization scheme
        sep_loss_choice: Separation loss choice
        chi_choice: Chi function choice
        W_constrain: Whether to constrain W matrix
        base_savepath: Base directory for saving results (subdirs created for each run)
        key_seed: Starting random seed (incremented for each run)
        generate_plots: Whether to generate analysis plots for each run

    Returns:
        List of dictionaries containing results and metadata for each run
    """

    # Create all parameter combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    combinations = list(itertools.product(*param_values))

    print(f"\n{'='*80}")
    print(f"Starting parameter sweep with {len(combinations)} combinations")
    print(f"Sweeping over: {param_names}")
    print(f"{'='*80}\n")

    results = []

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

        # Create unique savepath for this run
        if base_savepath is None:
            today = datetime.strftime(datetime.now(), '%y%m%d')
            now = datetime.strftime(datetime.now(), '%H%M%S')
            run_savepath = f"data/{today}/sweep_{now}/run{idx:03d}/"
        else:
            run_savepath = os.path.join(base_savepath, f"run{idx:03d}/")

        # Ensure directory exists
        os.makedirs(run_savepath, exist_ok=True)

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
                savepath=run_savepath,
                key_seed=key_seed + idx
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
