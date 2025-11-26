"""
Functions for running parameter sweeps and generating analysis plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
from datetime import datetime
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any
import itertools

from NRT_functions import losses
from NRT_functions import helper_functions
from NRT_functions import plotters
from Plane import run_plane_optimization


def generate_analysis_plots(savepath: str, counter: int = 0, save_fig: bool = True, plot_scale: float = 1) -> Dict[str, plt.Figure]:
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
    with open(os.path.join(savepath, f'W{adding_string}{counter}.pkl'), 'rb') as f:
        W = pickle.load(f)

    with open(os.path.join(savepath, f'om{adding_string}{counter}.pkl'), 'rb') as f:
        om = pickle.load(f)

    with open(os.path.join(savepath, f'min_L{adding_string}{counter}.pkl'), 'rb') as f:
        min_L = pickle.load(f)

    with open(os.path.join(savepath, f'L{adding_string}{counter}.pkl'), 'rb') as f:
        L = pickle.load(f)

    with open(os.path.join(savepath, f'W_init{adding_string}{counter}.pkl'), 'rb') as f:
        W_init = pickle.load(f)

    W_final = W

    # Add default parameters if missing
    if "pos_lengthscale" not in parameters:
        parameters["pos_lengthscale"] = 2
    if "dim" not in parameters:
        parameters["dim"] = 1

    # Use final weights if min_L indicates
    if min_L[0] < 1:
        W = W_final

    # Generate dimension-specific plots
    if parameters["dim"] == 1:
        figures.update(_generate_1d_plots(W, W_init, om, L, min_L, parameters, savepath, save_fig))
    elif parameters["dim"] == 2:
        figures.update(_generate_2d_plots(W, W_init, om, L, min_L, parameters, savepath, save_fig, plot_scale))

    # Generate loss plots (common to all dimensions)
    fig_loss = _generate_loss_plots(L, min_L)
    figures['losses'] = fig_loss
    if save_fig:
        fig_loss.savefig(os.path.join(savepath, "Losses.png"))

    return figures


def _generate_2d_plots(W, W_init, om, L, min_L, parameters, savepath, save_fig, plot_scale):
    """Generate 2D-specific analysis plots."""
    figures = {}

    # Frequency plot
    fig_freq = plotters.freq_plot(W, om, 1.1, 1, parameters)
    figures['freq_plot'] = fig_freq
    if save_fig:
        fig_freq.savefig(os.path.join(savepath, "freq_plot.png"))

    # Neural response plots
    if "epsilon_w" in parameters:
        N_plot = 70

        # Small plot (training scale)
        phi_plot_small = np.linspace(-np.pi, np.pi, N_plot) / 2
        phi_plot_small = np.meshgrid(phi_plot_small, phi_plot_small)
        phi_plot_small = np.hstack([
            np.ndarray.flatten(phi_plot_small[0])[:, None],
            np.ndarray.flatten(phi_plot_small[1])[:, None]
        ])

        # Large plot (generalization scale)
        phi_plot_large = np.linspace(-np.pi, np.pi, N_plot) * parameters["pos_lengthscale"] * plot_scale
        phi_plot_large = np.meshgrid(phi_plot_large, phi_plot_large)
        phi_plot_large = np.hstack([
            np.ndarray.flatten(phi_plot_large[0])[:, None],
            np.ndarray.flatten(phi_plot_large[1])[:, None]
        ])

        # Compute neural responses
        I_large = helper_functions.init_irreps_2D(om, phi_plot_large)
        I_small = helper_functions.init_irreps_2D(om, phi_plot_small)
        V_large = np.matmul(W, I_large)
        V_small = np.matmul(W, I_small)

        # Normalize by large norms
        large_norms = np.linalg.norm(V_large, axis=1)
        V_small = V_small / large_norms[:, None]
        V_large = V_large / large_norms[:, None]

        N = V_large.shape[0]

        # Determine phi for loss calculation
        if parameters["sampling_choice"] == 1:
            phi_calc = np.random.normal(0, 1, [N_plot * N_plot, 2])
        elif parameters['sampling_choice'] == 0 or parameters['sampling_choice'] == 2:
            r = np.sqrt(np.random.uniform(size=[N_plot * N_plot, 1]))
            theta = np.random.uniform(size=[N_plot * N_plot, 1]) * 2 * np.pi
            phi_calc = np.hstack([r * np.cos(theta), r * np.sin(theta)])
        else:
            phi_calc = phi_plot_small

        # Calculate per-neuron losses
        neur_losses = np.zeros(N)
        if parameters["sep_loss_choice"] == 3:
            chi = helper_functions.calc_chi_plane(phi_calc, parameters["sigma_theta"], parameters["f"])
            # Note: Calculating per-neuron losses can be expensive, skip if not needed
            # for neuron in range(N):
            #     neur_losses[neuron] = losses.sep_plane_KernChi(
            #         W[neuron, :][None, :], om, phi_calc, parameters["sigma_sq"], chi
            #     )

        # Generate neuron response plots
        Vs = [V_small, V_large]
        phi_plots = [phi_plot_small, phi_plot_large]

        RowsD = int(np.ceil(np.sqrt(parameters["D"])))
        ColumnsD = int(np.ceil(parameters["D"] / RowsD))

        for plot_counter, V_plot in enumerate(Vs):
            fig = plt.figure(figsize=(20, 16))
            for neuron in range(parameters["D"]):
                plt.subplot(RowsD, ColumnsD, neuron + 1)
                plt.axis('off')
                plt.imshow(
                    np.reshape(V_plot[neuron, :], [N_plot, N_plot]),
                    vmin=V_plot.min(),
                    vmax=V_plot.max()
                )
                plt.colorbar()
                if neur_losses[neuron] != 0:
                    plt.title(f"{neur_losses[neuron]:.3f}")

            fig.tight_layout()
            figures[f'neurons_plot_{plot_counter + 1}'] = fig
            if save_fig:
                fig.savefig(os.path.join(savepath, f"Neurons_Plot_{plot_counter + 1}.png"))

    return figures


def _generate_1d_plots(W, W_init, om, L, min_L, parameters, savepath, save_fig):
    """Generate 1D-specific analysis plots."""
    figures = {}

    N_plot = 1000
    RowsD = 4
    ColumnsD = int(np.ceil(parameters["D"] / RowsD))

    if "epsilon_w" in parameters:
        phi_plot_large = np.linspace(-np.pi, np.pi, N_plot) * parameters["pos_lengthscale"]
        phi_plot_small = np.linspace(-np.pi, np.pi, N_plot)
        I_large = helper_functions.init_irreps_1D(om, phi_plot_large)
        I_small = helper_functions.init_irreps_1D(om, phi_plot_small)
        V_large = np.matmul(W, I_large)
        V_small = np.matmul(W, I_small)

        large_norms = np.linalg.norm(V_large, axis=1)
        V_small = V_small / large_norms[:, None]
        V_large = V_large / large_norms[:, None]
        N = V_large.shape[0]

        # Determine phi for loss calculation
        if "sampling_choice" in parameters and parameters["sampling_choice"] == 1:
            phi_calc = np.random.normal(0, 1, N_plot)
        else:
            phi_calc = phi_plot_small

        # Calculate per-neuron losses
        neur_losses = np.zeros(N)
        if parameters["sep_loss_choice"] == 0:
            for neuron in range(N):
                neur_losses[neuron] = losses.sep_line_Euc(W[neuron, :][None, :], om, phi_calc)
        elif parameters["sep_loss_choice"] == 1:
            chi = helper_functions.calc_chi_line(phi_calc, parameters["sigma_theta"], parameters["f"])
            for neuron in range(N):
                neur_losses[neuron] = losses.sep_line_EucChi(W[neuron, :][None, :], om, phi_calc, chi)
        elif parameters["sep_loss_choice"] == 2:
            for neuron in range(N):
                neur_losses[neuron] = losses.sep_line_Kern(
                    W[neuron, :][None, :], om, phi_calc, parameters["sigma_sq"]
                )

        Vs = [V_small, V_large]
        phi_plots = [phi_plot_small, phi_plot_large]

        # Frequency plot
        om_scrunch_threshold = 0.4
        thresh_percentage = 1
        fig_freq = plotters.freq_plot(W, om, thresh_percentage, om_scrunch_threshold, parameters)
        figures['freq_plot'] = fig_freq
        if save_fig:
            fig_freq.savefig(os.path.join(savepath, "freq_plot.png"))

        # Neural response plots
        for plot_counter, V_plot in enumerate(Vs):
            phi_here = phi_plots[plot_counter]
            plot_min = np.min(V_plot)
            plot_max = np.max(V_plot)

            fig, ax = plt.subplots(RowsD, ColumnsD, figsize=(15, 10))
            [axi.set_axis_off() for axi in ax.ravel()]

            for neuron in range(N):
                plt.subplot(RowsD, ColumnsD, neuron + 1)
                plt.title(f"{neur_losses[neuron]:.6f}")
                plt.plot(phi_here, V_plot[neuron, :])
                plt.ylim([plot_min, plot_max])

            figures[f'neurons_plot_{plot_counter + 1}'] = fig
            if save_fig:
                fig.savefig(os.path.join(savepath, f"Neurons_Plot_{plot_counter + 1}.png"))

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


def run_parameter_sweep(
    base_parameters: Dict[str, Any],
    sweep_params: Dict[str, List[Any]],
    om_init_scheme: int,
    sep_loss_choice: int,
    chi_choice: int,
    W_constrain: int = 0,
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
            opt_results = run_plane_optimization(
                parameters=run_parameters,
                om_init_scheme=om_init_scheme,
                sep_loss_choice=sep_loss_choice,
                chi_choice=chi_choice,
                W_constrain=W_constrain,
                savepath=run_savepath,
                key_seed=key_seed + idx
            )

            # Generate analysis plots
            if generate_plots:
                print(f"\nGenerating analysis plots...")
                try:
                    figures = generate_analysis_plots(run_savepath, counter=0, save_fig=True)
                    plt.close('all')  # Close all figures to free memory
                    print(f"Analysis plots saved to {run_savepath}")
                except Exception as e:
                    print(f"Warning: Failed to generate plots for run {idx}: {e}")

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
        "D": 64, "T": 50000, "K": 1, "N_rand": 150, "phi_std": 1,
        "N_shift": 15, "resample_iters": 5, "save_iters": 5,
        "lambda_pos_init": 0.1, "k_p": -9, "alpha_p": 0.9, "gamma_p": 0.0001,
        "lambda_norm_init": 0.005, "k_norm": -5.25, "alpha_norm": 0.9,
        "gamma_norm": 0.0001, "beta1": 0.9, "beta2": 0.9, "eta": 1e-8,
        "epsilon_w": 0.1, "epsilon_om": 0.1, "dim": 2,
        "sampling_choice": 1, "norm_size": 1,
        "Shift_std": 3, "shift_points_sep": 0,
        "om_init_scheme": 0, "om_init_scale": 2,
        "sigma_sq": 0.04, "sigma_theta": 0.5, "f": 1, "chi_choice": 0,
        "sep_loss_choice": 3
    }

    sweep_params = {
        "lambda_pos_init": [0.05, 0.1, 0.15],
        "k_p": [-8, -9, -10]
    }
