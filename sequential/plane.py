# Load up packages
from jax import grad, jit, random
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import os
import pickle
import glob

# And functions I've written
from nrt import helpers
from nrt import losses


def find_latest_checkpoint(savepath, k):
    """
    Find the latest checkpoint for a given K iteration.

    Args:
        savepath: Base save directory
        k: Which K iteration to look for

    Returns:
        Tuple of (step, checkpoint_dir) or (None, None) if no checkpoint found
    """
    checkpoint_dir = os.path.join(savepath, f"checkpoints/k{k}/")
    if not os.path.exists(checkpoint_dir):
        return None, None

    # Find all g0 checkpoint files
    pattern = os.path.join(checkpoint_dir, "g0_step_*.pkl")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None, None

    # Extract step numbers and find the maximum
    steps = []
    for f in checkpoint_files:
        try:
            step = int(os.path.basename(f).replace("g0_step_", "").replace(".pkl", ""))
            steps.append(step)
        except ValueError:
            continue

    if not steps:
        return None, None

    latest_step = max(steps)
    return latest_step, checkpoint_dir


def load_checkpoint(checkpoint_dir, step):
    """
    Load checkpoint state from disk.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        step: Step number to load

    Returns:
        Dictionary containing checkpoint state, or None if loading fails
    """
    try:
        state = {}

        # Load required files
        with open(os.path.join(checkpoint_dir, f"g0_step_{step}.pkl"), 'rb') as f:
            state['g0'] = pickle.load(f)
        with open(os.path.join(checkpoint_dir, f"om_step_{step}.pkl"), 'rb') as f:
            state['om'] = pickle.load(f)
        with open(os.path.join(checkpoint_dir, f"S_step_{step}.pkl"), 'rb') as f:
            state['S'] = pickle.load(f)
        with open(os.path.join(checkpoint_dir, f"L_step_{step}.pkl"), 'rb') as f:
            state['Losses'] = pickle.load(f)
        with open(os.path.join(checkpoint_dir, f"min_L_step_{step}.pkl"), 'rb') as f:
            state['min_L'] = pickle.load(f)
        with open(os.path.join(checkpoint_dir, f"lambda_pos_step_{step}.pkl"), 'rb') as f:
            state['lambda_pos_arr'] = pickle.load(f)
        with open(os.path.join(checkpoint_dir, f"lambda_norm_step_{step}.pkl"), 'rb') as f:
            state['lambda_norm_arr'] = pickle.load(f)

        # Load best parameters if they exist
        try:
            with open(os.path.join(checkpoint_dir, f"g0_best_step_{step}.pkl"), 'rb') as f:
                state['g0_best'] = pickle.load(f)
            with open(os.path.join(checkpoint_dir, f"om_best_step_{step}.pkl"), 'rb') as f:
                state['om_best'] = pickle.load(f)
            with open(os.path.join(checkpoint_dir, f"S_best_step_{step}.pkl"), 'rb') as f:
                state['S_best'] = pickle.load(f)
        except FileNotFoundError:
            state['g0_best'] = None
            state['om_best'] = None
            state['S_best'] = None

        # Load optimizer state if it exists (for newer checkpoints)
        try:
            with open(os.path.join(checkpoint_dir, f"optimizer_state_step_{step}.pkl"), 'rb') as f:
                state['optimizer_state'] = pickle.load(f)
        except FileNotFoundError:
            state['optimizer_state'] = None

        # Load initial values if they exist
        try:
            with open(os.path.join(checkpoint_dir, f"g0_init_step_{step}.pkl"), 'rb') as f:
                state['g0_init'] = pickle.load(f)
            with open(os.path.join(checkpoint_dir, f"S_init_step_{step}.pkl"), 'rb') as f:
                state['S_init'] = pickle.load(f)
            with open(os.path.join(checkpoint_dir, f"om_init_step_{step}.pkl"), 'rb') as f:
                state['om_init'] = pickle.load(f)
        except FileNotFoundError:
            state['g0_init'] = None
            state['S_init'] = None
            state['om_init'] = None

        # Load scalar state
        try:
            with open(os.path.join(checkpoint_dir, f"scalar_state_step_{step}.pkl"), 'rb') as f:
                state['scalar_state'] = pickle.load(f)
        except FileNotFoundError:
            state['scalar_state'] = None

        state['step'] = step
        return state

    except Exception as e:
        print(f"Failed to load checkpoint at step {step}: {e}")
        return None

def run_plane_sequential_optimization(parameters, dataloader, savepath = None, key_seed = 0, g0_init = None, om_init = None, S_init = None, same_init_across_K = False, resume = False):
    """
    Run plane optimization with given parameters.

    Args:
        parameters: Dictionary containing optimization parameters (D, T, K, N_rand, etc.)
        dataloader: Either a single TrajectoryDataset (used for all K runs) or a list of K TrajectoryDatasets
                   (one for each K run, enabling different dataset variants per run)
        savepath: Path to save results. If None, creates timestamped directory
        key_seed: Random seed for JAX
        g0_init: Optional initial g0 values (if None, random initialization)
        om_init: Optional initial om values (if None, random initialization)
        S_init: Optional initial S values (if None, random initialization)
        same_init_across_K: If True, use the same random initialization for all K runs.
                           If False (default), each K run gets different random initialization.
                           Note: Data batches will still be different across runs regardless of this setting.
        resume: If True, attempt to resume from the latest checkpoint for each K iteration.
                Completed K iterations will be skipped entirely.

    Returns:
        Dictionary containing optimization results
    """

    # Extract parameters from dict
    T = parameters["T"]
    D = parameters["D"]
    M = int(np.floor((D - 1) / 2))
    K = parameters["K"]
    N_shift = parameters["N_shift"]
    Shift_std = parameters["Shift_std"]
    resample_iters = parameters["resample_iters"]

    causal = parameters.get("causal", False)
    decay = parameters.get("decay", None)

    lambda_pos_init = parameters["lambda_pos_init"]
    k_p = parameters["k_p"]
    alpha_p = parameters["alpha_p"]
    gamma_p = parameters["gamma_p"]

    lambda_norm_init = parameters["lambda_norm_init"]
    k_norm = parameters["k_norm"]
    alpha_norm = parameters["alpha_norm"]
    gamma_norm = parameters["gamma_norm"]

    epsilon_om = parameters["epsilon_om"]
    epsilon_g0 = parameters["epsilon_g0"]
    epsilon_s = parameters["epsilon_s"]
    beta1 = parameters["beta1"]
    beta2 = parameters["beta2"]
    eta = parameters["eta"]

    save_iters = parameters["save_iters"]
    print_iters = parameters["print_iters"]
    checkpoint_iters = parameters.get("checkpoint_iters", None)  # None means no checkpointing

    om_init_scale = parameters["om_init_scale"]
    sigma_sq = parameters["sigma_sq"]
    sigma_theta = parameters["sigma_theta"]
    f = parameters["f"]

    # Convergence check parameters
    convergence = parameters.get("convergence", False)
    convergence_window = parameters.get("convergence_window", 50)  # Window size in save_iters units
    convergence_patience = parameters.get("convergence_patience", 50)  # Checks without improvement
    convergence_threshold = parameters.get("convergence_threshold", 0.05)  # Min relative improvement
    convergence_smoothing = parameters.get("convergence_smoothing", 200)  # Smooth over N recent values

    if causal:
        loss_sep = jit(losses.sep_plane_KernChi_seq_causal, static_argnames="decay")
        # loss_sep = losses.sep_plane_KernChi_seq
        grad_sep_g0 = jit(grad(losses.sep_plane_KernChi_seq_causal, argnums=0), static_argnames="decay")
        grad_sep_om = jit(grad(losses.sep_plane_KernChi_seq_causal, argnums=1), static_argnames="decay")
        grad_sep_S = jit(grad(losses.sep_plane_KernChi_seq_causal, argnums=2), static_argnames="decay")
        calc_chi = jit(helpers.calc_chi_plane_causal)
    else:
        loss_sep = jit(losses.sep_plane_KernChi_seq)
        # loss_sep = losses.sep_plane_KernChi_seq
        grad_sep_g0 = jit(grad(losses.sep_plane_KernChi_seq, argnums=0))
        grad_sep_om = jit(grad(losses.sep_plane_KernChi_seq, argnums=1))
        grad_sep_S = jit(grad(losses.sep_plane_KernChi_seq, argnums=2))
        calc_chi = jit(helpers.calc_chi_plane)
    
    loss_pos = jit(losses.pos_plane_seq)
    # loss_pos = losses.pos_plane_seq
    grad_pos_g0 = jit(grad(losses.pos_plane_seq, argnums=0))
    grad_pos_om = jit(grad(losses.pos_plane_seq, argnums=1))
    grad_pos_S = jit(grad(losses.pos_plane_seq, argnums=2))
    loss_norm = jit(losses.norm_plane_seq)
    # loss_norm = losses.norm_plane_seq
    grad_norm_g0 = jit(grad(losses.norm_plane_seq, argnums=0))
    grad_norm_om = jit(grad(losses.norm_plane_seq, argnums=1))
    grad_norm_S = jit(grad(losses.norm_plane_seq, argnums=2))
    key = random.key(key_seed)
    key_init = key  # Save initial key for potential reuse across K iterations

    # Setup save file locations
    if savepath is None:
        today = datetime.strftime(datetime.now(), '%y%m%d')
        now = datetime.strftime(datetime.now(), '%H%M%S')
        if not os.path.isdir(f"./data/"):
            os.mkdir(f"./data/")
        if not os.path.isdir(f"data/{today}/"):
            os.mkdir(f"data/{today}/")
        savepath = f"data/{today}/{now}/"
    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)

    helpers.save_parameters_json(parameters, "parameters", savepath)
    print("\nOPTIMISATION BEGINNING\n")

    results = {
        'g0_best_list': [],
        'om_best_list': [],
        'S_best_list': [],
        'g0_final_list': [],
        'om_final_list': [],
        'S_final_list': [],
        'losses_list': [],
        'min_L_list': [],
        'lambda_pos_list': [],
        'lambda_norm_list': []
    }

    for counter in range(K):
        # Check if this K iteration is already complete (has final output files)
        if resume and os.path.exists(os.path.join(savepath, f"g0_final_{counter}.pkl")):
            print(f"\n>>> K iteration {counter} already complete, loading saved results...")
            # Load saved results to populate the results dict
            with open(os.path.join(savepath, f"g0_{counter}.pkl"), 'rb') as file:
                g0_best = pickle.load(file)
            with open(os.path.join(savepath, f"om_{counter}.pkl"), 'rb') as file:
                om_best = pickle.load(file)
            with open(os.path.join(savepath, f"S_{counter}.pkl"), 'rb') as file:
                S_best = pickle.load(file)
            with open(os.path.join(savepath, f"g0_final_{counter}.pkl"), 'rb') as file:
                g0 = pickle.load(file)
            with open(os.path.join(savepath, f"om_final_{counter}.pkl"), 'rb') as file:
                om = pickle.load(file)
            with open(os.path.join(savepath, f"S_final_{counter}.pkl"), 'rb') as file:
                S = pickle.load(file)
            with open(os.path.join(savepath, f"L_{counter}.pkl"), 'rb') as file:
                Losses = pickle.load(file)
            with open(os.path.join(savepath, f"min_L_{counter}.pkl"), 'rb') as file:
                min_L = pickle.load(file)
            with open(os.path.join(savepath, f"lambda_pos_{counter}.pkl"), 'rb') as file:
                Lambdas_pos = pickle.load(file)
            with open(os.path.join(savepath, f"lambda_norm_{counter}.pkl"), 'rb') as file:
                Lambdas_norm = pickle.load(file)

            results["g0_best_list"].append(g0_best)
            results["om_best_list"].append(om_best)
            results["S_best_list"].append(S_best)
            results["g0_final_list"].append(g0)
            results["om_final_list"].append(om)
            results["S_final_list"].append(S)
            results["losses_list"].append(Losses)
            results["min_L_list"].append(min_L)
            results["lambda_pos_list"].append(Lambdas_pos)
            results["lambda_norm_list"].append(Lambdas_norm)
            print(f"SKIPPED ITERATION {counter}: Already complete with Min_Loss = {min_L[1]:.5f}\n")
            continue

        # Select appropriate dataloader for this K iteration
        if isinstance(dataloader, list):
            current_dataloader = dataloader[counter]
        else:
            current_dataloader = dataloader

        # Check for checkpoint to resume from
        start_step = 0
        checkpoint_loaded = False

        if resume:
            latest_step, checkpoint_dir = find_latest_checkpoint(savepath, counter)
            if latest_step is not None:
                print(f"\n>>> Found checkpoint at step {latest_step} for K iteration {counter}")
                checkpoint_state = load_checkpoint(checkpoint_dir, latest_step)
                if checkpoint_state is not None:
                    checkpoint_loaded = True
                    start_step = latest_step + 1
                    print(f">>> Resuming from step {start_step}")

        if checkpoint_loaded:
            # Restore state from checkpoint
            g0 = checkpoint_state['g0']
            om = checkpoint_state['om']
            S = checkpoint_state['S']

            # Restore best parameters
            if checkpoint_state['g0_best'] is not None:
                g0_best = checkpoint_state['g0_best']
                om_best = checkpoint_state['om_best']
                S_best = checkpoint_state['S_best']
            else:
                g0_best = g0
                om_best = om
                S_best = S

            # Restore initial values
            if checkpoint_state['g0_init'] is not None:
                g0_init_save = checkpoint_state['g0_init']
                om_init_save = checkpoint_state['om_init']
                S_init_save = checkpoint_state['S_init']
            else:
                g0_init_save = g0
                om_init_save = om
                S_init_save = S

            # Restore losses array (expand to full size)
            checkpoint_losses = checkpoint_state['Losses']
            Losses = np.zeros([4, int(T / save_iters)])
            Losses[:, :checkpoint_losses.shape[1]] = checkpoint_losses

            # Restore lambda arrays
            checkpoint_lambdas_pos = checkpoint_state['lambda_pos_arr']
            checkpoint_lambdas_norm = checkpoint_state['lambda_norm_arr']
            Lambdas_pos = np.zeros(int(T / save_iters))
            Lambdas_norm = np.zeros(int(T / save_iters))
            Lambdas_pos[:len(checkpoint_lambdas_pos)] = checkpoint_lambdas_pos
            Lambdas_norm[:len(checkpoint_lambdas_norm)] = checkpoint_lambdas_norm

            min_L = checkpoint_state['min_L']

            # Restore optimizer state if available
            if checkpoint_state['optimizer_state'] is not None:
                opt_state = checkpoint_state['optimizer_state']
                means_g0 = opt_state['means_g0']
                sec_moms_g0 = opt_state['sec_moms_g0']
                means_om = opt_state['means_om']
                sec_moms_om = opt_state['sec_moms_om']
                means_S = opt_state['means_S']
                sec_moms_S = opt_state['sec_moms_S']
            else:
                # If no optimizer state, initialize to zeros (will warm up quickly)
                print(">>> Warning: No optimizer state in checkpoint, reinitializing ADAM moments")
                means_g0 = jnp.zeros(jnp.shape(g0))
                sec_moms_g0 = jnp.zeros(jnp.shape(g0))
                means_om = jnp.zeros(jnp.shape(om))
                sec_moms_om = jnp.zeros(jnp.shape(om))
                means_S = jnp.zeros(jnp.shape(S))
                sec_moms_S = jnp.zeros(jnp.shape(S))

            # Restore scalar state if available
            if checkpoint_state['scalar_state'] is not None:
                scalar_state = checkpoint_state['scalar_state']
                L2 = scalar_state['L2']
                L3 = scalar_state['L3']
                lambda_pos = scalar_state['lambda_pos']
                lambda_norm = scalar_state['lambda_norm']
                save_counter = scalar_state['save_counter']

                # Restore convergence state
                if convergence and 'loss_history' in scalar_state:
                    loss_history = scalar_state['loss_history']
                    no_improvement_count = scalar_state['no_improvement_count']
                    converged = False
                elif convergence:
                    loss_history = {'L1': [], 'L2': [], 'L3': []}
                    no_improvement_count = 0
                    converged = False
            else:
                # Estimate scalar state from checkpoint
                L2 = 0
                L3 = 0
                lambda_pos = Lambdas_pos[len(checkpoint_lambdas_pos) - 1] if len(checkpoint_lambdas_pos) > 0 else lambda_pos_init
                lambda_norm = Lambdas_norm[len(checkpoint_lambdas_norm) - 1] if len(checkpoint_lambdas_norm) > 0 else lambda_norm_init
                save_counter = len(checkpoint_lambdas_pos)

                if convergence:
                    loss_history = {'L1': [], 'L2': [], 'L3': []}
                    no_improvement_count = 0
                    converged = False
        else:
            # Reset key to initial state if using same initialization across K runs
            if same_init_across_K and counter > 0:
                key = key_init

            # Randomly initialise g0, losses, moments, and best g0 and loss
            if g0_init is None:
                key, subkey1 = random.split(key)
                g0 = random.normal(subkey1, [2*M+1])    # Activity at origin
            else:
                g0 = g0_init

            if om_init is None:
                key, subkey2 = random.split(key)
                om = random.uniform(subkey2, [M, 2]) * om_init_scale
            else:
                om = om_init

            if S_init is None:
                key, subkey3 = random.split(key)
                S = random.normal(subkey3, [2*M+1, 2*M+1])
            else:
                S = S_init

            g0_init_save = g0
            means_g0 = jnp.zeros(jnp.shape(g0))     # Moments for ADAM
            sec_moms_g0 = jnp.zeros(jnp.shape(g0))
            g0_best = g0                          # Initialise best g0 somewhere

            om_init_save = om
            means_om = jnp.zeros(jnp.shape(om))  # Moments for ADAM
            sec_moms_om = jnp.zeros(jnp.shape(om))
            om_best = om

            S_init_save = S
            means_S = jnp.zeros(jnp.shape(S))     # Moments for ADAM
            sec_moms_S = jnp.zeros(jnp.shape(S))
            S_best = S

            Losses = np.zeros([4, int(T / save_iters)])
            Lambdas_pos = np.zeros(int(T / save_iters))
            Lambdas_norm = np.zeros(int(T / save_iters))
            min_L = np.zeros([5])
            min_L[1] = np.inf
            L2 = 0
            L3 = 0
            lambda_norm = lambda_norm_init
            lambda_pos = lambda_pos_init
            save_counter = 0

            # Convergence tracking
            if convergence:
                loss_history = {'L1': [], 'L2': [], 'L3': []}  # Track recent losses
                no_improvement_count = 0
                converged = False

        # Track if this is the first step (for forcing data load after resume)
        first_step_after_resume = checkpoint_loaded

        for step in range(start_step, T):
            # Load data on resample iterations, or on first step after resuming
            if step % resample_iters == 0 or first_step_after_resume:
                phi = current_dataloader.get_batch(int(step/resample_iters))
                B, L = phi.shape[0], phi.shape[1]
                
                phi_shift = np.random.normal(0, Shift_std, [N_shift, 2])
                # Offset trajectories    [B,N_shift,L,2]            [B   ,N_shift,L   ,2]
                phi_norm = np.reshape(phi[:,None   ,:,:] + phi_shift[None,:      ,None,:], [B*N_shift,L,2])

                phi_pos = np.concatenate([phi, phi_norm], axis=0)
                chi = calc_chi(phi, sigma_theta, f)
                first_step_after_resume = False  # Reset after first data load

            # Separation Term
            if decay:
                L1 = 100*loss_sep(g0, om, S, phi, sigma_sq, chi, decay)
                g0_grad1 = 100*grad_sep_g0(g0, om, S, phi, sigma_sq, chi, decay)
                om_grad1 = 100*grad_sep_om(g0, om, S, phi, sigma_sq, chi, decay)
                S_grad1 = 100*grad_sep_S(g0, om, S, phi, sigma_sq, chi, decay)
            else:
                L1 = 100*loss_sep(g0, om, S, phi, sigma_sq, chi)
                g0_grad1 = 100*grad_sep_g0(g0, om, S, phi, sigma_sq, chi)
                om_grad1 = 100*grad_sep_om(g0, om, S, phi, sigma_sq, chi)
                S_grad1 = 100*grad_sep_S(g0, om, S, phi, sigma_sq, chi)
            
            # Positivity Term
            pos = loss_pos(g0, om, S, phi_pos)
            if pos > 0:
                L2_Here = np.log(pos) - k_p
            else:
                L2_Here = -5
            L2 = L2*alpha_p + (1 - alpha_p)*L2_Here
            lambda_pos = lambda_pos*np.exp(L2*gamma_p)
            g0_grad2 = grad_pos_g0(g0, om, S, phi_pos)
            om_grad2 = grad_pos_om(g0, om, S, phi_pos)
            S_grad2 = grad_pos_S(g0, om, S, phi_pos)

            # Norm Term
            L3_Here = np.log(loss_norm(g0, om, S, phi, phi_norm)) - k_norm
            L3 = L3 * alpha_norm + (1 - alpha_norm) * L3_Here
            lambda_norm = lambda_norm * np.exp(L3 * gamma_norm)
            g0_grad3 = grad_norm_g0(g0, om, S, phi, phi_norm)
            om_grad3 = grad_norm_om(g0, om, S, phi, phi_norm)
            S_grad3 = grad_norm_S(g0, om, S, phi, phi_norm)

            # Update the moment averages, then bias correct them
            g0_grad = g0_grad1 + lambda_pos*g0_grad2 + lambda_norm*g0_grad3
            means_g0 = beta1*means_g0 + (1 - beta1)*g0_grad
            sec_moms_g0 = beta2*sec_moms_g0 + (1 - beta2)*np.power(g0_grad, 2)
            means_debiased_g0 = means_g0/(1 - np.power(beta1, step+1))
            sec_moms_debiased_g0 = sec_moms_g0/(1 - np.power(beta2, step + 1))

            om_grad = om_grad1 + lambda_pos * om_grad2 + lambda_norm * om_grad3
            means_om = beta1 * means_om + (1 - beta1) * om_grad
            sec_moms_om = beta2 * sec_moms_om + (1 - beta2) * np.power(om_grad, 2)
            means_debiased_om = means_om / (1 - np.power(beta1, step + 1))
            sec_moms_debiased_om = sec_moms_om / (1 - np.power(beta2, step + 1))

            S_grad = S_grad1 + lambda_pos*S_grad2 + lambda_norm*S_grad3
            means_S = beta1*means_S + (1 - beta1)*S_grad
            sec_moms_S = beta2*sec_moms_S + (1 - beta2)*np.power(S_grad, 2)
            means_debiased_S = means_S/(1 - np.power(beta1, step+1))
            sec_moms_debiased_S = sec_moms_S/(1 - np.power(beta2, step + 1))

            if step % save_iters == 0:        # Save and print the appropriate losses
                if L2 > 0:
                    Losses[0, save_counter] = L1 + L2 * lambda_pos
                else:
                    Losses[0, save_counter] = L1
                if L3 > 0:
                    Losses[0, save_counter] += L3 * lambda_norm
                Losses[1, save_counter] = L1
                Losses[2, save_counter] = L2_Here
                Losses[3, save_counter] = L3_Here
                Lambdas_pos[save_counter] = lambda_pos
                Lambdas_norm[save_counter] = lambda_norm
                save_counter = save_counter + 1

                # Convergence check
                if convergence:
                    loss_history['L1'].append(float(L1))
                    loss_history['L2'].append(float(L2_Here))
                    loss_history['L3'].append(float(L3_Here))

                    # Only check after we have enough history
                    if len(loss_history['L1']) >= convergence_window + convergence_smoothing:
                        # Compute rolling average over entire range including current
                        window_start = max(0, len(loss_history['L1']) - convergence_window)
                        kernel = np.ones(convergence_smoothing) / convergence_smoothing

                        L1_smoothed = np.convolve(loss_history['L1'][window_start:], kernel, mode='valid')
                        L2_smoothed = np.convolve(loss_history['L2'][window_start:], kernel, mode='valid')
                        L3_smoothed = np.convolve(loss_history['L3'][window_start:], kernel, mode='valid')

                        # Current smoothed value is the last entry, historical min is from earlier
                        L1_current = L1_smoothed[-1]
                        L2_current = L2_smoothed[-1]
                        L3_current = L3_smoothed[-1]

                        L1_window_min = np.min(L1_smoothed[:-1]) if len(L1_smoothed) > 1 else L1_current
                        L2_window_min = np.min(L2_smoothed[:-1]) if len(L2_smoothed) > 1 else L2_current
                        L3_window_min = np.min(L3_smoothed[:-1]) if len(L3_smoothed) > 1 else L3_current

                        # Check if any loss has improved by threshold
                        L1_improved = (L1_window_min - L1_current) / (abs(L1_window_min) + 1e-8) > convergence_threshold
                        L2_improved = (L2_window_min - L2_current) / (abs(L2_window_min) + 1e-8) > convergence_threshold
                        L3_improved = (L3_window_min - L3_current) / (abs(L3_window_min) + 1e-8) > convergence_threshold

                        if not (L1_improved or L2_improved or L3_improved):
                            no_improvement_count += 1
                            if no_improvement_count >= convergence_patience:
                                converged = True
                                print(f'\n>>> Convergence achieved at step {step}')
                                print(f'    No improvement in L1, L2, or L3 for {convergence_patience} checks')
                                print(f'    L1: {L1_current:.5f}, L2: {L2_current:.5f}, L3: {L3_current:.5f}')
                        else:
                            no_improvement_count = 0  # Reset counter if any loss improved

            if converged if convergence else False:
                break

            if step % print_iters == 0:
                print(f'Iteration: {step}, Loss: {Losses[1, save_counter-1]:.5f}\t Sep: {L1:.5f}\t Pos: {L2_Here:.5f}\t {L2:.5f}\t L P: {lambda_pos:.5f}\t Norm: {L3_Here:.5f}\t {L3:.5f}\t L N: {lambda_norm:.5f}')

            # Save checkpoint if checkpoint_iters is set
            if checkpoint_iters is not None and step > 0 and step % checkpoint_iters == 0:
                checkpoint_dir = os.path.join(savepath, f"checkpoints/k{counter}/")
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_losses = Losses[:, :save_counter].copy()  # Only save up to current point
                checkpoint_lambdas_pos = Lambdas_pos[:save_counter].copy()
                checkpoint_lambdas_norm = Lambdas_norm[:save_counter].copy()

                helpers.save_obj(g0, f"g0_step_{step}", checkpoint_dir)
                helpers.save_obj(om, f"om_step_{step}", checkpoint_dir)
                helpers.save_obj(S, f"S_step_{step}", checkpoint_dir)
                helpers.save_obj(checkpoint_losses, f"L_step_{step}", checkpoint_dir)
                helpers.save_obj(min_L, f"min_L_step_{step}", checkpoint_dir)
                helpers.save_obj(checkpoint_lambdas_pos, f"lambda_pos_step_{step}", checkpoint_dir)
                helpers.save_obj(checkpoint_lambdas_norm, f"lambda_norm_step_{step}", checkpoint_dir)

                # Also save best parameters if they exist
                if min_L[1] < np.inf:
                    helpers.save_obj(g0_best, f"g0_best_step_{step}", checkpoint_dir)
                    helpers.save_obj(om_best, f"om_best_step_{step}", checkpoint_dir)
                    helpers.save_obj(S_best, f"S_best_step_{step}", checkpoint_dir)

                # Save optimizer state (ADAM moments)
                optimizer_state = {
                    'means_g0': means_g0,
                    'sec_moms_g0': sec_moms_g0,
                    'means_om': means_om,
                    'sec_moms_om': sec_moms_om,
                    'means_S': means_S,
                    'sec_moms_S': sec_moms_S,
                }
                helpers.save_obj(optimizer_state, f"optimizer_state_step_{step}", checkpoint_dir)

                # Save initial values for proper result saving later
                helpers.save_obj(g0_init_save, f"g0_init_step_{step}", checkpoint_dir)
                helpers.save_obj(om_init_save, f"om_init_step_{step}", checkpoint_dir)
                helpers.save_obj(S_init_save, f"S_init_step_{step}", checkpoint_dir)

                # Save scalar state for resumption
                scalar_state = {
                    'L2': L2,
                    'L3': L3,
                    'lambda_pos': lambda_pos,
                    'lambda_norm': lambda_norm,
                    'save_counter': save_counter,
                }
                if convergence:
                    scalar_state['loss_history'] = loss_history
                    scalar_state['no_improvement_count'] = no_improvement_count
                helpers.save_obj(scalar_state, f"scalar_state_step_{step}", checkpoint_dir)

                print(f"  → Checkpoint saved at step {step}")

            # Potentially save the best results
            if Losses[1, save_counter-1] < min_L[1] and L2 <= 0 and L3 < 0:
                min_L = [save_counter-1, Losses[0, save_counter-1], Losses[1, save_counter-1], Losses[2, save_counter-1]]
                g0_best = g0
                om_best = om
                S_best = S

            # Take parameter step
            g0 = g0 - epsilon_g0*means_debiased_g0/(np.sqrt(sec_moms_debiased_g0 + eta))
            om = om - epsilon_om * means_debiased_om / (np.sqrt(sec_moms_debiased_om + eta))
            S = S - epsilon_s*means_debiased_S/(np.sqrt(sec_moms_debiased_S + eta))

        # Truncate arrays if convergence stopped training early
        if convergence and save_counter < len(Losses[0]):
            Losses = Losses[:, :save_counter]
            Lambdas_pos = Lambdas_pos[:save_counter]
            Lambdas_norm = Lambdas_norm[:save_counter]

        # Now save g0 and the losses
        print(f"Saving results to {savepath}...")
        helpers.save_obj(g0_best, f"g0_{counter}", savepath)
        helpers.save_obj(g0_init_save, f"g0_init_{counter}", savepath)
        helpers.save_obj(Losses, f"L_{counter}", savepath)
        helpers.save_obj(min_L, f"min_L_{counter}", savepath)
        helpers.save_obj(om, f"om_{counter}", savepath)
        helpers.save_obj(om_best, f"om_{counter}", savepath)
        helpers.save_obj(S, f"S_{counter}", savepath)
        helpers.save_obj(S_best, f"S_{counter}", savepath)
        helpers.save_obj(S_init_save, f"S_init_{counter}", savepath)
        helpers.save_obj(g0, f"g0_final_{counter}", savepath)
        helpers.save_obj(om, f"om_final_{counter}", savepath)
        helpers.save_obj(S, f"S_final_{counter}", savepath)
        helpers.save_obj(Lambdas_pos, f"lambda_pos_{counter}", savepath)
        helpers.save_obj(Lambdas_norm, f"lambda_norm_{counter}", savepath)
        print(f"Saved 14 pickle files for iteration {counter}")

        results["g0_best_list"].append(g0_best)
        results["om_best_list"].append(om_best)
        results["S_best_list"].append(S_best)
        results["g0_final_list"].append(g0)
        results["om_final_list"].append(om)
        results["S_final_list"].append(S)
        results["losses_list"].append(Losses)
        results["min_L_list"].append(min_L)
        results["lambda_pos_list"].append(Lambdas_pos)
        results["lambda_norm_list"].append(Lambdas_norm)

        # And print to say iteration done
        print(f"\nDONE ITERATION {counter}: Min_Loss = {min_L[1]:.5f}\n")
    
    results["savepath"] = savepath
    return results

if __name__ == "__main__":
    T = 5000                   # How many gradient steps
    D = 65                      # How many neurons
    K = 1                       # How many repeats to run
    N_rand = 150                # How many random angles, to use for separation loss
    phi_std = 1
    N_shift = 15                # Number of other rooms to measure positivity and norm
    Shift_std = 3               # Standard deviation of normal from which to sample shifts
    norm_size = 1               # How much bigger to make the room you take the norm over
    sampling_choice = 1         # 0 for square room, 1 for normal distribution, 2 for circular room
    shift_points_sep = 0        # 0 for room centered on (0,0), 1 for room whose centre shifts by shift_std every step
    resample_iters = 5          # How often to resample random points

    # Set of parameters for the positivity geco
    lambda_pos_init = 0.1         # 15 for euc, 5 for kern, 150 for euc_A (maybe 0.5 after N), 0.1 for kern_A, 0.05 BEFORE  # Initial positivity loss weighting
    k_p = -9                    # Positivity target
    alpha_p = 0.9               # Smoothing of positivity dynamics
    gamma_p = 0.0001             # Proportionality constant

    # Norm GECO parameters
    lambda_norm_init = 0.005      # Initial norm loss weighting
    k_norm = 4                   # norm target
    alpha_norm = 0.9             # Smoothing of norm dynamics
    gamma_norm = 0.0001         # Proportionality constant from mismatch to constrant movement

    # Parameters for ADAM
    epsilon_g0 = 0.1             # Step size parameter g0
    epsilon_om = 0.1            # Frequency step size
    epsilon_s = 0.1            # S step size
    beta1 = 0.9                 # Exp moving average parameter for first moment
    beta2 = 0.9                 # exp moving average parameter for second moment
    eta = 1e-8                   # Small regularising, non-exploding thingy, not v important it seems

    # Printing and saving
    save_iters = 5               # How often to save results
    print_iters = 250            # How often to print results
    checkpoint_iters = None      # Set to int (e.g., 10000) to save checkpoints every N iterations, None to disable

    sigma_sq = 0.04
    sigma_theta = 0.5
    f = 1

    om_init_scale = 2

    # Convergence check parameters
    convergence = False            # Set to True to enable convergence checking
    convergence_window = 50        # Look back window size (in save_iters units)
    convergence_patience = 10      # Number of checks without improvement before stopping
    convergence_threshold = 0.001  # Minimum relative improvement required (0.1%)
    convergence_smoothing = 5      # Smooth current losses over this many recent values

    parameters = {
        "D": D, "T": T, "K": K, "N_rand": N_rand, "phi_std": phi_std,
        "N_shift": N_shift, "resample_iters": resample_iters, "save_iters": save_iters, "print_iters": print_iters,
        "checkpoint_iters": checkpoint_iters,
        "lambda_pos_init": lambda_pos_init, "k_p": k_p, "alpha_p": alpha_p, "gamma_p": gamma_p,
        "lambda_norm_init": lambda_norm_init, "k_norm": k_norm, "alpha_norm": alpha_norm,
        "gamma_norm": gamma_norm, "beta1": beta1, "beta2": beta2, "eta": eta,
        "epsilon_g0": epsilon_g0, "epsilon_om": epsilon_om, "epsilon_s": epsilon_s, "dim": 2,
        "sampling_choice": sampling_choice, "norm_size": norm_size, "om_init_scale": om_init_scale,
        "Shift_std": Shift_std, "shift_points_sep": shift_points_sep, "sigma_sq": sigma_sq, "sigma_theta": sigma_theta, "f": f,
        "convergence": convergence, "convergence_window": convergence_window,
        "convergence_patience": convergence_patience, "convergence_threshold": convergence_threshold,
        "convergence_smoothing": convergence_smoothing
    }

    results = run_plane_sequential_optimization(
        parameters=parameters,
        savepath=None, #f"data/{datetime.strftime(datetime.now(), '%y%m%d')}/seq_no_norm_noise",
        key_seed=0
    )
