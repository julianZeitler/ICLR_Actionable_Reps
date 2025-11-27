# Load up packages
from jax import grad, jit, random
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import os

# And functions I've written
from NRT_functions import helper_functions
from NRT_functions import losses

def run_plane_optimization(parameters, om_init_scheme, sep_loss_choice, chi_choice,
                          W_constrain=0, savepath=None, key_seed=0):
    """
    Run plane optimization with given parameters.

    Args:
        parameters: Dictionary containing optimization parameters (D, T, K, N_rand, etc.)
        om_init_scheme: Frequency initialization scheme (0-6)
        sep_loss_choice: Separation loss choice (0-3)
        chi_choice: Chi function choice (0-2)
        W_constrain: Whether to constrain W matrix (0 or 1)
        savepath: Path to save results. If None, creates timestamped directory
        key_seed: Random seed for JAX

    Returns:
        Dictionary containing optimization results
    """

    # Extract parameters from dict
    T = parameters["T"]
    D = parameters["D"]
    K = parameters["K"]
    N_rand = parameters["N_rand"]
    phi_std = parameters["phi_std"]
    N_shift = parameters["N_shift"]
    Shift_std = parameters["Shift_std"]
    norm_size = parameters["norm_size"]
    sampling_choice = parameters["sampling_choice"]
    shift_points_sep = parameters["shift_points_sep"]
    resample_iters = parameters["resample_iters"]

    lambda_pos_init = parameters["lambda_pos_init"]
    k_p = parameters["k_p"]
    alpha_p = parameters["alpha_p"]
    gamma_p = parameters["gamma_p"]

    lambda_norm_init = parameters["lambda_norm_init"]
    k_norm = parameters["k_norm"]
    alpha_norm = parameters["alpha_norm"]
    gamma_norm = parameters["gamma_norm"]

    epsilon_w = parameters["epsilon_w"]
    epsilon_om = parameters["epsilon_om"]
    beta1 = parameters["beta1"]
    beta2 = parameters["beta2"]
    eta = parameters["eta"]

    save_iters = parameters["save_iters"]
    print_iters = parameters["print_iters"]

    # Setup frequency initialization
    M = int(np.floor((D - 1) / 2))
    om = None
    mask = None

    if om_init_scheme == 0:
        om_init_scale = parameters["om_init_scale"]
    elif om_init_scheme == 1:
        base_freqs = np.array(parameters["base_freqs"])
        om = helper_functions.freq_selector(M)
        om = np.multiply(om, base_freqs[None, :])
    elif om_init_scheme == 1.5:
        base_lengthscale = parameters["base_lengthscale"]
        relative_scale = parameters["relative_scale"]
        relative_angle = parameters["relative_angle"]
        om = helper_functions.freqs_grid_plane(base_lengthscale, relative_scale, relative_angle, M)
    elif om_init_scheme == 2:
        max_freq = parameters["max_freq"]
        om = np.random.randint(1, max_freq, size=[M, 2])
    elif om_init_scheme == 3:
        # Grid params will be used in the loop
        pass
    elif om_init_scheme == 4:
        proportion = parameters["proportion"]
        base_lengthscale_1 = parameters["base_lengthscale_1"]
        relative_scale_1 = parameters["relative_scale_1"]
        relative_angle_1 = parameters["relative_angle_1"]
        base_lengthscale_2 = parameters["base_lengthscale_2"]
        relative_scale_2 = parameters["relative_scale_2"]
        relative_angle_2 = parameters["relative_angle_2"]
        module_angle = parameters["module_angle"]

        M_1 = int(M * proportion)
        M_2 = M - M_1

        om_1 = helper_functions.freqs_grid_plane(base_lengthscale_1, relative_scale_1, relative_angle_1, M_1)
        om_2 = helper_functions.freqs_grid_plane(base_lengthscale_2, relative_scale_2, relative_angle_2, M_2, module_angle)
        om = np.vstack([om_1, om_2])

        if W_constrain:
            mask = np.zeros([D, D])
            mask[1:M_1 + 1, M_1 + 1:] = 1
            mask = mask + np.transpose(mask)
    elif om_init_scheme == 5:
        Q = parameters["Q"]
        mod_init_scale = parameters["mod_init_scale"]
        M_Q = int(M / Q)
    elif om_init_scheme == 6:
        # Parameters for scheme 6
        base_lengthscale_1 = parameters.get("base_lengthscale_1", 1.3)
        relative_scale_1 = parameters.get("relative_scale_1", 1)
        relative_angle_1 = parameters.get("relative_angle_1", np.pi/3)
        relative_scale_2 = parameters.get("relative_scale_2", 1)
        relative_angle_2 = parameters.get("relative_angle_2", np.pi/3)
        proportion = parameters.get("proportion", 0.65)
        M_1 = int(M * proportion)
        M_2 = M - M_1

    # Setup loss functions
    if sep_loss_choice == 0:
        loss_sep = jit(losses.sep_plane_Euc)
        grad_sep_W = jit(grad(losses.sep_plane_Euc, argnums=0))
        grad_sep_om = jit(grad(losses.sep_plane_Euc, argnums=1))
    elif sep_loss_choice == 1:
        sigma_theta = parameters["sigma_theta"]
        f = parameters["f"]
        loss_sep = jit(losses.sep_plane_EucChi)
        grad_sep_W = jit(grad(losses.sep_plane_EucChi, argnums=0))
        grad_sep_om = jit(grad(losses.sep_plane_EucChi, argnums=1))
        if chi_choice == 0:
            calc_chi = jit(helper_functions.calc_chi_plane)
        elif chi_choice == 1:
            calc_chi = jit(helper_functions.calc_chi_plane_euc)
    elif sep_loss_choice == 2:
        sigma_sq = parameters["sigma_sq"]
        loss_sep = jit(losses.sep_plane_Kern)
        grad_sep_W = jit(grad(losses.sep_plane_Kern, argnums=0))
        grad_sep_om = jit(grad(losses.sep_plane_Kern, argnums=1))
    elif sep_loss_choice == 3:
        sigma_sq = parameters["sigma_sq"]
        sigma_theta = parameters["sigma_theta"]
        f = parameters["f"]
        if om_init_scheme == 5:
            loss_sep = jit(losses.sep_plane_KernChi_Module)
            grad_sep_W = jit(grad(losses.sep_plane_KernChi_Module, argnums=0))
            grad_sep_om = jit(grad(losses.sep_plane_KernChi_Module, argnums=1))
        else:
            loss_sep = jit(losses.sep_plane_KernChi)
            grad_sep_W = jit(grad(losses.sep_plane_KernChi, argnums=0))
            grad_sep_om = jit(grad(losses.sep_plane_KernChi, argnums=1))
        if chi_choice == 0:
            calc_chi = jit(helper_functions.calc_chi_plane)
        elif chi_choice == 1:
            calc_chi = jit(helper_functions.calc_chi_plane_euc)
        elif chi_choice == 2:
            calc_chi = jit(helper_functions.calc_chi_plane_exp)

    loss_pos = jit(losses.pos_plane)
    grad_pos_W = jit(grad(losses.pos_plane, argnums=0))
    grad_pos_om = jit(grad(losses.pos_plane, argnums=1))
    loss_norm = jit(losses.norm_plane)
    grad_norm_W = jit(grad(losses.norm_plane, argnums=0))
    grad_norm_om = jit(grad(losses.norm_plane, argnums=1))
    init_irreps = jit(helper_functions.init_irreps_2D)

    key = random.key(key_seed)

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
        os.mkdir(savepath)

    helper_functions.save_parameters_json(parameters, "parameters", savepath)
    print("\nOPTIMISATION BEGINNING\n")

    results = {
        'W_best_list': [],
        'om_best_list': [],
        'W_final_list': [],
        'om_final_list': [],
        'losses_list': [],
        'min_L_list': []
    }

    for counter in range(K):
        # Randomly initialise the weights
        key, subkey1 = random.split(key)
        W = random.normal(subkey1, [D, 2*M+1])

        if om_init_scheme == 0:
            key, subkey2 = random.split(key)
            om = random.uniform(subkey2, [M, 2]) * om_init_scale
        elif om_init_scheme == 3:
            base_freqs = np.ndarray.flatten(parameters["grid_params"][0])
            relative_freqs = np.ndarray.flatten(parameters["grid_params"][1])
            angles = np.ndarray.flatten(parameters["grid_params"][2])
            om = helper_functions.freqs_grid_plane(base_freqs[counter], relative_freqs[counter], angles[counter], M)
        elif om_init_scheme == 5:
            Q = parameters["Q"]
            mod_init_scale = parameters["mod_init_scale"]
            om = np.random.normal(0, mod_init_scale, [Q*4])
        elif om_init_scheme == 6:
            comb_param = parameters["comb_param"]
            om_1 = helper_functions.freqs_grid_plane(base_lengthscale_1, relative_scale_1, relative_angle_1, M_1)
            om_2 = helper_functions.freqs_grid_plane(comb_param[counter, 0], relative_scale_2, relative_angle_2, M_2, comb_param[counter, 1])
            om = np.vstack([om_1, om_2])

        W_init = W
        means_W = jnp.zeros(jnp.shape(W))
        sec_moms_W = jnp.zeros(jnp.shape(W))
        W_best = W

        om_init = om
        means_om = jnp.zeros(jnp.shape(om))
        sec_moms_om = jnp.zeros(jnp.shape(om))
        om_best = om

        Losses = np.zeros([4, int(T / save_iters)])
        min_L = np.zeros([5])
        min_L[1] = np.inf
        L2 = 0
        L3 = 0
        lambda_norm = lambda_norm_init
        lambda_pos = lambda_pos_init
        save_counter = 0

        for step in range(T):
            if step % resample_iters == 0:
                # Create the angles, shifts, irreps, and transforms
                if sampling_choice == 0:
                    phi = (np.random.sample([N_rand, 2]) - 0.5) * np.pi * 2
                elif sampling_choice == 1:
                    phi = np.random.normal(0, phi_std, [N_rand, 2])
                elif sampling_choice == 2:
                    r = np.sqrt(np.random.uniform(size=[N_rand, 1]))
                    theta = np.random.uniform(size=[N_rand, 1]) * 2 * np.pi
                    phi = np.hstack([r * np.cos(theta), r * np.sin(theta)])

                phi_shift = np.random.normal(0, Shift_std, [N_shift, 2])
                phi_norm = norm_size*np.reshape(phi[:, None, :] + phi_shift[None, :, :], [N_rand * N_shift, 2], order='F')
                phi_pos = np.vstack([phi, phi_norm])

                if shift_points_sep:
                    phi = phi + phi_shift[np.random.randint(N_shift), :][None, :]

                if sep_loss_choice == 1 or sep_loss_choice == 3:
                    chi = calc_chi(phi, sigma_theta, f)

            # Separation Term
            if sep_loss_choice == 0:
                L1 = loss_sep(W, om, phi)
                W_grad1 = grad_sep_W(W, om, phi)
                if om_init_scheme == 0:
                    om_grad1 = grad_sep_om(W, om, phi)
            elif sep_loss_choice == 1:
                L1 = loss_sep(W, om, phi, chi)
                W_grad1 = grad_sep_W(W, om, phi, chi)
                if om_init_scheme == 0:
                    om_grad1 = grad_sep_om(W, om, phi, chi)
            elif sep_loss_choice == 2:
                L1 = loss_sep(W, om, phi, sigma_sq)
                W_grad1 = grad_sep_W(W, om, phi, sigma_sq)
                if om_init_scheme == 0:
                    om_grad1 = grad_sep_om(W, om, phi, sigma_sq)
            elif sep_loss_choice == 3:
                L1 = 100*loss_sep(W, om, phi, sigma_sq, chi)
                W_grad1 = 100*grad_sep_W(W, om, phi, sigma_sq, chi)
                if om_init_scheme == 0 or om_init_scheme == 5:
                    om_grad1 = 100*grad_sep_om(W, om, phi, sigma_sq, chi)

            # Positivity Term
            pos = loss_pos(W, om, phi_pos, N_shift)
            if pos <= 0:
                L2_Here = -5
            else:
                L2_Here = np.log(pos) - k_p
            L2 = L2*alpha_p + (1 - alpha_p) * L2_Here
            lambda_pos = lambda_pos * np.exp(L2 * gamma_p)
            W_grad2 = grad_pos_W(W, om, phi_pos, N_shift)
            if om_init_scheme == 0 or om_init_scheme == 5:
                om_grad2 = grad_pos_om(W, om, phi_pos, N_shift)

            # Norm Term
            L3_Here = np.log(loss_norm(W, om, phi, phi_norm)) - k_norm
            L3 = L3 * alpha_norm + (1 - alpha_norm) * L3_Here
            lambda_norm = lambda_norm * np.exp(L3 * gamma_norm)
            W_grad3 = grad_norm_W(W, om, phi, phi_norm)
            if om_init_scheme == 0 or om_init_scheme == 5:
                om_grad3 = grad_norm_om(W, om, phi, phi_norm)

            # Update the moment averages, then bias correct them
            W_grad = W_grad1 + lambda_pos*W_grad2 + lambda_norm*W_grad3
            means_W = beta1*means_W + (1 - beta1)*W_grad
            sec_moms_W = beta2*sec_moms_W + (1 - beta2)*np.power(W_grad, 2)
            means_debiased_W = means_W/(1 - np.power(beta1, step+1))
            sec_moms_debiased_W = sec_moms_W/(1 - np.power(beta2, step + 1))

            if om_init_scheme == 0 or om_init_scheme == 5:
                om_grad = om_grad1 + lambda_pos * om_grad2 + lambda_norm * om_grad3
                means_om = beta1 * means_om + (1 - beta1) * om_grad
                sec_moms_om = beta2 * sec_moms_om + (1 - beta2) * np.power(om_grad, 2)
                means_debiased_om = means_om / (1 - np.power(beta1, step + 1))
                sec_moms_debiased_om = sec_moms_om / (1 - np.power(beta2, step + 1))

            if step % save_iters == 0:
                if L2 > 0:
                    Losses[0, save_counter] = L1 + L2 * lambda_pos
                else:
                    Losses[0, save_counter] = L1
                if L3 > 0:
                    Losses[0, save_counter] += L3 * lambda_norm
                Losses[1, save_counter] = L1
                Losses[2, save_counter] = L2_Here
                Losses[3, save_counter] = L3_Here
                save_counter = save_counter + 1

            if step % print_iters == 0:
                print(f'Iteration: {step}, Loss: {Losses[1, save_counter-1]:.5f}\t Sep: {L1:.5f}\t Pos: {L2_Here:.5f}\t {L2:.5f}\t L P: {lambda_pos:.5f}\t Norm: {L3_Here:.5f}\t {L3:.5f}\t L N: {lambda_norm:.5f}')

            # Potentially save the best results
            if Losses[1, save_counter-1] < min_L[1] and L2 <= 0 and L3 < 0:
                min_L = [save_counter-1, Losses[0, save_counter-1], Losses[1, save_counter-1], Losses[2, save_counter-1]]
                W_best = W
                om_best = om

            # Take parameter step
            W = W - epsilon_w*means_debiased_W/(np.sqrt(sec_moms_debiased_W + eta))
            if om_init_scheme == 0 or om_init_scheme == 5:
                om = om - epsilon_om * means_debiased_om / (np.sqrt(sec_moms_debiased_om + eta))
            if W_constrain:
                W = W.at[:M_1*2 + 2, M_1*2 + 1:].set(0)
                W = W.at[M_1*2 + 2:, 1:M_1*2 + 1].set(0)

        # Save the weights and the losses
        helper_functions.save_obj(W_best, f"W_{counter}", savepath)
        helper_functions.save_obj(W_init, f"W_init_{counter}", savepath)
        helper_functions.save_obj(Losses, f"L_{counter}", savepath)
        helper_functions.save_obj(min_L, f"min_L_{counter}", savepath)
        helper_functions.save_obj(om_best, f"om_{counter}", savepath)
        helper_functions.save_obj(W, f"W_final_{counter}", savepath)
        helper_functions.save_obj(om, f"om_final_{counter}", savepath)

        # Store results
        results['W_best_list'].append(W_best)
        results['om_best_list'].append(om_best)
        results['W_final_list'].append(W)
        results['om_final_list'].append(om)
        results['losses_list'].append(Losses)
        results['min_L_list'].append(min_L)

        print(f"\nDONE ITERATION {counter}: Min_Loss = {min_L[1]:.5f}\n")

    results['savepath'] = savepath
    return results


if __name__ == "__main__":
    # Example usage - set up parameters as before

    ##### Set a load of parameters ######
    T = 150000
    D = 64
    K = 1
    N_rand = 150
    phi_std = 1
    N_shift = 15
    Shift_std = 3
    norm_size = 1
    sampling_choice = 1
    shift_points_sep = 0
    resample_iters = 5

    lambda_pos_init = 0.1
    k_p = -9
    alpha_p = 0.9
    gamma_p = 0.0001

    lambda_norm_init = 0.005
    k_norm = 4
    alpha_norm = 0.9
    gamma_norm = 0.0001

    epsilon_w = 0.1
    epsilon_om = 0.1
    beta1 = 0.9
    beta2 = 0.9
    eta = 1e-8

    save_iters = 5
    print_iters = 250

    # Create parameter dict
    parameters = {
        "D": D, "T": T, "K": K, "N_rand": N_rand, "phi_std": phi_std,
        "N_shift": N_shift, "resample_iters": resample_iters, "save_iters": save_iters, "print_iters": print_iters,
        "lambda_pos_init": lambda_pos_init, "k_p": k_p, "alpha_p": alpha_p, "gamma_p": gamma_p,
        "lambda_norm_init": lambda_norm_init, "k_norm": k_norm, "alpha_norm": alpha_norm,
        "gamma_norm": gamma_norm, "beta1": beta1, "beta2": beta2, "eta": eta,
        "epsilon_w": epsilon_w, "epsilon_om": epsilon_om, "dim": 2,
        "sampling_choice": sampling_choice, "norm_size": norm_size,
        "Shift_std": Shift_std, "shift_points_sep": shift_points_sep
    }

    # Configuration choices
    om_init_scheme = 0
    sep_loss_choice = 3
    chi_choice = 0
    W_constrain = 0

    # Add scheme-specific parameters
    if om_init_scheme == 0:
        om_init_scale = 2
        parameters.update({"om_init_scheme": om_init_scheme, "om_init_scale": om_init_scale})

    if sep_loss_choice == 3:
        sigma_sq = 0.04
        sigma_theta = 0.5
        f = 1
        parameters.update({"sigma_sq": sigma_sq, "sigma_theta": sigma_theta, "f": f, "chi_choice": chi_choice})

    parameters.update({"sep_loss_choice": sep_loss_choice})

    # Run optimization
    results = run_plane_optimization(
        parameters=parameters,
        om_init_scheme=om_init_scheme,
        sep_loss_choice=sep_loss_choice,
        chi_choice=chi_choice,
        W_constrain=W_constrain,
        savepath=None,
        key_seed=0
    )
