from sweep_analysis import run_parameter_sweep

base_params = {
    "D": 64, "T": 150000, "K": 1, "N_rand": 150, "phi_std": 1,
    "N_shift": 15, "resample_iters": 5, "save_iters": 5, "print_iters": 1000,
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

# Parameters to sweep over
sweep_params = {
    "k_norm": [6, 8, 10, 12, 14]
}

print("Running parameter sweep...")
results = run_parameter_sweep(
    base_parameters=base_params,
    sweep_params=sweep_params,
    om_init_scheme=0,
    sep_loss_choice=3,
    chi_choice=0,
    W_constrain=0,
    base_savepath=None,  # Will auto-generate directory
    key_seed=0,
    generate_plots=True
)