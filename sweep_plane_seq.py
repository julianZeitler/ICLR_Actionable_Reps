from sweep_analysis_sequential import run_seq_parameter_sweep
from datetime import datetime

base_params = {
    "D": 65, "T": 30000, "K": 3, "N_rand": 150, "phi_std": 1,
    "N_shift": 15, "resample_iters": 5, "save_iters": 5, "print_iters": 1000,
    "lambda_pos_init": 0.1, "k_p": -9, "alpha_p": 0.9, "gamma_p": 0.0001,
    "lambda_norm_init": 0.005, "k_norm": -2, "alpha_norm": 0.9,
    "gamma_norm": 0.0001, "beta1": 0.9, "beta2": 0.9, "eta": 1e-8,
    "epsilon_g0": 0.1, "epsilon_om": 0.1, "epsilon_s": 0.1, "dim": 2,
    "sampling_choice": 1, "norm_size": 1,
    "Shift_std": 3, "shift_points_sep": 0,
    "om_init_scheme": 0, "om_init_scale": 2,
    "sigma_sq": 0.04, "sigma_theta": 0.5, "f": 1
}

# Parameters to sweep over
sweep_params = {
    "k_norm": [-2, 8, 10]
}

print("Running parameter sweep...")
results = run_seq_parameter_sweep(
    base_parameters=base_params,
    sweep_params=sweep_params,
    base_savepath=f"data/{datetime.strftime(datetime.now(), '%y%m%d')}/no_norm_L3",
    key_seed=0,
    generate_plots=True
)