from sweep_analysis_sequential import run_seq_parameter_sweep
from datetime import datetime

base_params = {
    "seq_len": 150, "batch": 1, "D": 65, "T": 5000, "K": 10, "N_shift": 15, "resample_iters": 5,
    "save_iters": 1, "print_iters": 1000,
    "checkpoint_iters": 200,  # Set to int (e.g., 10000) to save checkpoints every N iterations, None to disable
    "lambda_pos_init": 0.1, "k_p": -9, "alpha_p": 0.9, "gamma_p": 0.001,
    "lambda_norm_init": 0.005, "k_norm": -2, "alpha_norm": 0.9,
    "gamma_norm": 0.001, "beta1": 0.9, "beta2": 0.9, "eta": 1e-8,
    "epsilon_g0": 0.1, "epsilon_om": 0.1, "epsilon_s": 0.1, "dim": 2,
    "sampling_choice": 1, "Shift_std": 3, "shift_points_sep": 0,
    "om_init_scheme": 0, "om_init_scale": 2,
    "sigma_sq": 0.04, "sigma_theta": 0.5, "f": 1,
    "causal": True, "decay": 0.95,
    "convergence": False,
    "convergence_window": 1000,
    "convergence_patience": 100,
    "convergence_threshold": 0.1,
    "convergence_smoothing": 20
}

# Parameters to sweep over
sweep_params = [
    {"seq_len": 2, "batch": 500},
    {"seq_len": 5, "batch": 200},
    {"seq_len": 10, "batch": 100},
    {"seq_len": 50, "batch": 20},
    {"seq_len": 100, "batch": 10},
    {"seq_len": 200, "batch": 5},
    {"seq_len": 500, "batch": 2},
    {"seq_len": 1000, "batch": 1}
]
# sweep_params = {
#     # "k_norm": [-4, -3, -2, -1, 0, 2],
#     # "batch": [1]
# }

# Resume mode: Set to True to resume an interrupted sweep
# - Completed runs will be skipped automatically
# - Incomplete runs will resume from their latest checkpoint
# - Set base_savepath to the same path as the interrupted sweep
RESUME = True

print("Running parameter sweep...")
results = run_seq_parameter_sweep(
    base_parameters=base_params,
    sweep_params=sweep_params,
    base_savepath="data/260203/decay",#f"data/{datetime.strftime(datetime.now(), '%y%m%d')}/decay",
    key_seed=0,
    generate_plots=True,
    same_init_across_K=False,  # Set to True to use same initialization for all K runs
    use_dataset_variants=False,  # Set to True to use different dataset variants for each K run
    resume=RESUME  # Set to True to resume from where you left off
)