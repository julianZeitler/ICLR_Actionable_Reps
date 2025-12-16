from PlaneSequential import run_plane_sequential_optimization
from DataHandling import TrajectoryDataset
from sweep_analysis_sequential import generate_analysis_plots
from datetime import datetime
import pickle
import os

base_params = {
    "D": 65, "T": 30000, "K": 1, "N_rand": 150, "phi_std": 1,
    "N_shift": 15, "resample_iters": 5, "save_iters": 5, "print_iters": 1000,
    "lambda_pos_init": 0.1, "k_p": -9, "alpha_p": 0.9, "gamma_p": 0.0001,
    "lambda_norm_init": 0.005, "k_norm": -3, "alpha_norm": 0.9,
    "gamma_norm": 0.0001, "beta1": 0.9, "beta2": 0.9, "eta": 1e-8,
    "epsilon_g0": 0.1, "epsilon_om": 0.1, "epsilon_s": 0.1, "dim": 2,
    "sampling_choice": 1, "norm_size": 1,
    "Shift_std": 3, "shift_points_sep": 0,
    "om_init_scheme": 0, "om_init_scale": 2,
    "sigma_sq": 0.04, "sigma_theta": 0.5, "f": 1
}

# Specify directory to load initial values from (set to None for random initialization)
init_dir = None # "data/251204/sweep_102520/run000"  # Example: "data/251126/123456/"
counter = 0  # Which iteration to load from the previous run

# Load initial values if directory is specified
if init_dir is not None:
    print(f"Loading initial values from {init_dir}")
    with open(os.path.join(init_dir, f'g0_{counter}.pkl'), 'rb') as f:
        g0_init = pickle.load(f)
    with open(os.path.join(init_dir, f'om_{counter}.pkl'), 'rb') as f:
        om_init = pickle.load(f)
    with open(os.path.join(init_dir, f'S_{counter}.pkl'), 'rb') as f:
        S_init = pickle.load(f)
    print("Initial values loaded successfully")
else:
    g0_init = None
    om_init = None
    S_init = None
    print("Using random initialization")

dataloader = TrajectoryDataset('dataset/15122025/', num_workers=6, prefetch_batches=10)

print("Running optimization...")
results = run_plane_sequential_optimization(
    base_params,
    dataloader,
    savepath=None, #f"data/{datetime.strftime(datetime.now(), '%y%m%d')}/correct norm",
    key_seed=0,
    g0_init=g0_init,
    om_init=om_init,
    S_init=S_init
)

print("\nOptimization completed!")
savepath = results['savepath']
print(f"Results saved to: {savepath}")

# Generate analysis plots
print("\nGenerating analysis plots...")
for k in range(base_params["K"]):
    try:
        figures = generate_analysis_plots(savepath, counter=k)
        print(f"Analysis plots for iteration {k} saved to {savepath}")
    except Exception as e:
        print(f"Warning: Failed to generate plots for iteration {k}: {e}")


