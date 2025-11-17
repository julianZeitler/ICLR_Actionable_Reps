# Load up packages
from jax import grad, jit, random
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import os

# And functions I've written
from NRT_functions import helper_functions
from NRT_functions import losses

##### Set a load of parameters ######

T = 30000                   # How many gradient steps
D = 65                      # How many neurons
K = 1                       # How many repeats to run
N_rand = 150                # How many random angles, to use for separation loss
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
# save_traj = 1              # Do you want to save a history of w and om?

# Create and save parameter dict
parameters = {"D": D, "T": T, "K": K, "N_rand": N_rand, "N_shift": N_shift, "resample_iters": resample_iters, "save_iters": save_iters,
              "lambda_pos_init": lambda_pos_init, "k_p": k_p, "alpha_p": alpha_p, "gamma_p": gamma_p,
              "lambda_norm_init": lambda_norm_init, "k_norm": k_norm, "alpha_norm": alpha_norm, "gamma_norm": gamma_norm,
              "beta1": beta1, "beta2": beta2, "eta": eta, "epsilon_g0": epsilon_g0, "epsilon_om": epsilon_om, "epsilon_s": epsilon_s, "dim": 2,
              "sampling_choice": sampling_choice, "norm_size": norm_size, "Shift_std": Shift_std, "shift_points_sep": shift_points_sep}

# Frequency choices
# 0: just all the frequencies up to M, requires equivariance enforcing
# 1: square or rectangular lattice at some base frequency up to (D-1)/2 of them
# 1.5: An approximation to any 2D lattice, as close as you can get on integer lattice
# 2: some random set of (D-1)/2 frequencies
# 3: A sweep through a load of frequency grids
# 4: a pair of grids
# 5: a tuneable set of grid modules
# 6: search through some set of pairs of modules
om_init_scheme = 0
M = int(np.floor((D - 1) / 2))

om_init_scale = 2  # om init scale
parameters.update({"om_init_scheme": om_init_scheme, "om_init_scale": om_init_scale})


# Separation loss choices
# 0: Simple euclidean loss
# 1: Chi weighted euclidean
# 2: Kernel loss
# 3: Chi weighted kernel loss
sep_loss_choice = 3
# Chi choices
# 0: Standard inverted gaussian thing
# 1: Just euclidean growing!
# 2: Exponential growing
chi_choice = 0
sigma_sq = 0.04
sigma_theta = 0.5
f = 1

parameters.update({"sigma_sq": sigma_sq, "sigma_theta": sigma_theta, "f": f, "chi_choice": chi_choice})
loss_sep = jit(losses.sep_plane_KernChi_seq)
# loss_sep = losses.sep_plane_KernChi_seq
grad_sep_g0 = jit(grad(losses.sep_plane_KernChi_seq, argnums=0))
grad_sep_om = jit(grad(losses.sep_plane_KernChi_seq, argnums=1))
grad_sep_S = jit(grad(losses.sep_plane_KernChi_seq, argnums=2))
calc_chi = jit(helper_functions.calc_chi_plane)

parameters.update({"sep_loss_choice": sep_loss_choice})
loss_pos = jit(losses.pos_plane_seq)
grad_pos_g0 = jit(grad(losses.pos_plane_seq, argnums=0))
grad_pos_om = jit(grad(losses.pos_plane_seq, argnums=1))
grad_pos_S = jit(grad(losses.pos_plane_seq, argnums=2))
loss_norm = jit(losses.norm_plane_seq)
grad_norm_g0 = jit(grad(losses.norm_plane_seq, argnums=0))
grad_norm_om = jit(grad(losses.norm_plane_seq, argnums=1))
grad_norm_S = jit(grad(losses.norm_plane_seq, argnums=2))
key = random.PRNGKey(0)

# Setup save file locations
today = datetime.strftime(datetime.now(), '%y%m%d')
now = datetime.strftime(datetime.now(), '%H%M%S')
filepath = f"./data/{today}/{now}/"
# Make sure folder is there
if not os.path.isdir(f"./data/"):
    os.mkdir(f"./data/")
if not os.path.isdir(f"data/{today}/"):
    os.mkdir(f"data/{today}/")
# Now make a folder in there for this run
savepath = f"data/{today}/{now}/"
if not os.path.isdir(f"data/{today}/{now}"):
    os.mkdir(f"data/{today}/{now}")

helper_functions.save_obj(parameters, "parameters", savepath)
print("\nOPTIMISATION BEGINNING\n")

for counter in range(K):
    # Randomly initialise g0, losses, moments, and best g0 and loss
    key, subkey1 = random.split(key)     # How to do random things in jax
    g0 = random.normal(subkey1, [2*M+1, 1])    # Activity at origin
    key, subkey2 = random.split(key)
    om = random.uniform(subkey2, [M, 2]) * om_init_scale
    key, subkey3 = random.split(key)
    S = random.normal(subkey3, [2*M+1, 2*M+1])

    g0_init = g0
    means_g0 = jnp.zeros(jnp.shape(g0))     # Moments for ADAM
    sec_moms_g0 = jnp.zeros(jnp.shape(g0))
    g0_best = g0                          # Initialise best g0 somewhere

    om_init = om
    means_om = jnp.zeros(jnp.shape(om))  # Moments for ADAM
    sec_moms_om = jnp.zeros(jnp.shape(om))
    om_best = om

    S_init = S
    means_S = jnp.zeros(jnp.shape(S))     # Moments for ADAM
    sec_moms_S = jnp.zeros(jnp.shape(S))
    S_best = S

    Losses = np.zeros([4, int(T / save_iters)])   # Holder for losses, total, sep, and equi
    min_L = np.zeros([5])  # Step, Loss, Loss_Sep, and Loss_Equi at min Loss
    min_L[1] = np.inf                 # Set min Loss = infty
    L2 = 0                              # So that the positivity moving average has somewhere to start
    L3 = 0                              # Starting norm average
    lambda_norm = lambda_norm_init      # Starting lambda norm
    lambda_pos = lambda_pos_init        # And the positivity
    save_counter = 0

    #if save_traj:
    #    g0_hist = np.zeros([g0.shape[:], int(T / save_iters)])
    #    om_hist = np.zeros([om.shape[:], int(T / save_iters)])

    for step in range(T):
        if step % resample_iters == 0:
            # Create the angles, shifts, irreps, and transforms
            phi = (np.random.sample([N_rand, 2]) - 0.5) * np.pi * 2

            phi_shift = np.random.normal(0, Shift_std, [N_shift, 2])
            # phi_shift = (np.random.randint(1 + 2*Shift_std, size=N_shift) - Shift_std)*2 * np.pi
            phi_norm = norm_size*np.reshape(phi[:, None, :] + phi_shift[None, :, :], [N_rand * N_shift, 2], order='F')
            phi_pos = np.vstack([phi, phi_norm])

            chi = calc_chi(phi, sigma_theta, f)

        # Separation Term
        L1 = 100*loss_sep(g0, om, S, phi, sigma_sq, chi)
        g0_grad1 = 100*grad_sep_g0(g0, om, S, phi, sigma_sq, chi)
        om_grad1 = 100*grad_sep_om(g0, om, S, phi, sigma_sq, chi)
        S_grad1 = 100*grad_sep_S(g0, om, S, phi, sigma_sq, chi)

        # Positivity Term
        pos = loss_pos(g0, om, S, phi_pos, N_shift)
        g0_grad2 = grad_pos_g0(g0, om, S, phi_pos, N_shift)
        om_grad2 = grad_pos_om(g0, om, S, phi_pos, N_shift)
        S_grad2 = grad_pos_S(g0, om, S, phi_pos, N_shift)
        if pos > 0:
            L2_Here = np.log(pos) - k_p
            if L2_Here > 0:
                L2_Here = np.log(L2_Here)
        else:
            L2_Here = -5
        L2 = L2*alpha_p + (1 - alpha_p)*L2_Here
        lambda_pos = lambda_pos*np.exp(L2*gamma_p)

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
            save_counter = save_counter + 1

        if step % print_iters == 0:
            print(f'Iteration: {step}, Loss: {Losses[1, save_counter-1]:.5f}\t Sep: {L1:.5f}\t Pos: {L2_Here:.5f}\t {L2:.5f}\t L P: {lambda_pos:.5f}\t Norm: {L3_Here:.5f}\t {L3:.5f}\t L N: {lambda_norm:.5f}')

        # Potentially save the best results
        if Losses[1, save_counter-1] < min_L[1] and L2 <= 0 and L3 < 0:
            min_L = [save_counter-1, Losses[0, save_counter-1], Losses[1, save_counter-1], Losses[2, save_counter-1]]
            g0_best = g0
            om_best = om

        # Take parameter step
        g0 = g0 - epsilon_g0*means_debiased_g0/(np.sqrt(sec_moms_debiased_g0 + eta))
        om = om - epsilon_om * means_debiased_om / (np.sqrt(sec_moms_debiased_om + eta))
        S = S - epsilon_s*means_debiased_S/(np.sqrt(sec_moms_debiased_S + eta))

    # Now save g0 and the losses
    helper_functions.save_obj(g0_best, f"g0_{counter}", savepath)
    helper_functions.save_obj(g0_init, f"g0_init_{counter}", savepath)
    helper_functions.save_obj(Losses, f"L_{counter}", savepath)
    helper_functions.save_obj(min_L, f"min_L_{counter}", savepath)
    helper_functions.save_obj(om, f"om_{counter}", savepath)
    helper_functions.save_obj(om_best, f"om_{counter}", savepath)
    helper_functions.save_obj(S, f"S_{counter}", savepath)
    helper_functions.save_obj(S_best, f"S_{counter}", savepath)
    helper_functions.save_obj(S_init, f"S_init_{counter}", savepath)
    helper_functions.save_obj(g0, f"g0_final_{counter}", savepath)
    helper_functions.save_obj(om, f"om_final_{counter}", savepath)
    helper_functions.save_obj(S, f"S_final_{counter}", savepath)

    # And print to say iteration done
    print(f"\nDONE ITERATION {counter}: Min_Loss = {min_L[1]:.5f}\n")
