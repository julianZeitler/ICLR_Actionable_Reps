import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
from typing import Dict, Any
import json
import pickle

from nrt.scores import GridScorer
from nrt import helpers
from nrt import losses

def quantitative_analysis_seq(Vs, widths, counter, res=70, savepath=None):
    scores = {}
    fig_score, axes = plt.subplots(1, 3, figsize=(12, 5))
    for idx, V in enumerate(Vs):
        maps = [V[i,:] for i in range(V.shape[0])]

        starts = [0.2] * 10
        ends = np.linspace(0.4, 1.0, num=10)
        box_width=widths[idx]
        box_height=widths[idx]
        coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
        masks_parameters = zip(starts, ends.tolist())
        scorer = GridScorer(res, coord_range, masks_parameters)

        score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
            *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(maps)])
        score_60 = np.nan_to_num(score_60)
        score_90 = np.nan_to_num(score_90)

        axes[idx].hist(score_60, range=(-1,2.5), bins=15)
        axes[idx].set_xlabel('Grid score')
        axes[idx].set_ylabel('Count')            
        
        max_score = np.max(score_60)
        mean_score = np.mean(score_60)
        axes[idx].text(0.05, 0.95, f'Max: {max_score:.3f}\nMean: {mean_score:.3f}',
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        if idx == 0: # Small
            axes[idx].set_title('Small Ratemaps')

            scores["sm_60"] = score_60
            scores["sm_90"] = score_90
            scores["sm_60_max"] = max_score
            scores["sm_60_mean"] = mean_score
            scores["sm_90_max"] = np.max(score_90)
            scores["sm_90_mean"] = np.mean(score_90)
        elif idx == 1: # Medium
            axes[idx].set_title('Medium Ratemaps')

            scores["md_60"] = score_60
            scores["md_90"] = score_90
            scores["md_60_max"] = max_score
            scores["md_60_mean"] = mean_score
            scores["md_90_max"] = np.max(score_90)
            scores["md_90_mean"] = np.mean(score_90)
        elif idx == 2: # Large
            axes[idx].set_title('Large Ratemaps')

            scores["lg_60"] = score_60
            scores["lg_90"] = score_90
            scores["lg_60_max"] = max_score
            scores["lg_60_mean"] = mean_score
            scores["lg_90_max"] = np.max(score_90)
            scores["lg_90_mean"] = np.mean(score_90)

    fig_score.tight_layout()
    if savepath:
      fig_score.savefig(os.path.join(savepath, f"grid_scores_{counter}.png"))
      helpers.save_parameters_json(scores, f"grid_scores_{counter}", savepath)

    return fig_score, scores

def validation_losses(
    run_path: str,
    validation_path: str = "dataset/validation",
    use_best_checkpoints: bool = False
) -> Dict[str, Any]:
    """
    Compute validation losses for checkpoints saved during training.

    Args:
        run_path: Path to the run directory containing checkpoints/ and parameters.json
        validation_path: Path to directory containing validation datasets
        use_best_checkpoints: If True, use g0_best_step_*.pkl etc., otherwise use g0_step_*.pkl

    Returns:
        Dictionary with structure:
        {
            'steps': [200, 400, ...],
            'per_k': {
                0: {
                    'per_dataset': {'dataset_name': np.array([4, n_steps]), ...},
                    'mean_across_datasets': np.array([4, n_steps]),
                    'std_across_datasets': np.array([4, n_steps]),
                },
                ...
            },
            'mean_across_k': {
                'per_dataset': {'dataset_name': {'mean': ..., 'std': ...}, ...},
                'mean_across_datasets': np.array([4, n_steps]),
                'std_across_datasets': np.array([4, n_steps]),
            },
            'dataset_metadata': {'dataset_name': {...}, ...},
            'summary': {
                'final_mean_loss': float,
                'best_step': int,
                'best_mean_loss': float
            }
        }
    """
    # Load parameters
    with open(os.path.join(run_path, 'parameters.json'), 'r') as f:
        parameters = json.load(f)

    sigma_sq = parameters.get("sigma_sq", 0.04)
    sigma_theta = parameters.get("sigma_theta", 0.5)
    f_param = parameters.get("f", 1)
    N_shift = parameters.get("N_shift", 15)
    Shift_std = parameters.get("Shift_std", 3)

    # Discover k directories
    checkpoints_path = os.path.join(run_path, "checkpoints")
    k_dirs = sorted([d for d in os.listdir(checkpoints_path) if d.startswith('k')])
    k_indices = [int(d[1:]) for d in k_dirs]

    # Discover checkpoint steps from first k directory
    first_k_path = os.path.join(checkpoints_path, k_dirs[0])
    prefix = "g0_best_step_" if use_best_checkpoints else "g0_step_"
    step_files = [f for f in os.listdir(first_k_path) if f.startswith(prefix) and f.endswith('.pkl')]
    steps = sorted([int(f.replace(prefix, '').replace('.pkl', '')) for f in step_files])

    # Discover validation datasets
    val_datasets = sorted([d for d in os.listdir(validation_path)
                          if os.path.isdir(os.path.join(validation_path, d))])

    # Load validation dataset metadata
    dataset_metadata = {}
    for ds_name in val_datasets:
        with open(os.path.join(validation_path, ds_name, 'metadata.json'), 'r') as f:
            dataset_metadata[ds_name] = json.load(f)

    # Note: We'll load validation batches on-demand to avoid memory issues
    print(f"Found {len(val_datasets)} validation datasets")

    # JIT compile loss functions for GPU acceleration
    sep_loss_jit = jax.jit(losses.sep_plane_KernChi_seq)
    pos_loss_jit = jax.jit(losses.pos_plane_seq)
    norm_loss_jit = jax.jit(losses.norm_plane_seq)

    # Initialize results structure
    n_steps = len(steps)
    per_k = {}

    print(f"Computing validation losses for {len(k_indices)} k values, {n_steps} steps, {len(val_datasets)} datasets...")

    for k_idx in k_indices:
        k_path = os.path.join(checkpoints_path, f"k{k_idx}")
        per_k[k_idx] = {'per_dataset': {}}

        for ds_name in val_datasets:
            per_k[k_idx]['per_dataset'][ds_name] = np.zeros((4, n_steps))

        for step_idx, step in enumerate(steps):
            print(f"    Step {step} ({step_idx + 1}/{n_steps})...", end=" ", flush=True)

            # Load checkpoint
            prefix = "best_" if use_best_checkpoints else ""
            with open(os.path.join(k_path, f'g0_{prefix}step_{step}.pkl'), 'rb') as f:
                g0 = pickle.load(f)
            with open(os.path.join(k_path, f'om_{prefix}step_{step}.pkl'), 'rb') as f:
                om = pickle.load(f)
            with open(os.path.join(k_path, f'S_{prefix}step_{step}.pkl'), 'rb') as f:
                S = pickle.load(f)

            # Convert to JAX arrays for GPU computation
            g0 = jnp.array(g0)
            om = jnp.array(om)
            S = jnp.array(S)

            # Compute loss for each validation dataset (process batch-by-batch)
            for ds_name in val_datasets:
                ds_path = os.path.join(validation_path, ds_name)
                metadata = dataset_metadata[ds_name]
                num_batches = metadata['num_batches']

                # Accumulate losses over batches
                sep_losses, pos_losses, norm_losses = [], [], []

                for batch_idx in range(num_batches):
                    with open(os.path.join(ds_path, f'batch_{batch_idx:05d}.pkl'), 'rb') as f_batch:
                        phi = pickle.load(f_batch)  # [batch_size, seq_len, 2]

                    B, L = phi.shape[0], phi.shape[1]

                    # Convert to JAX arrays for GPU computation
                    phi_jax = jnp.array(phi)
                    chi = helpers.calc_chi_plane(phi_jax, sigma_theta, f_param)

                    # Compute individual loss components using JIT-compiled functions
                    sep_losses.append(float(sep_loss_jit(g0, om, S, phi_jax, sigma_sq, chi)))
                    pos_losses.append(float(pos_loss_jit(g0, om, S, phi_jax)))

                    # Compute norm loss: create shifted versions of trajectories
                    phi_shift = np.random.normal(0, Shift_std, [N_shift, 2])
                    phi_norm = jnp.array(np.reshape(
                        phi[:, None, :, :] + phi_shift[None, :, None, :],
                        [B * N_shift, L, 2]
                    ))
                    norm_losses.append(float(norm_loss_jit(g0, om, S, phi_jax, phi_norm)))

                # Average over batches
                sep_loss = np.mean(sep_losses)
                pos_loss = np.mean(pos_losses)
                norm_loss = np.mean(norm_losses)

                # Store: [total, separation, positivity, norm]
                per_k[k_idx]['per_dataset'][ds_name][0, step_idx] = sep_loss   # Total = separation
                per_k[k_idx]['per_dataset'][ds_name][1, step_idx] = sep_loss   # Separation
                per_k[k_idx]['per_dataset'][ds_name][2, step_idx] = pos_loss   # Positivity
                per_k[k_idx]['per_dataset'][ds_name][3, step_idx] = norm_loss  # Norm

            print("done", flush=True)

        # Compute mean/std across datasets for this k
        all_losses = np.stack([per_k[k_idx]['per_dataset'][ds] for ds in val_datasets], axis=0)
        per_k[k_idx]['mean_across_datasets'] = np.mean(all_losses, axis=0)
        per_k[k_idx]['std_across_datasets'] = np.std(all_losses, axis=0)

        print(f"  k={k_idx} done")

    # Aggregate across k values
    mean_across_k = {'per_dataset': {}}

    for ds_name in val_datasets:
        ds_losses = np.stack([per_k[k]['per_dataset'][ds_name] for k in k_indices], axis=0)
        mean_across_k['per_dataset'][ds_name] = {
            'mean': np.mean(ds_losses, axis=0),
            'std': np.std(ds_losses, axis=0)
        }

    # Mean across k and datasets
    all_k_dataset_losses = np.stack([per_k[k]['mean_across_datasets'] for k in k_indices], axis=0)
    mean_across_k['mean_across_datasets'] = np.mean(all_k_dataset_losses, axis=0)
    mean_across_k['std_across_datasets'] = np.std(all_k_dataset_losses, axis=0)

    # Summary statistics
    mean_total_losses = mean_across_k['mean_across_datasets'][0, :]  # Total loss across steps
    best_step_idx = np.argmin(mean_total_losses)
    summary = {
        'final_mean_loss': float(mean_total_losses[-1]),
        'best_step': steps[best_step_idx],
        'best_mean_loss': float(mean_total_losses[best_step_idx])
    }

    print(f"Done. Best step: {summary['best_step']} with loss {summary['best_mean_loss']:.6f}")

    return {
        'steps': steps,
        'per_k': per_k,
        'mean_across_k': mean_across_k,
        'dataset_metadata': dataset_metadata,
        'summary': summary
    }