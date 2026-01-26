import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp

from scores import GridScorer
import NRT_functions.helper_functions as hf

def compute_g_at_positions(g0, om, S, phi):
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
    T = hf.get_T_2D(om, phi, S)
    
    # Apply transformation
    g = jnp.einsum('nij,j->in', T, g0_processed)

    norms = jnp.linalg.norm(g, axis=1, keepdims=True)
    g = g / norms
    
    return g

def quantitative_analysis_seq(g0, S, om, parameters, counter, res=70, savepath=None):
    phi_small = np.linspace(-np.pi, np.pi, res)/6
    phi_small = np.meshgrid(phi_small, phi_small)
    phi_small = np.hstack([np.ndarray.flatten(phi_small[0])[:,None],
                                np.ndarray.flatten(phi_small[1])[:,None]])

    phi_medium = np.linspace(-np.pi, np.pi, res)/2
    phi_medium = np.meshgrid(phi_medium, phi_medium)
    phi_medium = np.hstack([np.ndarray.flatten(phi_medium[0])[:,None], 
                           np.ndarray.flatten(phi_medium[1])[:,None]])
    
    phi_large = np.linspace(-np.pi, np.pi, res)*parameters.get("pos_lengthscale", 2)
    phi_large = np.meshgrid(phi_large, phi_large)
    phi_large = np.hstack([np.ndarray.flatten(phi_large[0])[:,None], 
                           np.ndarray.flatten(phi_large[1])[:,None]])
    
    V_small = np.array(compute_g_at_positions(g0, om, S, phi_small))
    V_medium = np.array(compute_g_at_positions(g0, om, S, phi_medium))
    V_large = np.array(compute_g_at_positions(g0, om, S, phi_large))

    maps_large = [V_large[i,:] for i in range(V_large.shape[0])]
    maps_medium = [V_medium[i,:] for i in range(V_medium.shape[0])]
    maps_small = [V_small[i,:] for i in range(V_small.shape[0])]

    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    box_width=2*np.pi*parameters.get("pos_lengthscale", 2)
    box_height=2*np.pi*parameters.get("pos_lengthscale", 2)
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(res, coord_range, masks_parameters)
    scores = {}

    fig_score, axes = plt.subplots(1, 3, figsize=(12, 5))

    score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
      *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(maps_small)])
    score_60 = np.nan_to_num(score_60)
    score_90 = np.nan_to_num(score_90)
    scores["sm_60"] = score_60
    scores["sm_90"] = score_90

    axes[0].hist(score_60, range=(-1,2.5), bins=15)
    axes[0].set_title('Small Ratemaps')
    axes[0].set_xlabel('Grid score')
    axes[0].set_ylabel('Count')

    # Add max and mean statistics for small ratemaps
    max_score_small = np.max(score_60)
    mean_score_small = np.mean(score_60)
    scores["sm_60_max"] = max_score_small
    scores["sm_60_mean"] = mean_score_small
    scores["sm_90_max"] = np.max(score_90)
    scores["sm_90_mean"] = np.mean(score_90)
    axes[0].text(0.05, 0.95, f'Max: {max_score_small:.3f}\nMean: {mean_score_small:.3f}',
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
      *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(maps_medium)])
    score_60 = np.nan_to_num(score_60)
    score_90 = np.nan_to_num(score_90)
    scores["md_60"] = score_60
    scores["md_90"] = score_90

    axes[1].hist(score_60, range=(-1,2.5), bins=15)
    axes[1].set_title('Medium Ratemaps')
    axes[1].set_xlabel('Grid score')
    axes[1].set_ylabel('Count')

    # Add max and mean statistics for small ratemaps
    max_score_medium = np.max(score_60)
    mean_score_medium = np.mean(score_60)
    scores["md_60_max"] = max_score_medium
    scores["md_60_mean"] = mean_score_medium
    scores["md_90_max"] = np.max(score_90)
    scores["md_90_mean"] = np.mean(score_90)
    axes[1].text(0.05, 0.95, f'Max: {max_score_medium:.3f}\nMean: {mean_score_medium:.3f}',
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
      *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(maps_large)])
    score_60 = np.nan_to_num(score_60)
    score_90 = np.nan_to_num(score_90)
    scores["lg_60"] = score_60
    scores["lg_90"] = score_90

    axes[2].hist(score_60, range=(-1,2.5), bins=15)
    axes[2].set_title('Large Ratemaps')
    axes[2].set_xlabel('Grid score')
    axes[2].set_ylabel('Count')

    # Add max and mean statistics for large ratemaps
    max_score_large = np.max(score_60)
    mean_score_large = np.mean(score_60)
    scores["lg_60_max"] = max_score_large
    scores["lg_60_mean"] = mean_score_large
    scores["lg_90_max"] = np.max(score_90)
    scores["lg_90_mean"] = np.mean(score_90)
    axes[2].text(0.05, 0.95, f'Max: {max_score_large:.3f}\nMean: {mean_score_large:.3f}',
                 transform=axes[2].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig_score.tight_layout()
    if savepath:
      fig_score.savefig(os.path.join(savepath, f"grid_scores_{counter}.png"))
      hf.save_parameters_json(scores, f"grid_scores_{counter}", savepath)

    return fig_score, scores
