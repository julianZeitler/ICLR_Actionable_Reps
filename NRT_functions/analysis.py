import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from scores import GridScorer
from sweep_analysis_sequential import compute_g_at_positions

def quantitative_analysis_seq(g0, S, om, parameters, counter, res=70, savepath=None):
    phi_small = np.linspace(-np.pi, np.pi, res)/2
    phi_small = np.meshgrid(phi_small, phi_small)
    phi_small = np.hstack([np.ndarray.flatten(phi_small[0])[:,None], 
                                 np.ndarray.flatten(phi_small[1])[:,None]])
    
    phi_large = np.linspace(-np.pi, np.pi, res)*parameters.get("pos_lengthscale", 2)
    phi_large = np.meshgrid(phi_large, phi_large)
    phi_large = np.hstack([np.ndarray.flatten(phi_large[0])[:,None], 
                                 np.ndarray.flatten(phi_large[1])[:,None]])
    
    V_small = np.array(compute_g_at_positions(g0, om, S, phi_small))
    V_large = np.array(compute_g_at_positions(g0, om, S, phi_large))

    maps_large = [V_large[i,:] for i in range(V_large.shape[0])]
    maps_small = [V_small[i,:] for i in range(V_small.shape[0])]

    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    box_width=2*np.pi*parameters.get("pos_lengthscale", 2)
    box_height=2*np.pi*parameters.get("pos_lengthscale", 2)
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(res, coord_range, masks_parameters)

    score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
      *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(maps_small)])
    
    fig_score, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(score_60, range=(-1,2.5), bins=15)
    axes[0].set_title('Small Ratemaps')
    axes[0].set_xlabel('Grid score')
    axes[0].set_ylabel('Count')

    score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
      *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(maps_large)])

    axes[1].hist(score_60, range=(-1,2.5), bins=15)
    axes[1].set_title('Large Ratemaps')
    axes[1].set_xlabel('Grid score')
    axes[1].set_ylabel('Count')

    fig_score.tight_layout()
    if savepath:
        fig_score.savefig(os.path.join(savepath, f"grid_scores_{counter}.png"))

    return fig_score
