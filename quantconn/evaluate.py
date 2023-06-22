import numpy as np
from os.path import join as pjoin
from dipy.segment.bundles import bundle_shape_similarity
from dipy.align.streamwarp import (bundlewarp, bundlewarp_shape_analysis)
from dipy.tracking.streamline import (set_number_of_points, Streamlines)
from dipy.segment.bundles import bundle_shape_similarity
from dipy.stats.analysis import assignment_map

def evaluate_data(bundle1, bundle2):

    static = Streamlines(set_number_of_points(bundle1, 20))
    moving = Streamlines(set_number_of_points(bundle2, 20))
    
    deformed_bundle2, moving_aligned, distances, match_pairs, warp_map = bundlewarp(static, moving,
                                                                                    alpha=0.001, beta=20)
    
    shape_profile, stdv = bundlewarp_shape_analysis(moving_aligned, deformed_bundle, no_disks=10,
                                     plotting=True)
    
    np.save(pjoin(out_dir, 'shape_profile.npy'), shape_profile)
    np.save(pjoin(out_dir, 'shape_profile_stdv.npy'), stdv)
    
    rng = np.random.RandomState()
    clust_thr = [0]
    threshold = 2 # very strict threshold
    
    sm_score = bundle_shape_similarity(bundle1, bundle2, rng, clust_thr, threshold)
    np.save(pjoin(out_dir, 'shape_similarity_score.npy'), stdv)
    
    # write here BUAN profiles code
