import numpy as np
from os.path import join as pjoin
from dipy.segment.bundles import bundle_shape_similarity
from dipy.align.streamwarp import (bundlewarp, bundlewarp_shape_analysis)
from dipy.tracking.streamline import (set_number_of_points, Streamlines)
from dipy.segment.bundles import bundle_shape_similarity
from dipy.stats.analysis import assignment_map
from dipy.io.streamline import load_trk


def evaluate_data(bundles_A, bundles_B, model_bundle, metric_folder, out_dir):

    bundle1 = load_trk(bundles_A[0], reference="same",
                       bbox_valid_check=False).streamlines
    bundle2 = load_trk(bundles_B[0], reference="same",
                       bbox_valid_check=False).streamlines

    static = Streamlines(set_number_of_points(bundle1, 20))
    moving = Streamlines(set_number_of_points(bundle2, 20))

    deformed_bundle, moving_aligned, distances, match_pairs, warp_map = bundlewarp(static, moving,
                                                                                    alpha=0.001, beta=20)

    shape_profile, stdv = bundlewarp_shape_analysis(
        moving_aligned, deformed_bundle, no_disks=10, plotting=True)

    np.save(pjoin(out_dir, 'shape_profile.npy'), shape_profile)
    np.save(pjoin(out_dir, 'shape_profile_stdv.npy'), stdv)

    rng = np.random.RandomState()
    clust_thr = [0]
    threshold = 2 # very strict threshold

    sm_score = bundle_shape_similarity(bundle1, bundle2, rng, clust_thr,
                                       threshold)
    np.save(pjoin(out_dir, 'shape_similarity_score.npy'), stdv)

    # BUAN profiles code
    
    n=100 # number of segments along the length of the bundle
    
    for k in range(2):

        if k==0:
            folder = 'A'
        elif k=='1':
            folder = 'B'

        bundle = load_trk(bundles_+folder[0], reference="same",
                       bbox_valid_check=False).streamlines
        org_bundle = load_trk(bundles_folder[1], reference="same",
                       bbox_valid_check=False).streamlines
        
        indx = assignment_map(bundle, mobel_bundle, n)
        ind = np.array(indx)
        
        metric_files_names_dti = glob(os.path.join(metric_folder,
                                                   "*.nii.gz"))
        
        _, affine = load_nifti(metric_files_names_dti[0])
        
        affine_r = np.linalg.inv(affine)
        transformed_org_bundle = transform_streamlines(org_bundle,
                                                         affine_r)
        
        bm = os.path.split(model_bundle)[1][:-4]
        
        for file_name in metric_files_names_dti:
            
            metric, _ = load_nifti(file_name)
            values = map_coordinates(metric, transformed_org_bundle._data.T, order=1)
            
            buan_mean_profile = np.zeros(n)
            buan_stdv = np.zeros(n)
        
            ab = os.path.split(file_name)
            metric_name = ab[1]
        
            fm = metric_name[:-7]
            
            for i in range(n):
                       
                buan_mean_profile[i] = np.mean(offsets[indx == i])
                buan_stdv[i] = np.std(offsets[indx == i])
                
            np.save(pjoin(out_dir, bm+"_"+fm+'_'+folder+'_buan_mean_profile.npy'), buan_mean_profile)
            np.save(pjoin(out_dir, bm+"_"+fm+'_'+folder+'_buan_profile_stdv.npy'), buan_stdv)
