import numpy as np
import glob
import os
from os.path import join as pjoin

from rich import print
from scipy.ndimage import map_coordinates

from dipy.segment.bundles import bundle_shape_similarity
from dipy.align.streamwarp import (bundlewarp, bundlewarp_shape_analysis)
from dipy.tracking.streamline import (set_number_of_points, Streamlines,
                                      transform_streamlines)
from dipy.stats.analysis import assignment_map
from dipy.io.streamline import load_trk
from dipy.io.image import load_nifti


def evaluate_data(bundles_a_native_path, bundles_a_atlas_path,
                  bundles_b_native_path, bundles_b_atlas_path,
                  model_bundle_path, bundle_name,
                  metric_folder_a, metric_folder_b,
                  out_dir):

    # Bundle shape difference (between A & B) profile with BundleWarp
    # displacement field
    # print(':left_arrow_curving_right: Loading Bundles (model, A, B)')
    bundle_a_atlas = load_trk(bundles_a_atlas_path, reference="same",
                              bbox_valid_check=False).streamlines
    bundle_b_atlas = load_trk(bundles_b_atlas_path, reference="same",
                              bbox_valid_check=False).streamlines
    bundle_a_native = load_trk(bundles_a_native_path, reference="same",
                               bbox_valid_check=False).streamlines
    bundle_b_native = load_trk(bundles_b_native_path, reference="same",
                               bbox_valid_check=False).streamlines
    model_bundle = load_trk(model_bundle_path, reference="same",
                            bbox_valid_check=False).streamlines

    static = Streamlines(set_number_of_points(bundle_a_atlas, 20))
    moving = Streamlines(set_number_of_points(bundle_b_atlas, 20))

    # import ipdb; ipdb.set_trace()
    # TODO: check length of static and moving
    # save empty stats
    if len(static) == 0 or len(moving) == 0:
        save_empty_bundle_profiles(bundle_name, metric_folder_a, out_dir,
                                   stype='A')

        save_empty_bundle_profiles(bundle_name, metric_folder_b, out_dir,
                                   stype='B')
        np.save(pjoin(out_dir, 'shape_profile.npy'), np.zeros(10))
        np.save(pjoin(out_dir, 'shape_profile_stdv.npy'), np.zeros(10))
        np.save(pjoin(out_dir, 'shape_similarity_score.npy'), 0)
        return

    bw_results = bundlewarp(static, moving, alpha=0.001, beta=20)
    deformed_bundle, moving_aligned, distances, match_pairs, warp_map = bw_results

    shape_profile, stdv = bundlewarp_shape_analysis(
        moving_aligned, deformed_bundle, no_disks=10, plotting=False)

    np.save(pjoin(out_dir, 'shape_profile.npy'), shape_profile)
    np.save(pjoin(out_dir, 'shape_profile_stdv.npy'), stdv)

    # Bundle shape similarity score between two bundles (A & B)

    rng = np.random.RandomState()
    clust_thr = [0]
    threshold = 6  # very strict threshold

    sm_score = bundle_shape_similarity(bundle_a_atlas, bundle_b_atlas, rng, clust_thr,
                                       threshold)
    np.save(pjoin(out_dir, 'shape_similarity_score.npy'), sm_score)
    # This needs to be updated to something like following
    # np.save(pjoin(out_dir, bundle_name+'_shape_similarity_score.npy'), sm_score)

    # BUAN profiles of A & B bundles with DTI metrics

    create_buan_profiles(bundle_a_native, bundle_a_atlas, model_bundle,
                         bundle_name, metric_folder_a, out_dir, stype='A')

    create_buan_profiles(bundle_b_native, bundle_b_atlas, model_bundle,
                         bundle_name, metric_folder_b, out_dir, stype='B')


def create_buan_profiles(bundle_native, bundle_atlas, model_bundle,
                         bundle_name, metric_folder, out_dir, stype=None):

    n = 100  # number of segments along the length of the bundle
    index = assignment_map(bundle_atlas, model_bundle, n)
    index = np.array(index)

    metric_files_names_dti = glob.glob(os.path.join(metric_folder, "*.nii.gz"))

    _, affine = load_nifti(metric_files_names_dti[0])

    affine_r = np.linalg.inv(affine)
    transformed_org_bundle = transform_streamlines(bundle_native, affine_r)

    for file_name in metric_files_names_dti:

        metric, _ = load_nifti(file_name)
        if metric.ndim != 3:
            continue
        values = map_coordinates(metric, transformed_org_bundle._data.T,
                                 order=1)

        buan_mean_profile = np.zeros(n)
        buan_stdv = np.zeros(n)

        photometric_name = os.path.split(file_name)
        metric_name = photometric_name[1].replace('.nii.gz', '')

        for i in range(n):

            buan_mean_profile[i] = np.nanmean(values[index == i])
            buan_stdv[i] = np.std(values[index == i])

        np.save(pjoin(out_dir, f"{bundle_name}_{metric_name}_{stype}_buan_mean_profile.npy"), buan_mean_profile)
        np.save(pjoin(out_dir, f"{bundle_name}_{metric_name}_{stype}_buan_profile_stdv.npy"), buan_stdv)


def save_empty_bundle_profiles(bundle_name, metric_folder, out_dir, stype=None):
    metric_files_names_dti = glob.glob(os.path.join(metric_folder, "*.nii.gz"))
    for file_name in metric_files_names_dti:

        metric, _ = load_nifti(file_name)
        if metric.ndim != 3:
            continue

        photometric_name = os.path.split(file_name)
        metric_name = photometric_name[1].replace('.nii.gz', '')

        n = 100  # number of segments along the length of the bundle

        buan_mean_profile = np.zeros(n)
        buan_stdv = np.zeros(n)

        np.save(pjoin(out_dir, f"{bundle_name}_{metric_name}_{stype}_buan_mean_profile.npy"), buan_mean_profile)
        np.save(pjoin(out_dir, f"{bundle_name}_{metric_name}_{stype}_buan_profile_stdv.npy"), buan_stdv)
