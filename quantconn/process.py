import os
from os.path import join as pjoin

import numpy as np
import matplotlib.pyplot as plt
from rich import print

from HD_BET.run import run_hd_bet

import nibabel as nib
from nibabel.streamlines.trk import TrkFile
from dipy.align import affine_registration
from dipy.align.streamlinear import whole_brain_slr
from dipy.align.reslice import reslice
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.shm import CsaOdfModel
from dipy.io.utils import nifti1_symmat, create_tractogram_header
from dipy.reconst.dti import (TensorModel, color_fa, fractional_anisotropy,
                              geodesic_anisotropy, mean_diffusivity,
                              axial_diffusivity, radial_diffusivity,
                              lower_triangular)
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk, load_trk
from dipy.tracking.metrics import length
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.tracking.streamlinespeed import length
from dipy.segment.mask import median_otsu
from dipy.segment.bundles import RecoBundles

from dipy.reconst.shm import normalize_data, sph_harm_lookup, smooth_pinv
from dipy.core.sphere import HemiSphere
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from dipy.reconst.shm import anisotropic_power

from quantconn.download import get_30_bundles_atlas_hcp842


def signal_powermap(data, gtab, sh_order=8, smooth=0.0):
    gtab2 = gradient_table_from_bvals_bvecs(gtab.bvals[np.where(1-gtab.b0s_mask)[0]], gtab.bvecs[np.where(1-gtab.b0s_mask)[0]])
    normed_data = normalize_data(data, gtab.b0s_mask)
    normed_data = normed_data[..., np.where(1-gtab.b0s_mask)[0]]

    signal_native_pts = HemiSphere(xyz=gtab2.bvecs)
    sph_harm_basis = sph_harm_lookup.get(None)

    Ba, m, n = sph_harm_basis(sh_order, signal_native_pts.theta,
                              signal_native_pts.phi)
    L = -n * (n + 1)
    invB = smooth_pinv(Ba, np.sqrt(smooth) * L)

    # fit SH basis to DWI signal
    normed_data_sh = np.dot(normed_data, invB.T)
    ap_map_signal = anisotropic_power(normed_data_sh)

    return ap_map_signal


def process_data(nifti_fname, bval_fname, bvec_fname, t1_fname, output_path,
                 t1_labels_fname=None, group='B'):
    dwi_data, dwi_affine, dwi_img = load_nifti(nifti_fname, return_img=True)
    dwi_bvals, dwi_bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
    gtab = gradient_table(dwi_bvals, dwi_bvecs)

    print(':left_arrow_curving_right: Sampling/reslicing data')
    vox_sz = dwi_img.header.get_zooms()[:3]

    # Dynamic resampling
    vox_factor = 0.14
    voxsize_sorted = sorted(vox_sz)
    max_vox_size, smax_vox_size = voxsize_sorted[-1], voxsize_sorted[-2]
    new_vox_size = [smax_vox_size + (max_vox_size - smax_vox_size) * vox_factor ] * 3
    # TODO: Check reslice order. Try with 2 and compare data (trilinear vs cubic)
    resliced_data, resliced_affine = reslice(dwi_data, dwi_affine, vox_sz,
                                             new_vox_size, order=2)

    save_nifti(pjoin(output_path, 'resliced_data.nii.gz'),
               resliced_data, resliced_affine)

    print(':left_arrow_curving_right: Building mask')
    maskdata, mask = median_otsu(
        resliced_data, vol_idx=np.where(gtab.b0s_mask)[0][:2])

    # Power map
    powermap_data = signal_powermap(maskdata, gtab)
    save_nifti(pjoin(output_path, 'powermap_data.nii.gz'),
               powermap_data, resliced_affine)

    print(':left_arrow_curving_right: Computing DTI metrics')
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    tensor_vals = lower_triangular(tenfit.quadratic_form)
    ten_img = nifti1_symmat(tensor_vals, affine=resliced_affine)

    # Flipping -> xx xy xz yy yz zz -> xx xy yy xz yx zz
    save_nifti(pjoin(output_path, 'tensors.nii.gz'),
               ten_img.get_fdata().squeeze(), resliced_affine)
    fsl_order = [0, 1, 3, 2, 4, 5]
    save_nifti(pjoin(output_path, 'tensors_fsl.nii.gz'),
               ten_img.get_fdata()[..., fsl_order].squeeze(),
               resliced_affine)

    save_nifti(pjoin(output_path, 'fa.nii.gz'), FA.astype(np.float32),
               resliced_affine)

    GA = geodesic_anisotropy(tenfit.evals)
    save_nifti(pjoin(output_path, 'ga.nii.gz'), GA.astype(np.float32),
               resliced_affine)

    RGB = color_fa(FA, tenfit.evecs)
    save_nifti(pjoin(output_path, 'rgb.nii.gz'), np.array(255 * RGB, 'uint8'),
               resliced_affine)

    MD = mean_diffusivity(tenfit.evals)
    save_nifti(pjoin(output_path, 'md.nii.gz'), MD.astype(np.float32),
               resliced_affine)

    AD = axial_diffusivity(tenfit.evals)
    save_nifti(pjoin(output_path, 'ad.nii.gz'), AD.astype(np.float32),
               resliced_affine)

    RD = radial_diffusivity(tenfit.evals)
    save_nifti(pjoin(output_path, 'rd.nii.gz'), RD.astype(np.float32),
               resliced_affine)

    # TODO:  Get White matter mask
    # download The-HCP-MMP1.0-atlas
    # register T1 whith this atlas
    # visualize white matter index in this atlas
    # get white matter mask

    # temporary solution
    white_matter = FA > 0.1
    print(':left_arrow_curving_right: Reconstruction using CSA Model')
    csa_model = CsaOdfModel(gtab, sh_order=6)
    csa_peaks = peaks_from_model(csa_model, maskdata, default_sphere,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45,
                                 mask=white_matter)

    print(':left_arrow_curving_right: Whole Brain Tractography')
    stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .15)

    seeds = utils.seeds_from_mask(white_matter, resliced_affine,
                                  density=[2, 2, 2])

    streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                          affine=resliced_affine, step_size=.5)

    # remove short streamlines
    print(':left_arrow_curving_right: Remove short streamlines (< 30mm)')
    streamlines_generator = (s for s in streamlines_generator
                             if length(s) > 30.0)
    target_streamlines = Streamlines(streamlines_generator)

    print(':left_arrow_curving_right: Saving tractogram')
    header = create_tractogram_header(TrkFile, resliced_affine,
                                      maskdata.shape[:3],
                                      new_vox_size,
                                      ''.join(nib.aff2axcodes(resliced_affine)))
    target_sft = StatefulTractogram(target_streamlines, header, Space.RASMM)
    save_trk(target_sft, pjoin(output_path, "full_tractogram.trk"),
             bbox_valid_check=False)

    # import ipdb; ipdb.set_trace()
    #     from dipy.viz.horizon.app import horizon
    #     horizon(tractograms=[target_streamlines_in_t1_sft],
    #             images=[(label_data, label_affine)], interactive=True,
    #             cluster=True, world_coords=True)
    # Recobunble

    # Step 1: Register target tractogram to model atlas space using SLR
    print(':left_arrow_curving_right: Register target tractogram to model atlas')
    atlas_file, all_bundles_files = get_30_bundles_atlas_hcp842()
    sft_atlas = load_trk(atlas_file, "same", bbox_valid_check=True)
    atlas_header = create_tractogram_header(atlas_file,
                                            *sft_atlas.space_attributes)

    slr_result = whole_brain_slr(sft_atlas.streamlines, target_streamlines,
                                 x0='affine', verbose=True, progressive=True,
                                 rng=np.random.RandomState(1983))

    moved, transform, qb_centroids1, qb_centroids2 = slr_result
    moved_sft = StatefulTractogram(moved, atlas_header, Space.RASMM)

    np.save(pjoin(output_path, "slr_transform.npy"), transform)
    save_trk(moved_sft, pjoin(output_path, "full_tractogram_moved.trk"),
             bbox_valid_check=False)

    # Step 2: Recognize bundles in the target tractogram
    print(':left_arrow_curving_right: Detecting bundles (AF_R, AF_L, CST_L, CST_R, OR_L, OR_R) in the target tractogram')
    rb = RecoBundles(moved, verbose=True, rng=np.random.RandomState(2023))

    selected_bundles = ['AF_R', 'AF_L', 'CST_L', 'CST_R', 'OR_L', 'OR_R']
    for bundle_name in selected_bundles:
        model_bundle_path = all_bundles_files.get(bundle_name)
        if not model_bundle_path:
            print(f"Bundle {bundle_name} not found in the atlas")
            continue
        model_bundle = load_trk(model_bundle_path, "same")

        # TODO: Check if those parameters are good for all bundles
        # we might need to personalize them for each bundle
        # attempts = [(15, 7), (20, 10), (25, 12)]
        # for reduction_thr, pruning_thr in attempts:
        recognized_bundle, model_labels = rb.recognize(
            model_bundle=model_bundle.streamlines, model_clust_thr=0.1,
            reduction_thr=18, pruning_thr=8,
            reduction_distance='mdf', pruning_distance='mdf', slr=True)

            # if len(recognized_bundle):
            #     break

        reco = StatefulTractogram(recognized_bundle, atlas_header,
                                  Space.RASMM)
        save_trk(reco, pjoin(output_path, f"{bundle_name}_in_atlas_space.trk"),
                 bbox_valid_check=False)
        reco = StatefulTractogram(target_streamlines[model_labels],
                                  header, Space.RASMM)
        save_trk(reco, pjoin(output_path, f"{bundle_name}_in_orig_space.trk"),
                 bbox_valid_check=False)

    ##########################################################################
    #
    # Connectivity matrix
    #
    ##########################################################################

    if not t1_labels_fname:
        print(':left_arrow_curving_right: No T1 labels file provided, Skipping connectivity matrix')
        return

    print(':left_arrow_curving_right: Connectivity matrix: Loading data')
    t1_data, t1_affine, t1_img = load_nifti(t1_fname, return_img=True)
    label_data, label_affine, label_voxsize = load_nifti(t1_labels_fname,
                                                         return_voxsize=True)

    t1_skullstrip_fname = pjoin(os.path.dirname(__file__), 'data', 'brains',
                                os.path.basename(t1_fname))

    if not os.path.isfile(t1_skullstrip_fname):
        print(':left_arrow_curving_right: Connectivity matrix: T1 skullstripping')
        use_hd_bet = True
        t1_skullstrip_fname = pjoin(
                output_path,
                os.path.basename(t1_fname).replace('.nii.gz', 'skullstrip.nii.gz')
        )
        if use_hd_bet:
            run_hd_bet(t1_fname, t1_skullstrip_fname, mode='fast',
                       device='cpu', do_tta=False)
        else:
            from dipy.nn.evac import EVACPlus
            evac = EVACPlus()
            mask_volume = evac.predict(t1_data, t1_affine,
                                       t1_img.header.get_zooms()[:3])
            masked_volume = mask_volume * t1_data
            save_nifti(t1_skullstrip_fname, masked_volume, t1_affine)

    t1_noskull_data, t1_noskull_affine, t1_noskull_img = \
        load_nifti(t1_skullstrip_fname, return_img=True)

    t1_vox_sz = t1_noskull_img.header.get_zooms()[:3]
    t1_noskull_resliced_data,  \
        t1_noskull_resliced_affine = reslice(t1_noskull_data,
                                             t1_noskull_affine,
                                             t1_vox_sz, new_vox_size, order=2)

    save_nifti(pjoin(output_path, 't1_noskull_resliced.nii.gz'),
               t1_noskull_resliced_data, t1_noskull_resliced_affine)

    # test_image_registration(maskdata, gtab, t1_noskull_data,
    #                         t1_noskull_resliced_data, powermap_data,
    #                         resliced_affine, t1_noskull_affine,
    #                         t1_noskull_resliced_affine,
    #                         output_path)

    print(':left_arrow_curving_right: Connectivity matrix: Registering DWI B0s to T1 / labels')
    pipeline = ["center_of_mass", "translation", "rigid", "rigid_isoscaling", "rigid_scaling"]
    # Take one B0 instead of all of them or correct motion.
    mean_b0 = np.mean(maskdata[..., gtab.b0s_mask], -1)
    warped_b0, warped_b0_affine = affine_registration(
        mean_b0, t1_noskull_resliced_data, moving_affine=resliced_affine,
        static_affine=t1_noskull_resliced_affine, pipeline=pipeline)

    save_nifti(pjoin(output_path, "warped_b0_resliced.nii.gz"),
               warped_b0, t1_noskull_resliced_affine)

    print(':left_arrow_curving_right: Connectivity matrix: Transforming Streamlines')
    target_streamlines_in_t1 = transform_streamlines(
        target_streamlines, np.linalg.inv(warped_b0_affine))  # in_place=True)

    t1_noskull_resliced_data, t1_noskull_resliced_affine, t1_noskull_resliced_img =  \
        load_nifti(pjoin(output_path, 't1_noskull_resliced.nii.gz'), return_img=True)

    target_streamlines_in_resliced_t1_sft = StatefulTractogram(
        target_streamlines_in_t1, t1_noskull_resliced_img, Space.RASMM)

    save_trk(target_streamlines_in_resliced_t1_sft,
             pjoin(output_path, "full_tractogram_in_resliced_t1.trk"),
             bbox_valid_check=False)

    target_streamlines_in_t1_sft = StatefulTractogram(target_streamlines_in_t1,
                                                      t1_img, Space.RASMM)

    save_trk(target_streamlines_in_t1_sft,
             pjoin(output_path, "full_tractogram_in_t1.trk"),
             bbox_valid_check=False)

    # interactive = True
    # if interactive:
    #     from dipy.viz.horizon.app import horizon
    #     horizon(tractograms=[target_streamlines_in_t1_sft],
    #             images=[(label_data, label_affine)], interactive=True,
    #             cluster=True, world_coords=True)

    print(':left_arrow_curving_right: Connectivity matrix')
    # # Connectivity matrix
    M = utils.connectivity_matrix(
        target_streamlines_in_t1, t1_noskull_affine,
        label_data.astype(np.uint8))

    # Normalize it by dividing by the length of the streamlines

    np.save(pjoin(output_path, 'connectivity_matrice.npy'), M)
    # if interactive:
    #     plt.imshow(np.log1p(M), interpolation='nearest')
    #     plt.savefig(pjoin(output_path, "connectivity.png"))


def test_image_registration(maskdata, gtab, t1_noskull_data,
                            t1_noskull_resliced_data, powermap_data,
                            resliced_affine, t1_noskull_affine,
                            t1_noskull_resliced_affine,
                            output_path):

    pipeline_1 = ["center_of_mass", "translation", "rigid"]
    pipeline_2 = ["rigid_isoscaling"]
    pipeline_3 = ["rigid_isoscaling", "rigid"]
    pipeline_4 = ["center_of_mass", "rigid_isoscaling"]
    pipeline_5 = pipeline_1 + ["rigid_isoscaling"]
    pipeline_6 = pipeline_5 + ["rigid_scaling"]
    all_pipelines = [pipeline_1, pipeline_2, pipeline_3, pipeline_4,
                     pipeline_5, pipeline_6]
    print(':left_arrow_curving_right: Connectivity matrix: Registering DWI B0s to T1 / labels')
    for p_idx, pipeline in enumerate(all_pipelines, start=1):
        # Take one B0 instead of all of them or correct motion.
        mean_b0 = np.mean(maskdata[..., gtab.b0s_mask], -1)
        warped_b0, warped_b0_affine = affine_registration(
            mean_b0, t1_noskull_data, moving_affine=resliced_affine,
            static_affine=t1_noskull_affine, pipeline=pipeline)

        warped_b0_iso, warped_b0_iso_affine = affine_registration(
            mean_b0, t1_noskull_resliced_data, moving_affine=resliced_affine,
            static_affine=t1_noskull_resliced_affine, pipeline=pipeline)

        save_nifti(pjoin(output_path, f"warped_b0_{p_idx}.nii.gz"), warped_b0,
                   t1_noskull_affine)
        save_nifti(pjoin(output_path, f"warped_b0_{p_idx}_resliced.nii.gz"),
                   warped_b0_iso, t1_noskull_resliced_affine)

        # we use the powermap instead of the mean b0

        warped_pm, warped_pm_affine = affine_registration(
            powermap_data, t1_noskull_data, moving_affine=resliced_affine,
            static_affine=t1_noskull_affine, pipeline=pipeline)
        warped_pm_iso, warped_pm_iso_affine = affine_registration(
            powermap_data, t1_noskull_resliced_data,
            moving_affine=resliced_affine,
            static_affine=t1_noskull_resliced_affine, pipeline=pipeline)

        save_nifti(pjoin(output_path, f"warped_pm_{p_idx}.nii.gz"),
                   warped_pm, t1_noskull_affine)
        save_nifti(pjoin(output_path, f"warped_pm_{p_idx}_resliced.nii.gz"),
                   warped_pm_iso, t1_noskull_resliced_affine)


# Process 1: Bundles/Tractography/BUAN
# Input: 25 subjects, each with 2 DWI and corresponding bvecs/bvals (total of 50 NIFTI images).
# Output:
# Six major white matter bundles (left/right Arcuate Fasciculus, left/right Optic Radiations, left/right Corticospinal tract)
# Your favorite bundle similarity metric (BUAN)
# FA map
# MD map
# Whole brain connectome weighted by number of streamlines and using HCP’s multi-modal parcellation, version 1.0 (HCP_MMP1.0).
# DTI
# Process 2: Evaluation
# Input: Bundles, Connectome, FA/MD maps
# Output:
# Bundle metrics describing shape (length, volume)
# Bundle tractometry (FA/MD along bundle or average FA/MD in b


# How do we evaluate connectome ?

# 78 Subjects
# Processing: 9h 36m 42s
# Evaluation: 2h 25m 58s
# Total: 11h 2m 40s
