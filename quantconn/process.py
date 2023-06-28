from os.path import join as pjoin

import numpy as np
from rich import print

from dipy.align.streamlinear import whole_brain_slr
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
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.segment.mask import median_otsu
from dipy.segment.bundles import RecoBundles


from quantconn.download import get_30_bundles_atlas_hcp842


def process_data(nifti_fname, bval_fname, bvec_fname, t1_fname, output_path):
    data, affine, data_img = load_nifti(nifti_fname, return_img=True)
    bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    print(':left_arrow_curving_right: Building mask')
    maskdata, mask = median_otsu(data, median_radius=3,
                                 vol_idx=np.where(gtab.b0s_mask)[0],
                                 numpass=1, autocrop=True, dilate=2)

    print(':left_arrow_curving_right: Computing DTI metrics')
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    tensor_vals = lower_triangular(tenfit.quadratic_form)
    ten_img = nifti1_symmat(tensor_vals, affine=affine)

    save_nifti(pjoin(output_path, 'tensors.nii.gz'), ten_img.get_fdata(),
               affine)
    save_nifti(pjoin(output_path, 'fa.nii.gz'), FA.astype(np.float32), affine)

    GA = geodesic_anisotropy(tenfit.evals)
    save_nifti(pjoin(output_path, 'ga.nii.gz'), GA.astype(np.float32), affine)

    RGB = color_fa(FA, tenfit.evecs)
    save_nifti(pjoin(output_path, 'rgb.nii.gz'), np.array(255 * RGB, 'uint8'),
               affine)

    MD = mean_diffusivity(tenfit.evals)
    save_nifti(pjoin(output_path, 'md.nii.gz'), MD.astype(np.float32), affine)

    AD = axial_diffusivity(tenfit.evals)
    save_nifti(pjoin(output_path, 'ad.nii.gz'), AD.astype(np.float32), affine)

    RD = radial_diffusivity(tenfit.evals)
    save_nifti(pjoin(output_path, 'rd.nii.gz'), RD.astype(np.float32), affine)

    # TODO:  Get White matter mask
    # download The-HCP-MMP1.0-atlas
    # register T1 whith this atlas
    # visualize white matter index in this atlas
    # get white matter mask

    # temporary solution
    white_matter = FA > 0.2
    print(':left_arrow_curving_right: Reconstruction using CSA Model')
    csa_model = CsaOdfModel(gtab, sh_order=6)
    csa_peaks = peaks_from_model(csa_model, maskdata, default_sphere,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45,
                                 mask=white_matter)

    print(':left_arrow_curving_right: Whole Brain Tractography')
    stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .25)

    seeds = utils.seeds_from_mask(white_matter, affine, density=[2, 2, 2])

    streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                          affine=affine, step_size=.5)
    target_streamlines = Streamlines(streamlines_generator)

    target_sft = StatefulTractogram(target_streamlines, data_img, Space.RASMM)
    save_trk(target_sft, pjoin(output_path, "full_tractogram.trk"))

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
            reduction_thr=20, pruning_thr=10,
            reduction_distance='mdf', pruning_distance='mdf', slr=True)

            # if len(recognized_bundle):
            #     break

        reco = StatefulTractogram(recognized_bundle, atlas_header,
                                  Space.RASMM)
        save_trk(reco, pjoin(output_path, f"{bundle_name}_in_atlas_space.trk"),
                 bbox_valid_check=False)
        reco = StatefulTractogram(target_streamlines[model_labels],
                                  data_img, Space.RASMM)
        save_trk(reco, pjoin(output_path, f"{bundle_name}_in_orig_space.trk"),
                 bbox_valid_check=False)

    # Connectivity matrix




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