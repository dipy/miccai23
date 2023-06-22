from os.path import join as pjoin

import numpy as np

from dipy.align.streamlinear import whole_brain_slr
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.shm import CsaOdfModel
from dipy.io.utils import nifti1_symmat
from dipy.reconst.dti import (TensorModel, color_fa, fractional_anisotropy,
                              geodesic_anisotropy, mean_diffusivity,
                              axial_diffusivity, radial_diffusivity,
                              lower_triangular, mode as get_mode)
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.segment.mask import median_otsu
from dipy.segment.bundles import RecoBundles


# Process 1: Bundles/Tractography/BUAN
# Input: 25 subjects, each with 2 DWI and corresponding bvecs/bvals (total of 50 NIFTI images).
# Output:
# Six major white matter bundles (left/right Arcuate Fasciculus, left/right Optic Radiations, left/right Corticospinal tract)
# Your favorite bundle similarity metric (BUAN)
# FA map
# MD map
# Whole brain connectome weighted by number of streamlines and using HCP’s multi-modal parcellation, version 1.0 (HCP_MMP1.0).
# DTI


def process_data(nifti_fname, bval_fname, bvec_fname, t1_fname, output_path):
    data, affine, data_img = load_nifti(nifti_fname, return_img=True)
    bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    maskdata, mask = median_otsu(data, median_radius=3,
                                 vol_idx=np.where(gtab.b0s_mask)[0],
                                 numpass=1, autocrop=True, dilate=2)
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
    # download The-HCP-MMP1.0-atlas-in-FSL https://github.com/mbedini/The-HCP-MMP1.0-atlas-in-FSL
    # https://github.com/mbedini/The-HCP-MMP1.0-atlas-in-FSL/blob/master/MNI_Glasser_HCP_v1.0.nii.gz
    # create a Get function for the downloaded atlas
    # register T1 whith this atlas
    # visualize white matter index in this atlas
    # get white matter mask

    # temporary solution
    white_matter = FA > 0.2

    csa_model = CsaOdfModel(gtab, sh_order=6)
    csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45,
                                 mask=white_matter)

    stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .25)

    seeds = utils.seeds_from_mask(white_matter, affine, density=[2, 2, 2])

    streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                          affine=affine, step_size=.5)
    streamlines = Streamlines(streamlines_generator)

    sft = StatefulTractogram(streamlines, data_img, Space.RASMM)
    save_trk(sft, pjoin(output_path, "full_tractogram.trk"), streamlines)

    # recobunble
    # moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
    # atlas, target, x0='affine', verbose=True, progressive=True,
    # rng=np.random.RandomState(1984))

    # np.save("slr_transform.npy", transform)

    # rb = RecoBundles(moved, verbose=True, rng=np.random.RandomState(2001))

    # recognized_af_l, af_l_labels = rb.recognize(model_bundle=model_af_l,
    #                                             model_clust_thr=0.1,
    #                                             reduction_thr=15,
    #                                             pruning_thr=7,
    #                                             reduction_distance='mdf',
    #                                             pruning_distance='mdf',
    #                                             slr=True)
    # save the recognize bundle



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