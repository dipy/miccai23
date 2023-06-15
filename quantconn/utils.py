import numpy as np

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


def process_data(nifti_fname, bval_fname, bvec_fname):
    data, affine, hardi_img = load_nifti(nifti_fname, return_img=True)
    bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    maskdata, mask = median_otsu(data, median_radius=3,
                                 numpass=1, autocrop=True, dilate=2)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    tensor_vals = lower_triangular(tenfit.quadratic_form)
    ten_img = nifti1_symmat(tensor_vals, affine=affine)
    save_nifti(ten_img, 'tensors.nii.gz')

    save_nifti('fa.nii.gz', FA.astype(np.float32), affine)

    GA = geodesic_anisotropy(tenfit.evals)
    save_nifti('ga.nii.gz', GA.astype(np.float32), affine)

    RGB = color_fa(FA, tenfit.evecs)
    save_nifti('rgb.nii.gz', np.array(255 * RGB, 'uint8'), affine)

    MD = mean_diffusivity(tenfit.evals)
    save_nifti('md.nii.gz', MD.astype(np.float32), affine)

    AD = axial_diffusivity(tenfit.evals)
    save_nifti('ad.nii.gz', AD.astype(np.float32), affine)

    RD = radial_diffusivity(tenfit.evals)
    save_nifti('rd.nii.gz', RD.astype(np.float32), affine)

    csa_model = CsaOdfModel(gtab, sh_order=6)
    csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45,
                                 mask=white_matter)

    stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .25)

    seeds = utils.seeds_from_mask(seed_mask, affine, density=[2, 2, 2])

    streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                          affine=affine, step_size=.5)
    streamlines = Streamlines(streamlines_generator)

    sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
    save_trk(sft, "tractogram_EuDX.trk", streamlines)

