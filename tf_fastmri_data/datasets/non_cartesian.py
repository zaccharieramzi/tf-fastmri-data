import tensorflow as tf
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft.mri.dcomp_calc import calculate_density_compensator

from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.extract_smaps import non_cartesian_extract_smaps
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d
from tf_fastmri_data.preprocessing_utils.fourier.non_cartesian import nufft
from tf_fastmri_data.preprocessing_utils.non_cartesian_trajectories import (
    get_radial_trajectory,
    get_spiral_trajectory,
    get_debugging_cartesian_trajectory,
)
from tf_fastmri_data.preprocessing_utils.scaling import scale_tensors
from tf_fastmri_data.preprocessing_utils.crop import adjust_image_size


IMAGE_SIZE = (640, 400)

class NonCartesianFastMRIDatasetBuilder(FastMRIDatasetBuilder):
    def __init__(
            self,
            image_size=IMAGE_SIZE,
            acq_type='radial',
            dcomp=True,
            scale_factor=1e6,
            traj=None,
            crop_image_data=False,
            **kwargs,
        ):
        self.image_size = image_size
        self.acq_type = acq_type
        self.traj = traj
        self._check_acq_type()
        self.dcomp = dcomp
        self.scale_factor = scale_factor
        self.crop_image_data = crop_image_data
        self.nufft_obj = KbNufftModule(
            im_size=self.image_size,
            grid_size=None,
            norm='ortho',
        )
        super(NonCartesianFastMRIDatasetBuilder, self).__init__(
            **kwargs,
        )
        if self.brain:
            raise ValueError(
                'Currently the non cartesian data works only with knee data.')
        self._check_mode()
        self._check_dcomp_multicoil()

    def _check_acq_type(self,):
        if self.acq_type not in ['spiral', 'radial', 'cartesian_debug', 'other']:
            raise ValueError(
                f'acq_type must be spiral, radial or cartesian_debug but is {self.acq_type}'
            )
        if self.acq_type == 'other' and self.traj is None:
            raise ValueError(
                f'Please provide a trajectory as input in case `acq_type` is `other`'
            )

    def _check_mode(self,):
        if self.mode == 'test':
            raise ValueError('NonCartesian dataset cannot be used in test mode')

    def _check_dcomp_multicoil(self,):
        if self.multicoil and not self.dcomp:
            raise ValueError('You must use density compensation when in multicoil')

    def generate_trajectory(self,):
        if self.acq_type == 'radial':
            traj = get_radial_trajectory(self.image_size, af=self.af)
        elif self.acq_type == 'cartesian':
            traj = get_debugging_cartesian_trajectory()
        elif self.acq_type == 'spiral':
            traj = get_spiral_trajectory(self.image_size, af=self.af)
        elif self.acq_type == 'other':
            traj = self.traj
        return traj

    def preprocessing(self, image, kspace):
        traj = self.generate_trajectory()
        interpob = self.nufft_obj._extract_nufft_interpob()
        nufftob_forw = kbnufft_forward(interpob, multiprocessing=True)
        nufftob_back = kbnufft_adjoint(interpob, multiprocessing=True)
        if self.dcomp:
            dcomp = calculate_density_compensator(
                interpob,
                nufftob_forw,
                nufftob_back,
                traj[0],
            )
        traj = tf.repeat(traj, tf.shape(image)[0], axis=0)
        orig_image_channels = ortho_ifft2d(kspace)
        if self.crop_image_data:
            image = adjust_image_size(image, self.image_size)
        nc_kspace = nufft(self.nufft_obj, orig_image_channels, traj, self.image_size, multicoil=self.multicoil)
        nc_kspace, image = scale_tensors(nc_kspace, image, scale_factor=self.scale_factor)
        image = image[..., None]
        nc_kspaces_channeled = nc_kspace[..., None]
        orig_shape = tf.ones([tf.shape(kspace)[0]], dtype=tf.int32) * self.image_size[-1]
        if not self.crop_image_data:
            output_shape = tf.shape(image)[1:][None, :]
            output_shape = tf.tile(output_shape, [tf.shape(image)[0], 1])
        extra_args = (orig_shape,)
        if self.dcomp:
            dcomp = tf.ones(
                [tf.shape(kspace)[0], tf.shape(dcomp)[0]],
                dtype=dcomp.dtype,
            ) * dcomp[None, :]
            extra_args += (dcomp,)
        model_inputs = (nc_kspaces_channeled, traj)
        if self.multicoil:
            smaps = non_cartesian_extract_smaps(nc_kspace, traj, dcomp, nufftob_back, self.image_size)
            model_inputs += (smaps,)
        if not self.crop_image_data:
            model_inputs += (output_shape,)
        model_inputs += (extra_args,)
        return model_inputs, image
