import tensorflow as tf
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf

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


IMAGE_SIZE = (640, 400)

class NonCartesianFastMRIDatasetBuilder(FastMRIDatasetBuilder):
    def __init__(
            self,
            image_size=IMAGE_SIZE,
            acq_type='radial',
            dcomp=True,
            **kwargs,
        ):
        self.image_size = image_size
        self.acq_type = acq_type
        self._check_acq_type()
        self.dcomp = dcomp
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
        if self.acq_type not in ['spiral', 'radial', 'cartesian_debug']:
            raise ValueError(
                f'acq_type must be spiral, radial or cartesian_debug but is {self.acq_type}'
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
        return traj

    def preprocessing(self, image, kspace):
        traj = self.generate_trajectory()
        interpob = self.nfft_obj._extract_nufft_interpob()
        nufftob_forw = kbnufft_forward(interpob)
        nufftob_back = kbnufft_adjoint(interpob)
        if self.dcomp:
            dcomp = calculate_radial_dcomp_tf(
                interpob,
                nufftob_forw,
                nufftob_back,
                traj[0],
            )
        traj = tf.repeat(traj, tf.shape(image)[0], axis=0)
        orig_image_channels = ortho_ifft2d(kspace)
        nc_kspace = nufft(self.nfft_obj, orig_image_channels, traj, self.image_size)
        nc_kspace, image = scale_tensors(nc_kspace, image, scale_factor=self.scale_factor)
        image = image[..., None]
        nc_kspace = nc_kspace[..., None]
        orig_shape = tf.ones([tf.shape(kspace)[0]], dtype=tf.int32) * tf.shape(kspace)[-1]
        extra_args = (orig_shape,)
        if self.dcomp:
            dcomp = tf.ones(
                [tf.shape(kspace)[0], tf.shape(dcomp)[0]],
                dtype=dcomp.dtype,
            ) * dcomp[None, :]
            extra_args += (dcomp,)
        model_inputs = (nc_kspaces_channeled, traj)
        if self.multicoil
            smaps = non_cartesian_extract_smaps(nc_kspace, traj, dcomp, nufftob_back, orig_shape)
            model_inputs += (smaps,)
        model_inputs += (extra_args,)
        return model_inputs, image
