"""Chunked, streaming HDF5 output for velocity and uncertainty fields."""

import h5py
import numpy as np


def block_average(field, window_height, window_width):
    """
    Average a field over exact, non-overlapping blocks.

    Args:
        field: (H, W, C) array.
        window_height, window_width: block size in pixels.

    Returns:
        (H // window_height, W // window_width, C) array.  Block ``(j, i)`` is
        the mean over source rows ``j*wh : (j+1)*wh`` and columns
        ``i*ww : (i+1)*ww``, so its centre sits at
        ``(j*wh + (wh-1)/2, i*ww + (ww-1)/2)``.  Use :func:`block_centers` to
        recover those coordinates rather than assuming them.

    Any trailing rows or columns that do not fill a whole block are dropped.
    """
    h, w = field.shape[:2]
    n_rows = h // window_height
    n_cols = w // window_width
    trimmed = field[:n_rows * window_height, :n_cols * window_width]
    reshaped = trimmed.reshape(n_rows, window_height, n_cols, window_width, -1)
    return reshaped.mean(axis=(1, 3))


def block_centers(n_blocks, window):
    """Pixel coordinates of block centres produced by :func:`block_average`."""
    return np.arange(n_blocks) * window + (window - 1) / 2.0


def legacy_block_centers(n_blocks, window):
    """
    Block centres as produced by the pre-0.2.0 writer.

    That writer reflect-padded the field, ran ``cv2.filter2D`` with a
    ``window x window`` kernel, then took a strided slice.  For an even kernel
    OpenCV anchors at index ``window // 2`` rather than the true centre, so the
    value stored for block *k* is the mean of source rows
    ``[k*w - w//2, k*w - w//2 + w)`` -- centred at ``k*w + (w-1)/2 - w//2``,
    which is half a window below where :func:`block_centers` places it.

    Reading a legacy file with the current convention misplaces every profile by
    ``window // 2`` pixels.  That is a *different physical distance for each
    imaging configuration*, so it distorts comparisons between configurations
    more than it distorts any single profile.

    Block 0 is additionally contaminated by the reflection padding and is not
    a clean average of any source region; no offset can repair it.
    """
    return np.arange(n_blocks) * window + (window - 1) / 2.0 - (window // 2)


def uses_legacy_block_centers(attrs):
    """
    Whether an HDF5 file was written before the block-averaging fix.

    Detected by the absence of ``code_version``, which every current tracker run
    records.  Passing the attributes of any dataset in the file works.
    """
    return 'code_version' not in attrs


class HDF5Writer:
    """
    Window-averages and streams velocity and uncertainty fields to HDF5.

    Datasets written:
        velocity      (T, Ny, Nx, 2) float32 -- pixel displacement per frame pair
        uncertainty   (T, Ny, Nx, 2) float32 -- |residual| in pixels
        residual      (T, Ny, Nx, 2) float32 -- signed residual in pixels
        valid_fraction(T, Ny, Nx)    float32 -- fraction of the block that was
                                                informative, in [0, 1]

    ``uncertainty`` is a magnitude, so it is averaged over a block directly.
    That treats the errors within one block as fully correlated, which is the
    appropriate assumption here: the NCC windows overlap heavily and RAFT
    upsamples convexly from a 1/8-resolution field, so neighbouring residuals
    are not independent.  Consumers that need a different correlation model
    should work from ``residual`` instead.
    """

    def __init__(self, output_path, frame_count, h, w, window_height, window_width,
                 metadata, compression='lzf'):
        self.output_path = output_path
        self.window_height = window_height
        self.window_width = window_width

        n_pairs = frame_count - 1
        ny = h // window_height
        nx = w // window_width

        data_shape = (n_pairs, ny, nx, 2)
        mask_shape = (n_pairs, ny, nx)
        chunk_t = max(1, min(10, n_pairs))

        self.file = h5py.File(output_path, 'w')

        def make(name, shape):
            return self.file.create_dataset(
                name, shape=shape, chunks=(chunk_t,) + shape[1:],
                dtype='float32', compression=compression)

        self.velData = make('velocity', data_shape)
        self.uncertData = make('uncertainty', data_shape)
        self.residData = make('residual', data_shape)
        self.validData = make('valid_fraction', mask_shape)

        # Attributes are mirrored onto every dataset so any one of them is
        # self-describing when read in isolation.
        self._datasets = [self.velData, self.uncertData, self.residData, self.validData]
        for key, value in metadata.items():
            if value is None:
                continue
            for ds in self._datasets:
                ds.attrs[key] = value
            self.file.attrs[key] = value

    def write_batch(self, start_idx, flow_batch, uncertainty):
        """
        Args:
            start_idx: index of the first frame pair in this batch.
            flow_batch: (B, H, W, 2) array of pixel displacements.
            uncertainty: :class:`~pipeline_utils.uncertainty.UncertaintyResult`.
        """
        batch = flow_batch.shape[0]
        wh, ww = self.window_height, self.window_width

        if ww > 1 or wh > 1:
            for i in range(batch):
                self.velData[start_idx + i] = block_average(flow_batch[i], wh, ww)
                self.uncertData[start_idx + i] = block_average(uncertainty.sigma[i], wh, ww)
                self.residData[start_idx + i] = block_average(uncertainty.residual[i], wh, ww)
                valid = uncertainty.valid[i].astype(np.float32)[..., None]
                self.validData[start_idx + i] = block_average(valid, wh, ww)[..., 0]
        else:
            end = start_idx + batch
            self.velData[start_idx:end] = flow_batch
            self.uncertData[start_idx:end] = uncertainty.sigma
            self.residData[start_idx:end] = uncertainty.residual
            self.validData[start_idx:end] = uncertainty.valid.astype(np.float32)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
