import torch
import numpy as np
import cv2
from scipy.interpolate import griddata

from openpiv.windef import PIVSettings, first_pass, multipass_img_deform
from openpiv.validation import sig2noise_val, global_val, global_std

from .base import BaseOpticalFlowModel
from pipeline_utils.uncertainty import compute_uncertainty_batch

class OpenPIVModel(BaseOpticalFlowModel):
    def __init__(self, uncert_window=8, uncert_max_shift=None, **kwargs):
        # OpenPIV natively runs on CPU.
        super().__init__(device='cpu')
        self.uncert_window = uncert_window
        self.uncert_max_shift = uncert_max_shift
        
        # Match pivLabrun.m settings
        self.settings = PIVSettings()
        self.settings.windowsizes = (64, 32, 16, 8)
        self.settings.overlap = (32, 16, 8, 4)
        self.settings.num_iterations = 4
        
        # Advanced configurations matching MATLAB script 's' and 'p' parameters
        self.settings.subpixel_method = 'gaussian'
        self.settings.deformation_method = 'symmetric'
        self.settings.interpolation_order = 1 # Linear to avoid small-roi Spline crashes
        self.settings.sig2noise_method = 'peak2mean'
        self.settings.sig2noise_threshold = 1.0 # Standard weak peak removal
        
        # Outlier removal options
        self.settings.replace_vectors = True
        self.settings.smoothn = False
        self.settings.filter_method = 'localmean'
        self.settings.max_filter_iteration = 4
        self.settings.filter_kernel_size = 2

    def process_pair(self, img1, img2):
        """
        Processes a single pair of grayscale images through OpenPIV.
        """
        # (CLAHE preprocessing is now globally handled natively through `data.py`)
        
        # Iteration 1
        x, y, u, v, s2n = first_pass(img1, img2, self.settings)
        
        # openpiv.windef.multipass_img_deform expects masked arrays for tracking validations over iterations
        u = np.ma.masked_array(u, np.isnan(u))
        v = np.ma.masked_array(v, np.isnan(v))
        
        # Iterations 2..N (Deformation)
        flags = None
        for i in range(1, self.settings.num_iterations):
            x, y, u, v, grid_mask, flags = multipass_img_deform(
                img1, img2, i, x, y, u, v, self.settings
            )

        # OpenPIV's own outlier flags are a validated per-vector quality signal;
        # keep them so invalid vectors can be excluded downstream.
        return x, y, u, v, flags
            
    def _process_single_frame(self, b, image1_np, image2_np, h, w):
        gray1 = np.clip(image1_np[:, :, 0], 0, 255).round().astype(np.uint8)
        gray2 = np.clip(image2_np[:, :, 0], 0, 255).round().astype(np.uint8)

        x_pts, y_pts, u_pts, v_pts, flags = self.process_pair(gray1, gray2)

        u_flat = np.ma.filled(u_pts, np.nan).flatten()
        v_flat = np.ma.filled(v_pts, np.nan).flatten()
        points = np.column_stack((x_pts.flatten(), y_pts.flatten()))

        # Drop vectors OpenPIV flagged as outliers, plus any that are already NaN.
        good = np.isfinite(u_flat) & np.isfinite(v_flat)
        if flags is not None:
            good &= ~np.asarray(flags).flatten().astype(bool)
        if not good.any():
            nan_field = np.full((h, w), np.nan)
            return b, nan_field, nan_field.copy()

        grid_y, grid_x = np.mgrid[0:h, 0:w]

        # NaN outside the convex hull rather than a fabricated zero velocity:
        # an unmeasured point must not enter the statistics as if it were data.
        u_dense = griddata(points[good], u_flat[good], (grid_x, grid_y),
                           method='linear', fill_value=np.nan)
        v_dense = griddata(points[good], v_flat[good], (grid_x, grid_y),
                           method='linear', fill_value=np.nan)

        return b, u_dense, v_dense
    def predict_batch(self, image1_batch, image2_batch):
        """
        Runs OpenPIV optical flow sequentially over the batch.
        
        Args:
            image1_batch: PyTorch tensor (B, 3, H, W) normalized [0, 255]
            image2_batch: PyTorch tensor (B, 3, H, W) normalized [0, 255]
            
        Returns:
            numpy.ndarray of dense flow shape (B, H, W, 2)
        """
        batch_size, _, h, w = image1_batch.shape
        flow_out = np.zeros((batch_size, h, w, 2), dtype=np.float32)
        
        # ----- HPC / SLURM Safe CPU Parallelization -----
        import concurrent.futures
        import os
        
        # os.cpu_count() returns the total physical node cores (e.g., 128),
        # which ignores SLURM's --cpus-per-task allocation and leads to oversubscription.
        # os.sched_getaffinity(0) returns the exact CPU cores allocated to this process/cgroup.
        if hasattr(os, 'sched_getaffinity'):
            available_cores = len(os.sched_getaffinity(0))
        else:
            available_cores = os.cpu_count() or 4
            
        workers = min(batch_size, available_cores)
        
        # Convert PyTorch tensors to numpy arrays (B,H,W,C) for OpenPIV/OpenCV
        image1_np = image1_batch.permute(0, 2, 3, 1).cpu().numpy()
        image2_np = image2_batch.permute(0, 2, 3, 1).cpu().numpy()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._process_single_frame, b, image1_np[b], image2_np[b], h, w)
                for b in range(batch_size)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                b_idx, u_dense, v_dense = future.result()
                flow_out[b_idx, :, :, 0] = u_dense
                flow_out[b_idx, :, :, 1] = v_dense
            
        # The uncertainty estimator needs a finite flow field to warp with;
        # substitute zero there but record those points as invalid.
        finite = np.isfinite(flow_out).all(axis=-1)
        flow_tensor = torch.from_numpy(np.nan_to_num(flow_out)).permute(0, 3, 1, 2)
        uncertainty = compute_uncertainty_batch(
            image1_batch, image2_batch, flow_tensor,
            window_size=self.uncert_window, max_shift=self.uncert_max_shift)
        uncertainty = uncertainty._replace(valid=uncertainty.valid & finite)

        return flow_out, uncertainty
