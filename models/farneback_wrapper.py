import cv2
import numpy as np
import torch

from .base import BaseOpticalFlowModel
from pipeline_utils.uncertainty import compute_uncertainty_batch

class FarnebackOpticalFlow(BaseOpticalFlowModel):
    """OpenCV's Farneback dense optical flow. CPU only."""

    def __init__(self, uncert_window=8, uncert_max_shift=None, **kwargs):
        super().__init__(device='cpu')
        self.uncert_window = uncert_window
        self.uncert_max_shift = uncert_max_shift

    def predict_batch(self, image1_batch, image2_batch):
        """
        Runs Farneback optical flow sequentially over the batch.
        
        Args:
            image1_batch: PyTorch tensor (B, 3, H, W)
            image2_batch: PyTorch tensor (B, 3, H, W)
            
        Returns:
            (flow_batch, uncertainty) - see BaseOpticalFlowModel.
        """
        # Convert PyTorch Tensors (B, 3, H, W) back to numpy (B, H, W, 3) 
        # and from RGB to Grayscale for cv2.calcOpticalFlowFarneback
        batch_size, _, h, w = image1_batch.shape
        flow_out = np.zeros((batch_size, h, w, 2), dtype=np.float32)
        
        image1_np = image1_batch.permute(0, 2, 3, 1).numpy()
        image2_np = image2_batch.permute(0, 2, 3, 1).numpy()
        
        for b in range(batch_size):
            # The tensor holds three identical channels on a 0-255 float scale;
            # take one and round rather than truncating with astype(uint8).
            gray1 = np.clip(image1_np[b, :, :, 0], 0, 255).round().astype(np.uint8)
            gray2 = np.clip(image2_np[b, :, :, 0], 0, 255).round().astype(np.uint8)
            
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            flow_out[b] = flow
            
        flow_tensor = torch.from_numpy(flow_out).permute(0, 3, 1, 2)
        uncertainty = compute_uncertainty_batch(
            image1_batch, image2_batch, flow_tensor,
            window_size=self.uncert_window, max_shift=self.uncert_max_shift)

        return flow_out, uncertainty
