import torch
import os
import sys
import numpy as np

# Add local core directory so we can load RAFT
current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(current_dir, '..', 'core')
sys.path.append(core_path)

try:
    from raft import RAFT
except ImportError as exc:  # pragma: no cover - depends on install layout
    raise ImportError(
        "Could not import the bundled RAFT implementation from core/. "
        "Run from the repository root, or install the package with 'pip install -e .'."
    ) from exc

from .base import BaseOpticalFlowModel
from pipeline_utils.uncertainty import compute_uncertainty_batch

class RAFTOpticalFlow(BaseOpticalFlowModel):
    def __init__(self, model_path, device='cuda', small=False, mixed_precision=False,
                 alternate_corr=False, iters=20, keep_iters=5, uncert_window=8,
                 uncert_max_shift=None, **kwargs):
        super().__init__(device=device)
        self.device = device
        self.iters = iters
        self.keep_iters = keep_iters
        self.uncert_window = uncert_window
        self.uncert_max_shift = uncert_max_shift

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RAFT weights not found: {model_path}\n"
                "Download a checkpoint into weights/ and pass it with --model."
            )
        
        # Build argparse namespace required by original RAFT constructor
        class Args:
            def __contains__(self, item):
                return hasattr(self, item)
            
            def __iter__(self):
                return iter(vars(self).keys())
                
        args = Args()
        args.small = small
        args.mixed_precision = mixed_precision
        args.alternate_corr = alternate_corr
        
        # Load the PyTorch Module via DataParallel (Natives the 'module.' prefix)
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        
        self.model.to(self.device)
        self.model.eval()

    def _compute_pad_params(self, ht, wd):
        """Compute padding to make dimensions multiples of 8."""
        pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
        pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
        return [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def predict_batch(self, image1_batch, image2_batch):
        """
        Args:
            image1_batch: (B, 3, H, W) on CPU
            image2_batch: (B, 3, H, W) on CPU
            
        Returns: 
            (flow_batch, uncert_batch) as numpy arrays (B, H, W, 2)
        """
        ht, wd = image1_batch.shape[-2:]
        pad_params = self._compute_pad_params(ht, wd)
        
        b_image1 = image1_batch.to(self.device)
        b_image2 = image2_batch.to(self.device)
        
        b_image1_pad = torch.nn.functional.pad(b_image1, pad_params, mode='replicate')
        b_image2_pad = torch.nn.functional.pad(b_image2, pad_params, mode='replicate')
        
        with torch.no_grad():
            outputs = self.model(b_image1_pad, b_image2_pad, iters=self.iters, test_mode=True)
            flow_up_pad, iterates = outputs[1], (outputs[2] if len(outputs) > 2 else None)

            # Unpad back to original dims
            c = [pad_params[2], flow_up_pad.shape[-2] - pad_params[3],
                 pad_params[0], flow_up_pad.shape[-1] - pad_params[1]]
            flow_up = flow_up_pad[..., c[0]:c[1], c[2]:c[3]]

            # Spread of the final GRU iterates: a convergence-based epistemic
            # signal that costs nothing, since RAFT computes these anyway.
            self.last_iterate_spread = None
            if iterates:
                stack = torch.stack([it[..., c[0]:c[1], c[2]:c[3]] for it in iterates])
                self.last_iterate_spread = stack.std(dim=0).permute(0, 2, 3, 1).cpu().numpy()

            uncertainty = compute_uncertainty_batch(
                b_image1, b_image2, flow_up,
                window_size=self.uncert_window, max_shift=self.uncert_max_shift)

        # (B, 2, H, W) -> (B, H, W, 2)
        flow_up = flow_up.permute(0, 2, 3, 1).cpu().numpy()
        return flow_up, uncertainty
