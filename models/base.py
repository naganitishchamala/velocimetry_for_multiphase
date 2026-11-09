from abc import ABC, abstractmethod


class BaseOpticalFlowModel(ABC):
    """Interface implemented by every dense optical flow backend."""

    @abstractmethod
    def __init__(self, device='cpu', **kwargs):
        self.device = device

    @abstractmethod
    def predict_batch(self, image1_batch, image2_batch):
        """
        Estimate flow and its uncertainty for a batch of frame pairs.

        Args:
            image1_batch, image2_batch: (B, 3, H, W) float tensors on the 0-255 scale.

        Returns:
            (flow_batch, uncertainty) where ``flow_batch`` is a (B, H, W, 2)
            numpy array of pixel displacements and ``uncertainty`` is an
            :class:`~pipeline_utils.uncertainty.UncertaintyResult`.
        """
