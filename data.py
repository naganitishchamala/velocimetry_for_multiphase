"""Frame-pair dataset for optical flow over video files."""

import cv2
import numpy as np
from torch.utils.data import Dataset

from pipeline_utils.preprocessing import (
    crop_frame, apply_clahe, prepare_bgr_to_tensor, rotate_frame, scale_to_8bit_range,
)


class _SequentialVideoReader:
    """
    Reads frames from a video, keeping one capture open.

    Frame-accurate seeking is unreliable on inter-frame-compressed containers:
    ``CAP_PROP_POS_FRAMES`` may land on a nearby keyframe, silently pairing the
    wrong frames.  Because the tracker consumes indices in order, this reader
    decodes sequentially and only falls back to a seek when asked to jump
    backwards or far forwards.
    """

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        self.next_index = 0
        self._cache_index = None
        self._cache_frame = None

    def frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _read_next(self):
        ok, frame = self.cap.read()
        if not ok:
            raise IndexError(f"Failed to read frame {self.next_index} from {self.path}")
        self.next_index += 1
        return frame

    def frame(self, idx):
        if idx == self._cache_index:
            return self._cache_frame

        if idx < self.next_index or idx - self.next_index > 32:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.next_index = idx

        while self.next_index < idx:
            self._read_next()

        frame = self._read_next()
        self._cache_index = idx
        self._cache_frame = frame
        return frame

    def pair(self, idx):
        """Frames ``idx`` and ``idx + 1``, decoded consecutively."""
        first = self.frame(idx)
        second = self.frame(idx + 1)
        return first, second

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoPairDataset(Dataset):
    """
    Yields ``(image1, image2, index)`` for consecutive frame pairs.

    Applies optional rotation, cropping and CLAHE, in that order, so the ROI is
    taken from an already-levelled image.
    """

    def __init__(self, video_path, roi=(0, -1, 0, -1), use_clahe=False,
                 clahe_clip_limit=2.0, clahe_tile_size=8, max_frames=-1,
                 rotate_angle=0.0, rotate_center=None, bit_depth=None):
        self.video_path = video_path
        self.roi = roi
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.max_frames = max_frames
        self.rotate_angle = rotate_angle
        self.rotate_center = rotate_center  # (y, x) or None
        self.bit_depth = bit_depth

        self.is_cine = video_path.lower().endswith('.cine')
        self._reader = None
        self._cine = None

        if self.is_cine:
            frames = len(self._open_cine())
        else:
            reader = _SequentialVideoReader(video_path)
            frames = reader.frame_count()
            reader.close()

        if frames <= 1:
            raise ValueError(f"{video_path} contains {frames} frame(s); need at least 2.")

        self.frame_count = min(frames, max_frames) if max_frames > 0 else frames
        self.length = max(0, self.frame_count - 1)

    def _open_cine(self):
        if self._cine is None:
            try:
                import pims
            except ImportError as exc:
                raise ImportError(
                    "Reading .cine files requires the 'pims' package: pip install pims"
                ) from exc
            self._cine = pims.open(self.video_path)
        return self._cine

    def __len__(self):
        return self.length

    def preprocess(self, frame_raw):
        is_grayscale = (frame_raw.ndim == 2)

        if is_grayscale:
            # Float 0-255: keeps the sensor's full dynamic range through
            # rotation and CLAHE instead of quantising to 8 bits up front.
            frame_raw = scale_to_8bit_range(frame_raw, self.bit_depth)

        if self.rotate_angle != 0.0 and self.rotate_center is not None:
            frame_raw = rotate_frame(frame_raw, self.rotate_center, self.rotate_angle)

        frame = crop_frame(frame_raw, self.roi)

        if self.use_clahe:
            frame = apply_clahe(frame, self.clahe_clip_limit, self.clahe_tile_size,
                                is_grayscale=is_grayscale)

        return prepare_bgr_to_tensor(frame, is_grayscale=is_grayscale)

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(f"index {idx} out of range for {self.length} pairs")

        if self.is_cine:
            video = self._open_cine()
            frame1_raw = np.asarray(video[idx])
            frame2_raw = np.asarray(video[idx + 1])
        else:
            if self._reader is None:
                self._reader = _SequentialVideoReader(self.video_path)
            frame1_raw, frame2_raw = self._reader.pair(idx)

        return self.preprocess(frame1_raw), self.preprocess(frame2_raw), idx

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._cine is not None:
            self._cine.close()
            self._cine = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
