import cv2
import numpy as np
import scipy.signal
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, max_frames=None, skip=1):
    """
    Extracts frames from a video file and saves them as images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames is not None and saved >= max_frames):
            break
            
        if count % skip == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{saved:05d}.png"), frame)
            saved += 1
            
        count += 1
        
    cap.release()
    return saved

def change_framerate(video_path, output_path, keep_every_n=2, new_fps=30):
    """
    Reduces the temporal resolution of a video by keeping only every Nth frame.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    
    count = 0
    for _ in tqdm(range(total_frames), desc="Changing Framerate"):
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % keep_every_n == 0:
            out.write(frame)
        count += 1
        
    cap.release()
    out.release()

def compute_shedding_frequency(video_path, roi=None, fps=None, nperseg=8192, band=None):
    """
    Dominant shedding frequency from brightness fluctuations, via Welch's PSD.

    Args:
        video_path: recording to analyse.
        roi: optional (y1, y2, x1, x2) crop; -1 means 'to the end'.
        fps: capture rate in Hz. Required -- read it from the HDF5 metadata
            (``fps_capture``) rather than assuming a value, or every reported
            frequency is wrong by the ratio of the true rate to the assumed one.
        nperseg: Welch segment length. Frequency resolution is fps / nperseg.
        band: (low, high) Hz to search for the peak. Defaults to
            (50, fps / 20), which skips the DC-adjacent bins that otherwise
            dominate a non-stationary brightness signal.

    Returns:
        (signal, frequencies, psd, dominant_frequency)
    """
    if fps is None:
        raise ValueError(
            "fps is required. Pass the 'fps_capture' attribute from the "
            "tracker's HDF5 output, or the camera's capture rate in Hz.")

    cap = cv2.VideoCapture(video_path)
    brightness_over_time = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if roi is not None:
            y1, y2, x1, x2 = roi
            y_slice = slice(y1, y2 if y2 != -1 else None)
            x_slice = slice(x1, x2 if x2 != -1 else None)
            gray = gray[y_slice, x_slice]
            
        brightness_over_time.append(np.mean(gray))
        
    cap.release()
    
    signal = np.array(brightness_over_time)
    if signal.size < 4:
        raise ValueError(f"Only {signal.size} frame(s) read from {video_path}")
    signal = signal - np.mean(signal)  # remove DC offset

    f, Pxx = scipy.signal.welch(signal, fs=fps, nperseg=min(nperseg, signal.size))

    low, high = band if band is not None else (50.0, fps / 20.0)
    in_band = (f >= low) & (f <= high)
    if not in_band.any():
        in_band = f > 0
    candidates = np.flatnonzero(in_band)
    dominant_freq = f[candidates[np.argmax(Pxx[candidates])]]

    return signal, f, Pxx, dominant_freq

def compute_cavity_metrics(video_path, x_min=100):
    """
    Time-averaged brightness and variance, and the cavity extent implied by them.

    Uses Welford's single-pass algorithm, so the statistics are divided by the
    number of frames actually decoded rather than the container's frame-count
    estimate, and no rewind is needed between passes.

    Args:
        x_min: ignore the first this many pixel columns when locating the
            maximum-variance point (the throat region is always unsteady).

    Returns:
        (mean, variance, cavity_mask, max_variance_index) -- the last two are
        None if the frame is narrower than *x_min*.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = 0
    mean = np.zeros((height, width), dtype=np.float64)
    m2 = np.zeros((height, width), dtype=np.float64)

    for _ in tqdm(range(max(frame_count, 1)), desc="Cavity mean/variance"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        n += 1
        delta = gray - mean
        mean += delta / n
        m2 += delta * (gray - mean)

    cap.release()
    if n == 0:
        raise ValueError(f"No frames could be read from {video_path}")

    avg = mean.astype(np.float32)
    var = (m2 / n).astype(np.float32)

    limit = x_min
    if var.shape[1] > limit:
        # Find maximum variance pixel after 'limit' x-coordinate
        max_indx = np.unravel_index(np.argmax(var[:, limit:]), var[:, limit:].shape)
        max_indx = (max_indx[0], max_indx[1] + limit)
        
        mag = avg[max_indx]
        cavity_mask = avg > mag
        
        return avg, var, cavity_mask, max_indx
    else:
        return avg, var, None, None
