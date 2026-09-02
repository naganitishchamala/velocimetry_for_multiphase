"""Overlay renderers for inspecting velocity fields against the source video."""

import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from pipeline_utils.preprocessing import rotate_frame, crop_frame


def block_center(index, window):
    """Pixel centre of averaging block *index*, matching pipeline_utils.io."""
    return index * window + (window - 1) / 2.0


def _draw_arrows(frame, u, v, win_h, win_w, skip, scale):
    """Draw the vector field, anchored at true block centres."""
    ny, nx = u.shape
    for j in range(0, ny, skip):
        for i in range(0, nx, skip):
            if not (np.isfinite(u[j, i]) and np.isfinite(v[j, i])):
                continue
            # round, not truncate: int() would silently drop every vector
            # shorter than one pixel, hiding slow regions entirely.
            du = int(round(float(u[j, i]) * scale))
            dv = int(round(float(v[j, i]) * scale))
            y_loc = int(round(block_center(j, win_h)))
            x_loc = int(round(block_center(i, win_w)))
            cv2.arrowedLine(frame, (x_loc, y_loc), (x_loc + du, y_loc + dv),
                            (0, 0, 255), 1, tipLength=0.3)

def _get_processed_metadata(velData, width, height):
    """Helper to extract and compute geometric properties from HDF5 attributes."""
    attrs = dict(velData.attrs)
    roi = list(attrs.get('roi', [0, -1, 0, -1]))
    if roi[1] == -1: roi[1] = height
    if roi[3] == -1: roi[3] = width
    
    rotate_angle = attrs.get('rotate_angle', 0.0)
    throat_loc = attrs.get('throat_loc_px', None)
    
    win_w = int(attrs.get('window_width', 1))
    win_h = int(attrs.get('window_height', 1))
    
    out_height = roi[1] - roi[0]
    out_width = roi[3] - roi[2]
    
    return roi, rotate_angle, throat_loc, win_w, win_h, out_height, out_width

def _align_frame(frame, throat_loc, rotate_angle, roi):
    """Applies rotation and cropping to align the raw video frame with the HDF5 data fields."""
    if rotate_angle != 0.0 and throat_loc is not None:
        frame = rotate_frame(frame, tuple(throat_loc), rotate_angle)
    return crop_frame(frame, roi)

def create_quiver_video(video_path, h5_filepath, output_path, scale=2.0, skip=16, fps=30):
    """
    Overlays a sparse velocity vector field (quiver plot) onto the aligned video frames.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with h5py.File(h5_filepath, 'r') as f:
        velData = f['velocity']
        frames = velData.shape[0]
        
        roi, rotate_angle, throat_loc, win_w, win_h, out_height, out_width = _get_processed_metadata(velData, width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        ret, _ = cap.read() 
        
        for k in tqdm(range(frames), desc="Rendering Quiver Video"):
            ret, frame = cap.read()
            if not ret: break
            
            frame = _align_frame(frame, throat_loc, rotate_angle, roi)
            
            _draw_arrows(frame, velData[k, :, :, 0], velData[k, :, :, 1],
                         win_h, win_w, skip, scale)
            out.write(frame)
            
    cap.release()
    out.release()

def create_quiver_image(video_path, h5_filepath, output_path, frame_idx=0, scale=2.0, skip=16):
    """
    Overlays a sparse velocity vector field onto a single aligned video frame, saved as an image.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
    ret, frame = cap.read()
    if not ret: return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with h5py.File(h5_filepath, 'r') as f:
        velData = f['velocity']
        roi, rotate_angle, throat_loc, win_w, win_h, out_height, out_width = _get_processed_metadata(velData, width, height)
        
        frame = _align_frame(frame, throat_loc, rotate_angle, roi)
        
        u = velData[frame_idx, :, :, 0]
        v = velData[frame_idx, :, :, 1]

    _draw_arrows(frame, u, v, win_h, win_w, skip, scale)
    cv2.imwrite(output_path, frame)
    cap.release()

def create_spacetime_diagram(video_path, roi):
    """
    Creates an x-t (space-time) diagram by extracting a specific row (or averaged rows)
    from a video over time to track cavity shedding.
    
    Args:
        video_path: Path to the raw high-speed video.
        roi: Tuple of (y1, y2, x1, x2) defining the horizontal strip to extract and average.
    Returns:
        numpy.ndarray representing the (Time, Space) image.
    """
    cap = cv2.VideoCapture(video_path)
    y1, y2, x1, x2 = roi
    
    xt_image = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        y_slice = slice(y1, y2 if y2 != -1 else None)
        x_slice = slice(x1, x2 if x2 != -1 else None)
        
        strip = gray[y_slice, x_slice]
        
        # Average vertically across the strip thickness to get a 1D intensity profile
        intensity_profile = np.mean(strip, axis=0)
        xt_image.append(intensity_profile)
        
    cap.release()
    return np.array(xt_image)

def create_contour_video(video_path, h5_filepath, dataset_name, output_path, alpha=0.5, fps=30):
    """
    Overlays a translucent scalar field (e.g., uncertainty, divergence) as a heatmap
    onto the aligned grayscale video.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with h5py.File(h5_filepath, 'r') as f:
        data = f[dataset_name]
        velData = f['velocity'] if 'velocity' in f else data
        roi, rotate_angle, throat_loc, win_w, win_h, out_height, out_width = _get_processed_metadata(velData, width, height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        frames = data.shape[0]

        # Percentile range from a subsample: the full dataset can be many GB.
        sample_idx = np.unique(np.linspace(0, frames - 1, min(frames, 50)).astype(int))
        sample = np.concatenate([np.asarray(data[i]).ravel() for i in sample_idx])
        sample = sample[np.isfinite(sample)]
        vmin, vmax = np.percentile(sample, [5, 95]) if sample.size else (0.0, 1.0)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        
        ret, _ = cap.read()
        
        for k in tqdm(range(frames), desc=f"Rendering {dataset_name} Contour Video"):
            ret, frame = cap.read()
            if not ret: break
            
            frame = _align_frame(frame, throat_loc, rotate_angle, roi)
                
            scalar_field = np.asarray(data[k])
            if scalar_field.ndim == 3 and scalar_field.shape[-1] == 2:
                scalar_field = np.linalg.norm(scalar_field, axis=2)
                
            scalar_resized = cv2.resize(scalar_field, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
            normed = np.clip((scalar_resized - vmin) / (vmax - vmin), 0, 1) * 255
            normed = normed.astype(np.uint8)
            heatmap = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
            
            blended = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
            out.write(blended)
            
    cap.release()
    out.release()

def create_profile_video(video_path, h5_filepath, output_path, num_profiles=10, fps=10):
    """
    Renders an mp4 video overlaying 10 equidistant instantaneous U-velocity profiles 
    on top of the velocity magnitude heatmap and aligned camera frames.
    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with h5py.File(h5_filepath, 'r') as f:
        velData = f['velocity']
        attrs = dict(velData.attrs)
        roi, rotate_angle, throat_loc, win_w, win_h, out_height, out_width = _get_processed_metadata(velData, width, height)
        
        mm_per_px = attrs.get('mm_per_px', 1.0)
        fps_capture = attrs.get('fps_capture', 1.0)
        n_frames = velData.shape[0]
        Ny, Nx = velData.shape[1:3]
        
    scale_fac = mm_per_px * 1e-3 * fps_capture
    extent = [0, out_width, out_height, 0]
    
    x_indices = np.linspace(0, Nx - 1, num_profiles, dtype=int)
    y_px = block_center(np.arange(Ny), win_h)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    profile_scale_px = 2.0
    ret, _ = cap.read()

    print(f"Generating profile intersection animation at {fps} fps...")
    handle = h5py.File(h5_filepath, 'r')
    try:
      for k in tqdm(range(n_frames), desc="Rendering Profile Video"):
        ret, frame = cap.read()
        if not ret: break

        frame = _align_frame(frame, throat_loc, rotate_angle, roi)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        vel = handle['velocity'][k]
        uncert = handle['uncertainty'][k] if 'uncertainty' in handle else None
        
        u = vel[..., 0] * scale_fac
        v = -vel[..., 1] * scale_fac
        mag = np.sqrt(u**2 + v**2)
        
        if uncert is not None:
            sigma_u = uncert[..., 0] * scale_fac
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.imshow(img_rgb)
        contour = ax.imshow(mag, cmap='jet', alpha=0.4, extent=extent, vmin=0, vmax=13.0)
        
        cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Velocity Magnitude (m/s)')
        
        for idx_x in x_indices:
            x_base_px = block_center(idx_x, win_w)
            u_prof = u[:, idx_x]
            
            ax.axvline(x=x_base_px, color='white', linestyle='--', alpha=0.4)
            
            if uncert is not None:
                su_prof = sigma_u[:, idx_x]
                ax.fill_betweenx(y_px, 
                                 x_base_px + (u_prof - su_prof) * profile_scale_px,
                                 x_base_px + (u_prof + su_prof) * profile_scale_px,
                                 color='red', alpha=0.3)
                
            ax.plot(x_base_px + u_prof * profile_scale_px, y_px, color='red', linewidth=1.5)
            
        ax.set_title(f"Instantaneous Profile Flow field (Frame {k})")
        ax.set_xlim(0, out_width)
        ax.set_ylim(out_height, 0)
        ax.axis('off')
        
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        
        img_rgba = np.asarray(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, fps, (img_bgr.shape[1], img_bgr.shape[0]))
            
        out.write(img_bgr)
        plt.close(fig)
    finally:
        handle.close()

    cap.release()
    if out is not None:
        out.release()
