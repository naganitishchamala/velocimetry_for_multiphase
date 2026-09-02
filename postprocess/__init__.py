"""Post-processing: statistics extraction and field inspection."""

from .velocity import (
    compute_spatial_derivatives,
    extract_profile_slice,
    rotate_velocity,
    compute_time_average_fields,
    extract_line_profiles,
    integral_timescale,
    copy_provenance,
)
from .video import (
    extract_frames,
    change_framerate,
    compute_shedding_frequency,
    compute_cavity_metrics,
)
from .visualize import (
    create_quiver_video,
    create_quiver_image,
    create_spacetime_diagram,
    create_contour_video,
    create_profile_video,
    block_center,
)

__all__ = [
    'compute_spatial_derivatives', 'extract_profile_slice', 'rotate_velocity',
    'compute_time_average_fields', 'extract_line_profiles', 'integral_timescale',
    'copy_provenance', 'extract_frames', 'change_framerate',
    'compute_shedding_frequency', 'compute_cavity_metrics',
    'create_quiver_video', 'create_quiver_image', 'create_spacetime_diagram',
    'create_contour_video', 'create_profile_video', 'block_center',
]
