"""Custom/optional utilities.

This subpackage intentionally contains optional features that may depend on
non-core third-party libraries.
"""

from .remove_ground import remove_ground, align_pointcloud_to_ground_oxz  # noqa: F401
from .statistic_plane import statistic_plane  # noqa: F401
from .remove_outlier_cc import remove_outlier_cc  # noqa: F401
from .tsdf_fusion import tsdf_fuse_views  # noqa: F401
from .view_consistent_merge import view_consistent_merge  # noqa: F401
from .wall_slab_grid import (  # noqa: F401
	read_point_cloud_with_bbox,
	compute_bbox_scale,
	compute_wall_slab_by_y_span,
	build_slab_occupancy_grid_xz,
	build_slab_cut_lines,
	wall_slab_grid_from_points,
)
