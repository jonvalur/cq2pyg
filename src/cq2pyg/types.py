"""
Type constants for B-Rep curve and surface types.

These map OpenCASCADE GeomAbs types to integer indices for one-hot encoding.
"""

from enum import IntEnum


class CurveType(IntEnum):
    """Curve types from OpenCASCADE GeomAbs_CurveType."""
    LINE = 0
    CIRCLE = 1
    ELLIPSE = 2
    HYPERBOLA = 3
    PARABOLA = 4
    BEZIER = 5
    BSPLINE = 6
    OFFSET = 7
    OTHER = 8


class SurfaceType(IntEnum):
    """Surface types from OpenCASCADE GeomAbs_SurfaceType."""
    PLANE = 0
    CYLINDER = 1
    CONE = 2
    SPHERE = 3
    TORUS = 4
    BEZIER = 5
    BSPLINE = 6
    REVOLUTION = 7
    EXTRUSION = 8
    OFFSET = 9
    OTHER = 10


# Number of types for one-hot encoding dimensions
NUM_CURVE_TYPES = len(CurveType)
NUM_SURFACE_TYPES = len(SurfaceType)

# Feature dimensions
VERTEX_FEATURE_DIM = 3  # x, y, z
CONTROL_POINT_FEATURE_DIM = 4  # x, y, z, weight

# Edge features: curve_type (one-hot) + orientation + degree + is_closed + t_bounds + geometric params
# Geometric params cover: direction (3), center (3), axis (3), radius (1) = 10
EDGE_FEATURE_DIM = NUM_CURVE_TYPES + 1 + 1 + 1 + 2 + 10  # = 24

# Face features: surface_type (one-hot) + orientation + degrees (2) + closed (2) + uv_bounds (4) + geometric params
# Geometric params cover: normal (3), origin (3), axis_dir (3), axis_origin (3), radii (2) = 14
FACE_FEATURE_DIM = NUM_SURFACE_TYPES + 1 + 2 + 2 + 4 + 14  # = 34
