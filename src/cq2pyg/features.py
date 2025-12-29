"""
Feature tensor building for PyG HeteroData.

Converts geometry data into fixed-size tensors for each node type.
"""

from typing import List, Tuple

import torch

from .geometry import VertexGeometry, EdgeGeometry, FaceGeometry, ControlPoint
from .types import (
    CurveType, SurfaceType,
    NUM_CURVE_TYPES, NUM_SURFACE_TYPES,
    VERTEX_FEATURE_DIM, EDGE_FEATURE_DIM, FACE_FEATURE_DIM, CONTROL_POINT_FEATURE_DIM
)


def build_vertex_features(geometries: List[VertexGeometry]) -> torch.Tensor:
    """
    Build vertex feature tensor.

    Args:
        geometries: List of VertexGeometry objects

    Returns:
        Tensor of shape [num_vertices, 3] containing (x, y, z) coordinates
    """
    if not geometries:
        return torch.empty((0, VERTEX_FEATURE_DIM), dtype=torch.float32)

    features = torch.zeros((len(geometries), VERTEX_FEATURE_DIM), dtype=torch.float32)
    for i, geom in enumerate(geometries):
        features[i, 0] = geom.x
        features[i, 1] = geom.y
        features[i, 2] = geom.z
    return features


def build_edge_features(geometries: List[EdgeGeometry]) -> torch.Tensor:
    """
    Build edge (curve) feature tensor.

    Features layout:
    - [0:NUM_CURVE_TYPES]: curve type one-hot
    - [NUM_CURVE_TYPES]: orientation
    - [NUM_CURVE_TYPES+1]: degree
    - [NUM_CURVE_TYPES+2]: is_closed
    - [NUM_CURVE_TYPES+3:NUM_CURVE_TYPES+5]: t_min, t_max
    - [NUM_CURVE_TYPES+5:NUM_CURVE_TYPES+8]: line_direction (dx, dy, dz)
    - [NUM_CURVE_TYPES+8:NUM_CURVE_TYPES+11]: center (cx, cy, cz)
    - [NUM_CURVE_TYPES+11:NUM_CURVE_TYPES+14]: axis (ax, ay, az)
    - [NUM_CURVE_TYPES+14]: radius

    Args:
        geometries: List of EdgeGeometry objects

    Returns:
        Tensor of shape [num_edges, EDGE_FEATURE_DIM]
    """
    if not geometries:
        return torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    features = torch.zeros((len(geometries), EDGE_FEATURE_DIM), dtype=torch.float32)
    offset = 0

    for i, geom in enumerate(geometries):
        # Curve type one-hot
        features[i, geom.curve_type] = 1.0
        offset = NUM_CURVE_TYPES

        # Orientation
        features[i, offset] = geom.orientation
        offset += 1

        # Degree
        features[i, offset] = geom.degree
        offset += 1

        # Is closed
        features[i, offset] = 1.0 if geom.is_closed else 0.0
        offset += 1

        # Parameter bounds
        features[i, offset] = geom.t_min
        features[i, offset + 1] = geom.t_max
        offset += 2

        # Line direction
        if geom.line_direction is not None:
            features[i, offset:offset + 3] = torch.tensor(geom.line_direction)
        offset += 3

        # Center
        if geom.center is not None:
            features[i, offset:offset + 3] = torch.tensor(geom.center)
        offset += 3

        # Axis
        if geom.axis is not None:
            features[i, offset:offset + 3] = torch.tensor(geom.axis)
        offset += 3

        # Radius
        if geom.radius is not None:
            features[i, offset] = geom.radius

    return features


def build_face_features(geometries: List[FaceGeometry]) -> torch.Tensor:
    """
    Build face (surface) feature tensor.

    Features layout:
    - [0:NUM_SURFACE_TYPES]: surface type one-hot
    - [NUM_SURFACE_TYPES]: orientation
    - [+1:+3]: u_degree, v_degree
    - [+3:+5]: is_u_closed, is_v_closed
    - [+5:+9]: u_min, u_max, v_min, v_max
    - [+9:+12]: plane_normal
    - [+12:+15]: plane_origin
    - [+15:+18]: axis_direction
    - [+18:+21]: axis_origin
    - [+21:+23]: radius, radius2

    Args:
        geometries: List of FaceGeometry objects

    Returns:
        Tensor of shape [num_faces, FACE_FEATURE_DIM]
    """
    if not geometries:
        return torch.empty((0, FACE_FEATURE_DIM), dtype=torch.float32)

    features = torch.zeros((len(geometries), FACE_FEATURE_DIM), dtype=torch.float32)

    for i, geom in enumerate(geometries):
        offset = 0

        # Surface type one-hot
        features[i, geom.surface_type] = 1.0
        offset = NUM_SURFACE_TYPES

        # Orientation
        features[i, offset] = geom.orientation
        offset += 1

        # Degrees
        features[i, offset] = geom.u_degree
        features[i, offset + 1] = geom.v_degree
        offset += 2

        # Closed flags
        features[i, offset] = 1.0 if geom.is_u_closed else 0.0
        features[i, offset + 1] = 1.0 if geom.is_v_closed else 0.0
        offset += 2

        # Parameter bounds
        features[i, offset] = geom.u_min
        features[i, offset + 1] = geom.u_max
        features[i, offset + 2] = geom.v_min
        features[i, offset + 3] = geom.v_max
        offset += 4

        # Plane normal
        if geom.plane_normal is not None:
            features[i, offset:offset + 3] = torch.tensor(geom.plane_normal)
        offset += 3

        # Plane origin
        if geom.plane_origin is not None:
            features[i, offset:offset + 3] = torch.tensor(geom.plane_origin)
        offset += 3

        # Axis direction
        if geom.axis_direction is not None:
            features[i, offset:offset + 3] = torch.tensor(geom.axis_direction)
        offset += 3

        # Axis origin
        if geom.axis_origin is not None:
            features[i, offset:offset + 3] = torch.tensor(geom.axis_origin)
        offset += 3

        # Radii
        if geom.radius is not None:
            features[i, offset] = geom.radius
        if geom.radius2 is not None:
            features[i, offset + 1] = geom.radius2

    return features


def build_control_point_features(control_points: List[ControlPoint]) -> torch.Tensor:
    """
    Build control point feature tensor.

    Args:
        control_points: List of ControlPoint objects

    Returns:
        Tensor of shape [num_control_points, 4] containing (x, y, z, weight)
    """
    if not control_points:
        return torch.empty((0, CONTROL_POINT_FEATURE_DIM), dtype=torch.float32)

    features = torch.zeros((len(control_points), CONTROL_POINT_FEATURE_DIM), dtype=torch.float32)
    for i, cp in enumerate(control_points):
        features[i, 0] = cp.x
        features[i, 1] = cp.y
        features[i, 2] = cp.z
        features[i, 3] = cp.weight
    return features


def build_edge_index(pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Build edge_index tensor from list of (source, target) pairs.

    Args:
        pairs: List of (source_idx, target_idx) tuples

    Returns:
        Tensor of shape [2, num_edges] in COO format
    """
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(pairs, dtype=torch.long).t().contiguous()
