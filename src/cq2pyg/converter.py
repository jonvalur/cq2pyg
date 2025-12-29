"""
Main converter for CadQuery to PyG HeteroData.

This module provides the main API for converting CadQuery shapes to
PyTorch Geometric heterogeneous graphs.
"""

from typing import List, Tuple, Union

import torch
from torch_geometric.data import HeteroData

import cadquery as cq
from OCP.TopoDS import TopoDS_Shape

from .topology import extract_topology, TopologyData
from .geometry import (
    extract_vertex_geometry, extract_edge_geometry, extract_face_geometry,
    VertexGeometry, EdgeGeometry, FaceGeometry, ControlPoint
)
from .features import (
    build_vertex_features, build_edge_features, build_face_features,
    build_control_point_features, build_edge_index
)


def cadquery_to_pyg(shape: Union[cq.Workplane, cq.Shape, TopoDS_Shape]) -> HeteroData:
    """
    Convert a CadQuery shape to a PyG heterogeneous graph.

    The resulting graph has the following structure:

    Node types:
        - 'vertex': Geometric vertices with (x, y, z) coordinates
        - 'edge': Topological edges (curves) with type, orientation, and parameters
        - 'face': Topological faces (surfaces) with type, orientation, and parameters
        - 'control_point': NURBS/B-spline control points with (x, y, z, weight)

    Edge types (relationships):
        - ('vertex', 'bounds', 'edge'): Vertex is an endpoint of edge
        - ('edge', 'bounds', 'face'): Edge is part of face boundary
        - ('face', 'adjacent', 'face'): Faces share an edge
        - ('control_point', 'controls', 'edge'): Control point belongs to curve
        - ('control_point', 'controls', 'face'): Control point belongs to surface

    Args:
        shape: A CadQuery Workplane, Shape, or raw TopoDS_Shape

    Returns:
        HeteroData graph containing all topology and geometry information
    """
    # Handle different input types
    if isinstance(shape, cq.Workplane):
        occ_shape = shape.val().wrapped
    elif isinstance(shape, cq.Shape):
        occ_shape = shape.wrapped
    elif isinstance(shape, TopoDS_Shape):
        occ_shape = shape
    else:
        raise TypeError(f"Expected CadQuery Workplane, Shape, or TopoDS_Shape, got {type(shape)}")

    # Extract topology
    topo = extract_topology(occ_shape)

    # Extract geometry for each entity
    vertex_geoms = [extract_vertex_geometry(v) for v in topo.vertices]
    edge_geoms = [extract_edge_geometry(e) for e in topo.edges]
    face_geoms = [extract_face_geometry(f) for f in topo.faces]

    # Collect all control points and build their relationships
    control_points: List[ControlPoint] = []
    cp_to_edge: List[Tuple[int, int]] = []  # (cp_idx, edge_idx)
    cp_to_edge_attr: List[Tuple[int]] = []  # (sequence_index,)
    cp_to_face: List[Tuple[int, int]] = []  # (cp_idx, face_idx)
    cp_to_face_attr: List[Tuple[int, int]] = []  # (u_index, v_index)

    # Control points from edges
    for edge_idx, geom in enumerate(edge_geoms):
        if geom.control_points:
            for cp in geom.control_points:
                cp_idx = len(control_points)
                control_points.append(cp)
                cp_to_edge.append((cp_idx, edge_idx))
                cp_to_edge_attr.append(cp.index)

    # Control points from faces
    for face_idx, geom in enumerate(face_geoms):
        if geom.control_points:
            for cp in geom.control_points:
                cp_idx = len(control_points)
                control_points.append(cp)
                cp_to_face.append((cp_idx, face_idx))
                cp_to_face_attr.append(cp.index)

    # Build HeteroData
    data = HeteroData()

    # Node features
    data['vertex'].x = build_vertex_features(vertex_geoms)
    data['edge'].x = build_edge_features(edge_geoms)
    data['face'].x = build_face_features(face_geoms)
    data['control_point'].x = build_control_point_features(control_points)

    # Store knot vectors as auxiliary data (variable length, stored as lists)
    edge_knots = []
    edge_multiplicities = []
    for geom in edge_geoms:
        edge_knots.append(geom.knots if geom.knots else [])
        edge_multiplicities.append(geom.multiplicities if geom.multiplicities else [])
    data['edge'].knots = edge_knots
    data['edge'].multiplicities = edge_multiplicities

    face_u_knots = []
    face_v_knots = []
    face_u_multiplicities = []
    face_v_multiplicities = []
    for geom in face_geoms:
        face_u_knots.append(geom.u_knots if geom.u_knots else [])
        face_v_knots.append(geom.v_knots if geom.v_knots else [])
        face_u_multiplicities.append(geom.u_multiplicities if geom.u_multiplicities else [])
        face_v_multiplicities.append(geom.v_multiplicities if geom.v_multiplicities else [])
    data['face'].u_knots = face_u_knots
    data['face'].v_knots = face_v_knots
    data['face'].u_multiplicities = face_u_multiplicities
    data['face'].v_multiplicities = face_v_multiplicities

    # Topology edge indices
    data['vertex', 'bounds', 'edge'].edge_index = build_edge_index(topo.vertex_to_edge)
    data['edge', 'bounds', 'face'].edge_index = build_edge_index(topo.edge_to_face)
    data['face', 'adjacent', 'face'].edge_index = build_edge_index(topo.face_to_face)

    # Control point edge indices
    data['control_point', 'controls', 'edge'].edge_index = build_edge_index(cp_to_edge)
    data['control_point', 'controls', 'face'].edge_index = build_edge_index(cp_to_face)

    # Control point edge attributes (indices for ordering)
    if cp_to_edge_attr:
        # For curves: single sequence index
        data['control_point', 'controls', 'edge'].edge_attr = torch.tensor(
            [idx[0] for idx in cp_to_edge_attr], dtype=torch.long
        ).unsqueeze(1)
    else:
        data['control_point', 'controls', 'edge'].edge_attr = torch.empty((0, 1), dtype=torch.long)

    if cp_to_face_attr:
        # For surfaces: (u_index, v_index)
        data['control_point', 'controls', 'face'].edge_attr = torch.tensor(
            cp_to_face_attr, dtype=torch.long
        )
    else:
        data['control_point', 'controls', 'face'].edge_attr = torch.empty((0, 2), dtype=torch.long)

    return data
