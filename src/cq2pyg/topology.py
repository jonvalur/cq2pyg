"""
Topology extraction from CadQuery/OpenCASCADE shapes.

Extracts vertices, edges, and faces along with their relationships.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Vertex, TopoDS_Edge, TopoDS_Face, TopoDS_Wire
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape


@dataclass
class TopologyData:
    """Container for extracted topology data."""

    # Maps from shape hash to index
    vertex_map: Dict[int, int] = field(default_factory=dict)
    edge_map: Dict[int, int] = field(default_factory=dict)
    face_map: Dict[int, int] = field(default_factory=dict)

    # Lists of shapes (indexed by their map values)
    vertices: List[TopoDS_Vertex] = field(default_factory=list)
    edges: List[TopoDS_Edge] = field(default_factory=list)
    faces: List[TopoDS_Face] = field(default_factory=list)

    # Relationships as (source_idx, target_idx) pairs
    vertex_to_edge: List[Tuple[int, int]] = field(default_factory=list)  # vertex bounds edge
    edge_to_face: List[Tuple[int, int]] = field(default_factory=list)    # edge bounds face
    face_to_face: List[Tuple[int, int]] = field(default_factory=list)    # faces are adjacent


def _shape_hash(shape: TopoDS_Shape) -> int:
    """Get a unique hash for a TopoDS_Shape based on its underlying TShape."""
    return shape.HashCode(2147483647)


def extract_topology(shape: TopoDS_Shape) -> TopologyData:
    """
    Extract all topological entities and relationships from a shape.

    Args:
        shape: A TopoDS_Shape (the wrapped OCP shape from CadQuery)

    Returns:
        TopologyData containing all vertices, edges, faces and their relationships
    """
    data = TopologyData()

    # Extract all vertices
    vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = TopoDS.Vertex_s(vertex_explorer.Current())
        h = _shape_hash(vertex)
        if h not in data.vertex_map:
            idx = len(data.vertices)
            data.vertex_map[h] = idx
            data.vertices.append(vertex)
        vertex_explorer.Next()

    # Extract all edges
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = TopoDS.Edge_s(edge_explorer.Current())
        h = _shape_hash(edge)
        if h not in data.edge_map:
            idx = len(data.edges)
            data.edge_map[h] = idx
            data.edges.append(edge)
        edge_explorer.Next()

    # Extract all faces
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = TopoDS.Face_s(face_explorer.Current())
        h = _shape_hash(face)
        if h not in data.face_map:
            idx = len(data.faces)
            data.face_map[h] = idx
            data.faces.append(face)
        face_explorer.Next()

    # Build vertex-to-edge relationships
    for edge_idx, edge in enumerate(data.edges):
        # Get vertices of this edge
        v_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while v_explorer.More():
            vertex = TopoDS.Vertex_s(v_explorer.Current())
            h = _shape_hash(vertex)
            if h in data.vertex_map:
                vertex_idx = data.vertex_map[h]
                data.vertex_to_edge.append((vertex_idx, edge_idx))
            v_explorer.Next()

    # Build edge-to-face relationships
    for face_idx, face in enumerate(data.faces):
        # Get edges of this face
        e_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while e_explorer.More():
            edge = TopoDS.Edge_s(e_explorer.Current())
            h = _shape_hash(edge)
            if h in data.edge_map:
                edge_idx = data.edge_map[h]
                data.edge_to_face.append((edge_idx, face_idx))
            e_explorer.Next()

    # Build face-to-face adjacency (faces sharing an edge)
    # Use TopExp.MapShapesAndAncestors to find which faces each edge belongs to
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    adjacency_set = set()  # To avoid duplicate pairs

    for i in range(1, edge_face_map.Extent() + 1):
        edge = edge_face_map.FindKey(i)
        face_list = edge_face_map.FindFromIndex(i)

        # Get all faces adjacent to this edge
        # OCP wraps TopTools_ListOfShape as a Python iterable
        adjacent_face_indices = []
        for face_shape in face_list:
            face = TopoDS.Face_s(face_shape)
            h = _shape_hash(face)
            if h in data.face_map:
                adjacent_face_indices.append(data.face_map[h])

        # Create adjacency pairs
        for i1 in range(len(adjacent_face_indices)):
            for i2 in range(i1 + 1, len(adjacent_face_indices)):
                f1, f2 = adjacent_face_indices[i1], adjacent_face_indices[i2]
                if (f1, f2) not in adjacency_set and (f2, f1) not in adjacency_set:
                    adjacency_set.add((f1, f2))
                    data.face_to_face.append((f1, f2))
                    data.face_to_face.append((f2, f1))  # Bidirectional

    return data
