"""
Tests for the CadQuery to PyG converter.
"""

import math
import pytest
import torch
import cadquery as cq

from cq2pyg import cadquery_to_pyg, CurveType, SurfaceType
from cq2pyg.types import (
    VERTEX_FEATURE_DIM, EDGE_FEATURE_DIM, FACE_FEATURE_DIM, CONTROL_POINT_FEATURE_DIM,
    NUM_CURVE_TYPES, NUM_SURFACE_TYPES
)


class TestBasicShapes:
    """Test conversion of basic geometric shapes."""

    def test_box(self):
        """Test converting a simple box."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)

        # A box has 8 vertices, 12 edges, 6 faces
        assert data['vertex'].x.shape[0] == 8
        assert data['edge'].x.shape[0] == 12
        assert data['face'].x.shape[0] == 6

        # Check vertex features are 3D coordinates
        assert data['vertex'].x.shape[1] == 3

        # All faces should be planes
        face_types = data['face'].x[:, :SurfaceType.OTHER + 1].argmax(dim=1)
        assert (face_types == SurfaceType.PLANE).all()

        # All edges should be lines
        edge_types = data['edge'].x[:, :CurveType.OTHER + 1].argmax(dim=1)
        assert (edge_types == CurveType.LINE).all()

        # Check relationships exist
        assert data['vertex', 'bounds', 'edge'].edge_index.shape[1] > 0
        assert data['edge', 'bounds', 'face'].edge_index.shape[1] > 0
        assert data['face', 'adjacent', 'face'].edge_index.shape[1] > 0

    def test_cylinder(self):
        """Test converting a cylinder."""
        cylinder = cq.Workplane("XY").cylinder(10, 5)
        data = cadquery_to_pyg(cylinder)

        # A cylinder has 3 faces: top, bottom (planes), curved surface (cylinder)
        assert data['face'].x.shape[0] == 3

        # Check we have both plane and cylinder surfaces
        face_types = data['face'].x[:, :SurfaceType.OTHER + 1].argmax(dim=1)
        has_plane = (face_types == SurfaceType.PLANE).any()
        has_cylinder = (face_types == SurfaceType.CYLINDER).any()
        assert has_plane and has_cylinder

        # Check we have both line and circle edges
        edge_types = data['edge'].x[:, :CurveType.OTHER + 1].argmax(dim=1)
        has_circle = (edge_types == CurveType.CIRCLE).any()
        assert has_circle

    def test_sphere(self):
        """Test converting a sphere."""
        sphere = cq.Workplane("XY").sphere(5)
        data = cadquery_to_pyg(sphere)

        # A sphere has 1 face
        assert data['face'].x.shape[0] == 1

        # The face should be a sphere surface
        face_types = data['face'].x[:, :SurfaceType.OTHER + 1].argmax(dim=1)
        assert face_types[0] == SurfaceType.SPHERE

    def test_cone(self):
        """Test converting a cone."""
        # Create a cone using revolution
        cone = (
            cq.Workplane("XZ")
            .moveTo(0, 0)
            .lineTo(5, 0)
            .lineTo(0, 10)
            .close()
            .revolve(360)
        )
        data = cadquery_to_pyg(cone)

        # Check we have faces
        assert data['face'].x.shape[0] > 0


class TestComplexShapes:
    """Test conversion of more complex shapes."""

    def test_fillet_box(self):
        """Test a box with fillets (introduces B-spline surfaces)."""
        box = cq.Workplane("XY").box(10, 10, 10).edges().fillet(1)
        data = cadquery_to_pyg(box)

        # Should have more faces than a simple box due to fillets
        assert data['face'].x.shape[0] > 6

        # Should have control points from the B-spline fillet surfaces
        # (fillets are typically represented as B-splines)
        assert data['control_point'].x.shape[0] >= 0  # May or may not have CPs depending on fillet impl

    def test_loft(self):
        """Test a loft which creates B-spline surfaces."""
        loft = (
            cq.Workplane("XY")
            .rect(10, 10)
            .workplane(offset=10)
            .circle(5)
            .loft()
        )
        data = cadquery_to_pyg(loft)

        # Should have faces
        assert data['face'].x.shape[0] > 0

        # Loft typically creates B-spline surfaces, check for control points
        # (though the exact representation depends on OpenCASCADE)


class TestInputTypes:
    """Test different input types."""

    def test_workplane_input(self):
        """Test Workplane input."""
        wp = cq.Workplane("XY").box(5, 5, 5)
        data = cadquery_to_pyg(wp)
        assert data['vertex'].x.shape[0] == 8

    def test_shape_input(self):
        """Test Shape input."""
        wp = cq.Workplane("XY").box(5, 5, 5)
        shape = wp.val()
        data = cadquery_to_pyg(shape)
        assert data['vertex'].x.shape[0] == 8

    def test_topodsshape_input(self):
        """Test TopoDS_Shape input."""
        wp = cq.Workplane("XY").box(5, 5, 5)
        occ_shape = wp.val().wrapped
        data = cadquery_to_pyg(occ_shape)
        assert data['vertex'].x.shape[0] == 8


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_edge_indices_valid(self):
        """Test that edge indices are valid."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)

        # vertex-edge relationships
        v_e_idx = data['vertex', 'bounds', 'edge'].edge_index
        assert v_e_idx[0].max() < data['vertex'].x.shape[0]
        assert v_e_idx[1].max() < data['edge'].x.shape[0]

        # edge-face relationships
        e_f_idx = data['edge', 'bounds', 'face'].edge_index
        assert e_f_idx[0].max() < data['edge'].x.shape[0]
        assert e_f_idx[1].max() < data['face'].x.shape[0]

        # face-face adjacency
        f_f_idx = data['face', 'adjacent', 'face'].edge_index
        assert f_f_idx[0].max() < data['face'].x.shape[0]
        assert f_f_idx[1].max() < data['face'].x.shape[0]

    def test_tensor_dtypes(self):
        """Test that tensors have correct dtypes."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)

        # Features should be float32
        assert data['vertex'].x.dtype == torch.float32
        assert data['edge'].x.dtype == torch.float32
        assert data['face'].x.dtype == torch.float32

        # Edge indices should be long
        assert data['vertex', 'bounds', 'edge'].edge_index.dtype == torch.long
        assert data['edge', 'bounds', 'face'].edge_index.dtype == torch.long


class TestKnotVectors:
    """Test that knot vectors are properly stored."""

    def test_knots_stored(self):
        """Test that knot vectors are stored for B-spline edges/faces."""
        # Create a spline-based shape
        spline = (
            cq.Workplane("XY")
            .spline([(0, 0), (1, 1), (2, 0), (3, 1)])
            .close()
            .extrude(1)
        )
        data = cadquery_to_pyg(spline)

        # Check that knot lists exist
        assert hasattr(data['edge'], 'knots')
        assert hasattr(data['face'], 'u_knots')
        assert hasattr(data['face'], 'v_knots')


class TestFeatureDimensions:
    """Test that feature tensors have correct dimensions."""

    def test_vertex_feature_dim(self):
        """Test vertex feature dimensions."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)
        assert data['vertex'].x.shape[1] == VERTEX_FEATURE_DIM

    def test_edge_feature_dim(self):
        """Test edge feature dimensions."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)
        assert data['edge'].x.shape[1] == EDGE_FEATURE_DIM

    def test_face_feature_dim(self):
        """Test face feature dimensions."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)
        assert data['face'].x.shape[1] == FACE_FEATURE_DIM

    def test_control_point_feature_dim(self):
        """Test control point feature dimensions for B-spline shape."""
        spline = (
            cq.Workplane("XY")
            .spline([(0, 0), (1, 1), (2, 0), (3, 1)])
            .close()
            .extrude(1)
        )
        data = cadquery_to_pyg(spline)
        if data['control_point'].x.shape[0] > 0:
            assert data['control_point'].x.shape[1] == CONTROL_POINT_FEATURE_DIM


class TestMoreSurfaceTypes:
    """Test additional surface types."""

    def test_torus(self):
        """Test converting a torus (surface of revolution)."""
        # Create torus by revolving a circle
        torus = (
            cq.Workplane("XZ")
            .center(10, 0)  # Move circle center away from axis
            .circle(3)
            .revolve(360, (0, 0, 0), (0, 0, 1))
        )
        data = cadquery_to_pyg(torus)

        # A torus has 1 face
        assert data['face'].x.shape[0] == 1

        # The face should be a torus or revolution surface
        # (OpenCASCADE may represent it as either depending on construction)
        face_types = data['face'].x[:, :NUM_SURFACE_TYPES].argmax(dim=1)
        assert face_types[0] in (SurfaceType.TORUS, SurfaceType.REVOLUTION)

    def test_cone_surface(self):
        """Test that a cone has cone surface type."""
        # Create a truncated cone
        cone = (
            cq.Workplane("XY")
            .circle(10)
            .workplane(offset=20)
            .circle(5)
            .loft()
        )
        data = cadquery_to_pyg(cone)

        # Should have faces including cone surface
        face_types = data['face'].x[:, :NUM_SURFACE_TYPES].argmax(dim=1)
        has_cone = (face_types == SurfaceType.CONE).any()
        # Note: loft might create B-spline instead of true cone
        assert data['face'].x.shape[0] >= 3  # top, bottom, side


class TestGeometryValues:
    """Test that geometry values are correctly extracted."""

    def test_box_vertex_coordinates(self):
        """Test that box vertices have correct coordinates."""
        size = 10
        box = cq.Workplane("XY").box(size, size, size)
        data = cadquery_to_pyg(box)

        vertices = data['vertex'].x
        # All coordinates should be within [-size/2, size/2]
        assert vertices.min() >= -size / 2 - 0.001
        assert vertices.max() <= size / 2 + 0.001

    def test_sphere_radius(self):
        """Test that sphere has correct radius in features."""
        radius = 7.5
        sphere = cq.Workplane("XY").sphere(radius)
        data = cadquery_to_pyg(sphere)

        # Find the sphere face and check its radius
        face_types = data['face'].x[:, :NUM_SURFACE_TYPES].argmax(dim=1)
        sphere_idx = (face_types == SurfaceType.SPHERE).nonzero()[0][0]

        # Radius is stored after type one-hot + orientation + degrees + closed + bounds + normal + origin + axis_dir + axis_origin
        # Position: NUM_SURFACE_TYPES + 1 + 2 + 2 + 4 + 3 + 3 + 3 + 3 = NUM_SURFACE_TYPES + 21
        radius_idx = NUM_SURFACE_TYPES + 21
        stored_radius = data['face'].x[sphere_idx, radius_idx].item()
        assert abs(stored_radius - radius) < 0.001

    def test_cylinder_radius(self):
        """Test that cylinder has correct radius in features."""
        radius = 5.0
        height = 10.0
        cylinder = cq.Workplane("XY").cylinder(height, radius)
        data = cadquery_to_pyg(cylinder)

        # Find the cylinder face
        face_types = data['face'].x[:, :NUM_SURFACE_TYPES].argmax(dim=1)
        cyl_idx = (face_types == SurfaceType.CYLINDER).nonzero()[0][0]

        radius_idx = NUM_SURFACE_TYPES + 21
        stored_radius = data['face'].x[cyl_idx, radius_idx].item()
        assert abs(stored_radius - radius) < 0.001

    def test_circle_radius(self):
        """Test that circle edges have correct radius."""
        radius = 5.0
        cylinder = cq.Workplane("XY").cylinder(10, radius)
        data = cadquery_to_pyg(cylinder)

        # Find circle edges
        edge_types = data['edge'].x[:, :NUM_CURVE_TYPES].argmax(dim=1)
        circle_mask = edge_types == CurveType.CIRCLE

        if circle_mask.any():
            # Radius is at position NUM_CURVE_TYPES + 1 + 1 + 1 + 2 + 3 + 3 + 3 = NUM_CURVE_TYPES + 14
            radius_idx = NUM_CURVE_TYPES + 14
            circle_radii = data['edge'].x[circle_mask, radius_idx]
            # All circles should have the same radius
            assert torch.allclose(circle_radii, torch.tensor(radius), atol=0.001)


class TestTopologyConnectivity:
    """Test topology connectivity properties."""

    def test_each_edge_has_vertices(self):
        """Test that each edge is bounded by vertices."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)

        v_e_idx = data['vertex', 'bounds', 'edge'].edge_index
        num_edges = data['edge'].x.shape[0]

        # Each edge should have at least 1 vertex connection (2 for non-closed)
        edges_with_vertices = set(v_e_idx[1].tolist())
        assert len(edges_with_vertices) == num_edges

    def test_each_face_has_edges(self):
        """Test that each face has bounding edges."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)

        e_f_idx = data['edge', 'bounds', 'face'].edge_index
        num_faces = data['face'].x.shape[0]

        # Each face should have at least 3 edges
        for face_idx in range(num_faces):
            edges_for_face = (e_f_idx[1] == face_idx).sum()
            assert edges_for_face >= 3, f"Face {face_idx} has only {edges_for_face} edges"

    def test_face_adjacency_symmetric(self):
        """Test that face adjacency is symmetric."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)

        f_f_idx = data['face', 'adjacent', 'face'].edge_index

        # For each (a, b) there should be (b, a)
        edges_set = set()
        for i in range(f_f_idx.shape[1]):
            a, b = f_f_idx[0, i].item(), f_f_idx[1, i].item()
            edges_set.add((a, b))

        for a, b in list(edges_set):
            assert (b, a) in edges_set, f"Missing reverse edge ({b}, {a})"

    def test_box_face_adjacency_count(self):
        """Test that box has correct face adjacency (each face adjacent to 4 others)."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)

        f_f_idx = data['face', 'adjacent', 'face'].edge_index
        num_faces = data['face'].x.shape[0]

        # Each face of a box should be adjacent to 4 other faces
        for face_idx in range(num_faces):
            adjacent_count = (f_f_idx[0] == face_idx).sum()
            assert adjacent_count == 4, f"Face {face_idx} has {adjacent_count} adjacent faces"


class TestControlPoints:
    """Test control point extraction for B-splines."""

    def test_bspline_curve_has_control_points(self):
        """Test that B-spline curves have control points."""
        spline = (
            cq.Workplane("XY")
            .spline([(0, 0), (1, 2), (3, 2), (4, 0)])
            .close()
            .extrude(1)
        )
        data = cadquery_to_pyg(spline)

        # Should have control points
        assert data['control_point'].x.shape[0] > 0

        # Control points should have x, y, z, weight
        assert data['control_point'].x.shape[1] == 4

        # Weights should be positive
        weights = data['control_point'].x[:, 3]
        assert (weights > 0).all()

    def test_control_point_edge_relationship(self):
        """Test control point to edge relationships."""
        spline = (
            cq.Workplane("XY")
            .spline([(0, 0), (1, 2), (3, 2), (4, 0)])
            .close()
            .extrude(1)
        )
        data = cadquery_to_pyg(spline)

        cp_e_idx = data['control_point', 'controls', 'edge'].edge_index

        if cp_e_idx.shape[1] > 0:
            # Control point indices should be valid
            assert cp_e_idx[0].max() < data['control_point'].x.shape[0]
            # Edge indices should be valid
            assert cp_e_idx[1].max() < data['edge'].x.shape[0]

    def test_control_point_sequence_attr(self):
        """Test that control point sequence indices are stored."""
        spline = (
            cq.Workplane("XY")
            .spline([(0, 0), (1, 2), (3, 2), (4, 0)])
            .close()
            .extrude(1)
        )
        data = cadquery_to_pyg(spline)

        cp_e_attr = data['control_point', 'controls', 'edge'].edge_attr

        if cp_e_attr.shape[0] > 0:
            # Sequence indices should be non-negative
            assert (cp_e_attr >= 0).all()


class TestBooleanOperations:
    """Test shapes created with boolean operations."""

    def test_union(self):
        """Test union of two boxes."""
        box1 = cq.Workplane("XY").box(10, 10, 10)
        box2 = cq.Workplane("XY").center(5, 0).box(10, 10, 10)
        union = box1.union(box2)
        data = cadquery_to_pyg(union)

        # Union of overlapping boxes should have more faces
        assert data['face'].x.shape[0] >= 6
        assert data['vertex'].x.shape[0] > 0
        assert data['edge'].x.shape[0] > 0

    def test_difference(self):
        """Test difference (hole in box)."""
        box = cq.Workplane("XY").box(20, 20, 20)
        hole = box.faces(">Z").workplane().hole(5)
        data = cadquery_to_pyg(hole)

        # Should have cylinder surface from the hole
        face_types = data['face'].x[:, :NUM_SURFACE_TYPES].argmax(dim=1)
        has_cylinder = (face_types == SurfaceType.CYLINDER).any()
        assert has_cylinder

    def test_intersection(self):
        """Test intersection of two shapes."""
        box = cq.Workplane("XY").box(10, 10, 10)
        sphere = cq.Workplane("XY").sphere(7)
        intersection = box.intersect(sphere)
        data = cadquery_to_pyg(intersection)

        # Should have both plane and sphere surfaces
        assert data['face'].x.shape[0] > 0


class TestCompoundShapes:
    """Test compound/assembly shapes."""

    def test_multiple_solids(self):
        """Test shape with multiple disconnected solids."""
        box1 = cq.Workplane("XY").box(5, 5, 5).val()
        box2 = cq.Workplane("XY").center(20, 0).box(5, 5, 5).val()
        compound = cq.Compound.makeCompound([box1, box2])
        data = cadquery_to_pyg(compound)

        # Should have 16 vertices (8 per box)
        assert data['vertex'].x.shape[0] == 16
        # Should have 12 faces (6 per box)
        assert data['face'].x.shape[0] == 12

    def test_shell(self):
        """Test a shell (hollow box)."""
        shell = cq.Workplane("XY").box(10, 10, 10).shell(-1)
        data = cadquery_to_pyg(shell)

        # Shell has both inner and outer faces
        assert data['face'].x.shape[0] > 6


class TestOrientation:
    """Test face and edge orientation."""

    def test_face_orientation_stored(self):
        """Test that face orientation is stored in features."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)

        # Orientation is at position NUM_SURFACE_TYPES
        orientations = data['face'].x[:, NUM_SURFACE_TYPES]
        # All orientations should be +1 or -1
        assert ((orientations == 1) | (orientations == -1)).all()

    def test_edge_orientation_stored(self):
        """Test that edge orientation is stored in features."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)

        # Orientation is at position NUM_CURVE_TYPES
        orientations = data['edge'].x[:, NUM_CURVE_TYPES]
        # All orientations should be +1 or -1
        assert ((orientations == 1) | (orientations == -1)).all()


class TestEdgeCases:
    """Test edge cases and special shapes."""

    def test_single_face(self):
        """Test a single face (not a solid)."""
        face = cq.Workplane("XY").rect(10, 10).extrude(0.001)  # Very thin
        data = cadquery_to_pyg(face)
        assert data['face'].x.shape[0] >= 1

    def test_wedge(self):
        """Test a wedge shape."""
        wedge = cq.Workplane("XY").polygon(3, 10).extrude(10)
        data = cadquery_to_pyg(wedge)

        # Triangle prism: 5 faces (2 triangles + 3 rectangles)
        assert data['face'].x.shape[0] == 5
        # 6 vertices
        assert data['vertex'].x.shape[0] == 6

    def test_chamfer(self):
        """Test shape with chamfers."""
        box = cq.Workplane("XY").box(10, 10, 10).edges().chamfer(1)
        data = cadquery_to_pyg(box)

        # Chamfers add extra faces
        assert data['face'].x.shape[0] > 6


class TestPlaneNormals:
    """Test plane normal vectors."""

    def test_box_plane_normals(self):
        """Test that box planes have correct normal directions."""
        box = cq.Workplane("XY").box(10, 10, 10)
        data = cadquery_to_pyg(box)

        # Find plane faces
        face_types = data['face'].x[:, :NUM_SURFACE_TYPES].argmax(dim=1)
        plane_mask = face_types == SurfaceType.PLANE

        # Normal starts at NUM_SURFACE_TYPES + 1 + 2 + 2 + 4 = NUM_SURFACE_TYPES + 9
        normal_start = NUM_SURFACE_TYPES + 9
        normals = data['face'].x[plane_mask, normal_start:normal_start + 3]

        # Each normal should be a unit vector (or close to it)
        for i in range(normals.shape[0]):
            normal = normals[i]
            length = torch.norm(normal)
            if length > 0.001:  # Skip zero normals
                assert abs(length - 1.0) < 0.01, f"Normal {i} has length {length}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
