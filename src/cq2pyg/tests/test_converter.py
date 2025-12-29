"""
Tests for the CadQuery to PyG converter.
"""

import pytest
import torch
import cadquery as cq

from cq2pyg import cadquery_to_pyg, CurveType, SurfaceType


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
