---
description: Run cq2pyg test suite
allowed-tools: Bash(uv:*), Bash(pytest:*), Read, Grep
argument-hint: [test-pattern]
---

# Test Runner for cq2pyg

Run the test suite using uv and pytest.

## Usage

- `/test` - Run all tests
- `/test TestBasicShapes` - Run a specific test class
- `/test test_box` - Run a specific test method
- `/test -k "box or cylinder"` - Run tests matching pattern

## Command

```bash
uv run pytest src/cq2pyg/tests/test_converter.py $ARGUMENTS -v
```

If no arguments provided, run all tests with verbose output.

## Test Suite Overview

### TestBasicShapes (4 tests)
Tests conversion of primitive CAD shapes:
- `test_box` - 10x10x10 box: verifies 8 vertices, 12 edges, 6 plane faces, all line edges
- `test_cylinder` - Cylinder: checks for plane + cylinder surfaces, circle edges
- `test_sphere` - Sphere: single spherical face
- `test_cone` - Revolved cone: validates face generation

### TestComplexShapes (2 tests)
Tests shapes with B-spline geometry:
- `test_fillet_box` - Box with filleted edges: more than 6 faces, potential control points
- `test_loft` - Loft between rectangle and circle: B-spline surface generation

### TestInputTypes (3 tests)
Tests different CadQuery input types:
- `test_workplane_input` - Direct Workplane object
- `test_shape_input` - cq.Shape via .val()
- `test_topodsshape_input` - Raw OCP TopoDS_Shape via .wrapped

### TestDataIntegrity (2 tests)
Tests graph data validity:
- `test_edge_indices_valid` - All edge_index values within bounds
- `test_tensor_dtypes` - Features are float32, indices are long

### TestKnotVectors (1 test)
- `test_knots_stored` - B-spline shapes store knot vectors as list attributes

### TestFeatureDimensions (4 tests)
Tests tensor shapes match constants:
- `test_vertex_feature_dim` - Vertex features: 3 (x, y, z)
- `test_edge_feature_dim` - Edge features: 24 (type + params)
- `test_face_feature_dim` - Face features: 34 (type + params)
- `test_control_point_feature_dim` - Control points: 4 (x, y, z, weight)

### TestMoreSurfaceTypes (2 tests)
Tests additional surface types:
- `test_torus` - Revolved circle: torus or revolution surface
- `test_cone_surface` - Truncated cone via loft

### TestGeometryValues (4 tests)
Tests extracted geometric parameters:
- `test_box_vertex_coordinates` - Vertices within [-5, 5] for 10-unit box
- `test_sphere_radius` - Sphere radius=7.5 stored correctly
- `test_cylinder_radius` - Cylinder radius=5.0 stored correctly
- `test_circle_radius` - Circle edge radius matches cylinder

### TestTopologyConnectivity (4 tests)
Tests B-Rep relationships:
- `test_each_edge_has_vertices` - All edges connected to vertices
- `test_each_face_has_edges` - Each face has ≥3 bounding edges
- `test_face_adjacency_symmetric` - If (a,b) adjacent, (b,a) exists
- `test_box_face_adjacency_count` - Each box face adjacent to 4 others

### TestControlPoints (3 tests)
Tests B-spline control point extraction:
- `test_bspline_curve_has_control_points` - Spline curves have CPs with positive weights
- `test_control_point_edge_relationship` - CP→edge indices valid
- `test_control_point_sequence_attr` - Sequence indices are non-negative

### TestBooleanOperations (3 tests)
Tests CAD boolean operations:
- `test_union` - Union of overlapping boxes
- `test_difference` - Hole creates cylinder surface
- `test_intersection` - Box ∩ sphere

### TestCompoundShapes (2 tests)
Tests multi-body shapes:
- `test_multiple_solids` - Two separate boxes: 16 vertices, 12 faces
- `test_shell` - Hollow box has >6 faces

### TestOrientation (2 tests)
Tests orientation flags:
- `test_face_orientation_stored` - Face orientation is +1 or -1
- `test_edge_orientation_stored` - Edge orientation is +1 or -1

### TestEdgeCases (3 tests)
Tests special shapes:
- `test_single_face` - Very thin extrusion
- `test_wedge` - Triangular prism: 5 faces, 6 vertices
- `test_chamfer` - Chamfered box has >6 faces

### TestPlaneNormals (1 test)
- `test_box_plane_normals` - Plane normals are unit vectors
