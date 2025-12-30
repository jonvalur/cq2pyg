# cq2pyg

Convert CadQuery B-Rep CAD objects to PyTorch Geometric heterogeneous graphs.

## Installation

```bash
pip install git+https://github.com/jonvalur/cq2pyg.git
```

Requires Python 3.10-3.12, CadQuery >= 2.6.1.

## Usage

```python
import cadquery as cq
from cq2pyg import cadquery_to_pyg

box = cq.Workplane("XY").box(10, 10, 10)
data = cadquery_to_pyg(box)
print(data)
```

```
HeteroData(
  vertex={ x=[8, 3] },
  edge={ x=[12, 24], knots=[12], multiplicities=[12] },
  face={ x=[6, 34], u_knots=[6], v_knots=[6], ... },
  control_point={ x=[0, 4] },
  (vertex, bounds, edge)={ edge_index=[2, 24] },
  (edge, bounds, face)={ edge_index=[2, 24] },
  (face, adjacent, face)={ edge_index=[2, 24] },
  ...
)
```

## Graph Structure

### Nodes

| Type | Dim | Description |
|------|-----|-------------|
| `vertex` | 3 | x, y, z coordinates |
| `edge` | 24 | Curve geometry (see below) |
| `face` | 34 | Surface geometry (see below) |
| `control_point` | 4 | x, y, z, weight (NURBS only) |

### Edges (Relationships)

| Relationship | Meaning |
|--------------|---------|
| `(vertex, bounds, edge)` | Vertices bounding each edge |
| `(edge, bounds, face)` | Edges bounding each face |
| `(face, adjacent, face)` | Faces sharing an edge |
| `(control_point, controls, edge)` | Control points defining B-spline curves |
| `(control_point, controls, face)` | Control points defining B-spline surfaces |

## Feature Layouts

### Edge Features (24)

| Index | Feature |
|-------|---------|
| 0-8 | Curve type one-hot: LINE, CIRCLE, ELLIPSE, HYPERBOLA, PARABOLA, BEZIER, BSPLINE, OFFSET, OTHER |
| 9 | Orientation (+1/-1) |
| 10 | Degree |
| 11 | Is closed (0/1) |
| 12-13 | t_min, t_max (parameter bounds) |
| 14-16 | Line direction (dx, dy, dz) |
| 17-19 | Center (cx, cy, cz) |
| 20-22 | Axis (ax, ay, az) |
| 23 | Radius |

### Face Features (34)

| Index | Feature |
|-------|---------|
| 0-10 | Surface type one-hot: PLANE, CYLINDER, CONE, SPHERE, TORUS, BEZIER, BSPLINE, REVOLUTION, EXTRUSION, OFFSET, OTHER |
| 11 | Orientation (+1/-1) |
| 12-13 | u_degree, v_degree |
| 14-15 | is_u_closed, is_v_closed |
| 16-19 | u_min, u_max, v_min, v_max |
| 20-22 | Plane normal (nx, ny, nz) |
| 23-25 | Plane origin (ox, oy, oz) |
| 26-28 | Axis direction |
| 29-31 | Axis origin |
| 32-33 | radius, radius2 |

### Knot Vectors (B-splines only)

Stored as list attributes (variable length per edge/face):

- `edge.knots` / `edge.multiplicities` — knot vector for B-spline curves
- `face.u_knots`, `face.v_knots` — knot vectors for B-spline surfaces
- `face.u_multiplicities`, `face.v_multiplicities` — corresponding multiplicities

## License

MIT
