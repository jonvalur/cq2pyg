# cq2pyg

Convert CadQuery B-Rep objects to PyTorch Geometric graphs for machine learning.

## Installation

```bash
pip install -e .
```

Requires Python 3.10-3.12 (due to CadQuery/VTK dependencies).

## Usage

```python
import cadquery as cq
from cq2pyg import cadquery_to_pyg

# Create a shape
box = cq.Workplane("XY").box(10, 10, 10)

# Convert to PyG HeteroData
data = cadquery_to_pyg(box)

# Access node features
print(data['vertex'].x)        # [8, 3] - vertex coordinates
print(data['edge'].x)          # [12, 24] - curve features
print(data['face'].x)          # [6, 34] - surface features
print(data['control_point'].x) # [N, 4] - B-spline control points

# Access relationships
print(data['vertex', 'bounds', 'edge'].edge_index)
print(data['edge', 'bounds', 'face'].edge_index)
print(data['face', 'adjacent', 'face'].edge_index)
```

## Graph Structure

The output is a PyG `HeteroData` graph with:

**Node Types:**
- `vertex` - Geometric vertices (x, y, z coordinates)
- `edge` - Topological edges/curves (type, orientation, parameters)
- `face` - Topological faces/surfaces (type, orientation, parameters)
- `control_point` - B-spline/NURBS control points (x, y, z, weight)

**Edge Types (Relationships):**
- `('vertex', 'bounds', 'edge')` - Vertex is endpoint of edge
- `('edge', 'bounds', 'face')` - Edge is part of face boundary
- `('face', 'adjacent', 'face')` - Faces share an edge
- `('control_point', 'controls', 'edge')` - Control point belongs to curve
- `('control_point', 'controls', 'face')` - Control point belongs to surface

## License

MIT
