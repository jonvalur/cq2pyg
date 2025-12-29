# cq2pyg

Convert CadQuery B-Rep CAD objects to PyTorch Geometric heterogeneous graphs.

## Installation

```bash
uv sync --python 3.11   # or pip install -e .
```

Requires Python 3.10-3.12.

## Usage

```python
import cadquery as cq
from cq2pyg import cadquery_to_pyg

box = cq.Workplane("XY").box(10, 10, 10)
data = cadquery_to_pyg(box)

# Node features
data['vertex'].x         # [8, 3]  - (x, y, z)
data['edge'].x           # [12, 24] - curve type + params
data['face'].x           # [6, 34]  - surface type + params
data['control_point'].x  # [N, 4]  - NURBS control points

# Relationships
data['vertex', 'bounds', 'edge'].edge_index
data['edge', 'bounds', 'face'].edge_index
data['face', 'adjacent', 'face'].edge_index
data['control_point', 'controls', 'edge'].edge_index
data['control_point', 'controls', 'face'].edge_index
```

## Graph Structure

| Node Type | Features | Description |
|-----------|----------|-------------|
| `vertex` | 3 | x, y, z coordinates |
| `edge` | 24 | Curve type (one-hot), orientation, degree, parameters |
| `face` | 34 | Surface type (one-hot), orientation, degrees, parameters |
| `control_point` | 4 | x, y, z, weight (for NURBS curves/surfaces) |

## License

MIT
