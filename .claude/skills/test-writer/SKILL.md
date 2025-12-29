# Test Writer Skill

Write and maintain tests for the cq2pyg package.

## When to Use

Use this skill when:
- Adding new tests
- Modifying existing tests
- Fixing failing tests
- Adding test coverage for new features

## Testing Framework

- **Runner:** pytest via `uv run pytest`
- **Location:** `src/cq2pyg/tests/test_converter.py`
- **Style:** Class-based test organization with descriptive docstrings

## Test Structure

Tests are organized by category:

```python
class TestCategoryName:
    """Description of what this category tests."""

    def test_specific_thing(self):
        """What this specific test validates."""
        # 1. Create CadQuery shape
        shape = cq.Workplane("XY").box(10, 10, 10)

        # 2. Convert to PyG graph
        data = cadquery_to_pyg(shape)

        # 3. Assert expected properties
        assert data['vertex'].x.shape[0] == 8
```

## Writing Tests

### Required Imports

```python
import math
import pytest
import torch
import cadquery as cq

from cq2pyg import cadquery_to_pyg, CurveType, SurfaceType
from cq2pyg.types import (
    VERTEX_FEATURE_DIM, EDGE_FEATURE_DIM, FACE_FEATURE_DIM, CONTROL_POINT_FEATURE_DIM,
    NUM_CURVE_TYPES, NUM_SURFACE_TYPES
)
```

### What to Test

1. **Node counts** - Verify expected number of vertices, edges, faces
2. **Feature dimensions** - Check tensor shapes match constants
3. **Surface/curve types** - Verify one-hot encoded types are correct
4. **Geometry values** - Check extracted parameters (radius, normal, etc.)
5. **Topology relationships** - Validate edge_index connectivity
6. **Data integrity** - Ensure indices are within bounds, correct dtypes

### Accessing Features

```python
# Surface type (one-hot in first NUM_SURFACE_TYPES columns)
face_types = data['face'].x[:, :NUM_SURFACE_TYPES].argmax(dim=1)
has_plane = (face_types == SurfaceType.PLANE).any()

# Curve type
edge_types = data['edge'].x[:, :NUM_CURVE_TYPES].argmax(dim=1)

# Orientation (after type one-hot)
face_orientation = data['face'].x[:, NUM_SURFACE_TYPES]  # +1 or -1

# Geometry parameters (check features.py for exact positions)
# Face radius at: NUM_SURFACE_TYPES + 21
# Edge radius at: NUM_CURVE_TYPES + 14
```

### Common Assertions

```python
# Node counts
assert data['vertex'].x.shape[0] == 8
assert data['face'].x.shape[0] >= 6

# Feature dimensions
assert data['vertex'].x.shape[1] == VERTEX_FEATURE_DIM

# Type checking
assert (face_types == SurfaceType.PLANE).all()

# Relationship validity
assert data['vertex', 'bounds', 'edge'].edge_index[0].max() < data['vertex'].x.shape[0]

# Geometry values
assert abs(stored_radius - expected_radius) < 0.001

# Tensor dtypes
assert data['vertex'].x.dtype == torch.float32
assert data['edge', 'bounds', 'face'].edge_index.dtype == torch.long
```

## IMPORTANT: Update Test Documentation

After adding or modifying tests, you MUST update `.claude/commands/test.md`:

1. Add new test class to the "Test Suite Overview" section
2. Document what each new test validates
3. Keep the format consistent with existing entries:

```markdown
### TestNewCategory (N tests)
Description of category:
- `test_name` - Brief description of what it tests
```

## Running Tests

```bash
# All tests
uv run pytest src/cq2pyg/tests/test_converter.py -v

# Specific class
uv run pytest src/cq2pyg/tests/test_converter.py::TestClassName -v

# Specific test
uv run pytest src/cq2pyg/tests/test_converter.py::TestClassName::test_name -v

# Pattern matching
uv run pytest src/cq2pyg/tests/test_converter.py -k "pattern" -v
```

## CadQuery Tips

- `cq.Workplane("XY").box(w, h, d)` - Create box
- `cq.Workplane("XY").cylinder(height, radius)` - Create cylinder
- `cq.Workplane("XY").sphere(radius)` - Create sphere
- `.val()` - Get Shape from Workplane
- `.val().wrapped` - Get raw TopoDS_Shape
- `cq.Compound.makeCompound([shape1, shape2])` - Combine shapes
- `.edges().fillet(r)` - Add fillets
- `.edges().chamfer(d)` - Add chamfers
- `.shell(thickness)` - Create shell (negative = inward)
- `.union(other)`, `.intersect(other)`, `.cut(other)` - Boolean ops
