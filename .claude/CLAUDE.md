# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --python 3.11                # Install (requires Python 3.10-3.12)
uv run pytest src/cq2pyg/tests/ -v   # Run all tests
```

Use `/test` for convenient test running with options.

## Architecture

Converts CadQuery B-Rep CAD objects to PyTorch Geometric heterogeneous graphs.

**Pipeline:** `topology.py` → `geometry.py` → `features.py` → `converter.py`

**Key Design:** NURBS control points are graph nodes (not padded tensors) with `controls` edges to parent curves/surfaces.

**OCP Note:** Iterate `TopTools_ListOfShape` with Python protocol (`for item in list`), not `list.Value(i)`.
