"""
cq2pyg: Convert CadQuery B-Rep objects to PyTorch Geometric graphs.

This package provides lossless conversion of CadQuery shapes to PyG HeteroData
graphs, preserving full topological and geometric information.
"""

from .converter import cadquery_to_pyg, cadquery_to_pyg_simple
from .types import CurveType, SurfaceType

__version__ = "0.1.0"
__all__ = ["cadquery_to_pyg", "cadquery_to_pyg_simple", "CurveType", "SurfaceType"]
