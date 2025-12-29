"""
Geometry extraction from CadQuery/OpenCASCADE shapes.

Extracts geometric parameters from vertices, edges (curves), and faces (surfaces).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import (
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
    GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OffsetCurve,
    GeomAbs_OtherCurve,
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface
)
from OCP.TopoDS import TopoDS_Vertex, TopoDS_Edge, TopoDS_Face
from OCP.gp import gp_Pnt, gp_Vec, gp_Dir

from .types import CurveType, SurfaceType


# Map OpenCASCADE curve types to our enum
CURVE_TYPE_MAP = {
    GeomAbs_Line: CurveType.LINE,
    GeomAbs_Circle: CurveType.CIRCLE,
    GeomAbs_Ellipse: CurveType.ELLIPSE,
    GeomAbs_Hyperbola: CurveType.HYPERBOLA,
    GeomAbs_Parabola: CurveType.PARABOLA,
    GeomAbs_BezierCurve: CurveType.BEZIER,
    GeomAbs_BSplineCurve: CurveType.BSPLINE,
    GeomAbs_OffsetCurve: CurveType.OFFSET,
    GeomAbs_OtherCurve: CurveType.OTHER,
}

# Map OpenCASCADE surface types to our enum
SURFACE_TYPE_MAP = {
    GeomAbs_Plane: SurfaceType.PLANE,
    GeomAbs_Cylinder: SurfaceType.CYLINDER,
    GeomAbs_Cone: SurfaceType.CONE,
    GeomAbs_Sphere: SurfaceType.SPHERE,
    GeomAbs_Torus: SurfaceType.TORUS,
    GeomAbs_BezierSurface: SurfaceType.BEZIER,
    GeomAbs_BSplineSurface: SurfaceType.BSPLINE,
    GeomAbs_SurfaceOfRevolution: SurfaceType.REVOLUTION,
    GeomAbs_SurfaceOfExtrusion: SurfaceType.EXTRUSION,
    GeomAbs_OffsetSurface: SurfaceType.OFFSET,
    GeomAbs_OtherSurface: SurfaceType.OTHER,
}


@dataclass
class VertexGeometry:
    """Geometry data for a vertex."""
    x: float
    y: float
    z: float


@dataclass
class ControlPoint:
    """A control point for B-spline curves/surfaces."""
    x: float
    y: float
    z: float
    weight: float
    # For curves: sequence index (0, 1, 2, ...)
    # For surfaces: (u_index, v_index)
    index: Tuple[int, ...]


@dataclass
class EdgeGeometry:
    """Geometry data for an edge (curve)."""
    curve_type: CurveType
    orientation: int  # +1 forward, -1 reversed
    degree: int
    is_closed: bool
    t_min: float
    t_max: float

    # Line parameters
    line_direction: Optional[Tuple[float, float, float]] = None

    # Circle/ellipse parameters
    center: Optional[Tuple[float, float, float]] = None
    axis: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    minor_radius: Optional[float] = None  # For ellipse

    # B-spline specific
    knots: Optional[List[float]] = None
    multiplicities: Optional[List[int]] = None
    control_points: Optional[List[ControlPoint]] = None


@dataclass
class FaceGeometry:
    """Geometry data for a face (surface)."""
    surface_type: SurfaceType
    orientation: int  # +1 outward normal, -1 reversed
    u_degree: int
    v_degree: int
    is_u_closed: bool
    is_v_closed: bool
    u_min: float
    u_max: float
    v_min: float
    v_max: float

    # Plane parameters
    plane_normal: Optional[Tuple[float, float, float]] = None
    plane_origin: Optional[Tuple[float, float, float]] = None

    # Cylinder/cone/sphere/torus parameters
    axis_direction: Optional[Tuple[float, float, float]] = None
    axis_origin: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    radius2: Optional[float] = None  # Minor radius for torus, or second radius

    # Cone specific
    half_angle: Optional[float] = None

    # B-spline specific
    u_knots: Optional[List[float]] = None
    v_knots: Optional[List[float]] = None
    u_multiplicities: Optional[List[int]] = None
    v_multiplicities: Optional[List[int]] = None
    control_points: Optional[List[ControlPoint]] = None  # Flattened grid


def _pnt_to_tuple(pnt: gp_Pnt) -> Tuple[float, float, float]:
    """Convert gp_Pnt to tuple."""
    return (pnt.X(), pnt.Y(), pnt.Z())


def _dir_to_tuple(d: gp_Dir) -> Tuple[float, float, float]:
    """Convert gp_Dir to tuple."""
    return (d.X(), d.Y(), d.Z())


def extract_vertex_geometry(vertex: TopoDS_Vertex) -> VertexGeometry:
    """Extract geometry from a vertex."""
    pnt = BRep_Tool.Pnt_s(vertex)
    return VertexGeometry(x=pnt.X(), y=pnt.Y(), z=pnt.Z())


def extract_edge_geometry(edge: TopoDS_Edge) -> EdgeGeometry:
    """Extract geometry from an edge (curve)."""
    adaptor = BRepAdaptor_Curve(edge)

    occ_type = adaptor.GetType()
    curve_type = CURVE_TYPE_MAP.get(occ_type, CurveType.OTHER)

    # Basic properties
    orientation = 1 if edge.Orientation() == 0 else -1  # TopAbs_FORWARD = 0
    t_min = adaptor.FirstParameter()
    t_max = adaptor.LastParameter()
    is_closed = adaptor.IsClosed()

    geom = EdgeGeometry(
        curve_type=curve_type,
        orientation=orientation,
        degree=1,  # Default, updated for specific types
        is_closed=is_closed,
        t_min=t_min,
        t_max=t_max,
    )

    # Type-specific parameters
    if curve_type == CurveType.LINE:
        line = adaptor.Line()
        geom.degree = 1
        geom.line_direction = _dir_to_tuple(line.Direction())

    elif curve_type == CurveType.CIRCLE:
        circle = adaptor.Circle()
        geom.degree = 2
        geom.center = _pnt_to_tuple(circle.Location())
        geom.axis = _dir_to_tuple(circle.Axis().Direction())
        geom.radius = circle.Radius()

    elif curve_type == CurveType.ELLIPSE:
        ellipse = adaptor.Ellipse()
        geom.degree = 2
        geom.center = _pnt_to_tuple(ellipse.Location())
        geom.axis = _dir_to_tuple(ellipse.Axis().Direction())
        geom.radius = ellipse.MajorRadius()
        geom.minor_radius = ellipse.MinorRadius()

    elif curve_type == CurveType.BSPLINE:
        bspline = adaptor.BSpline()
        geom.degree = bspline.Degree()

        # Extract knots
        geom.knots = [bspline.Knot(i) for i in range(1, bspline.NbKnots() + 1)]
        geom.multiplicities = [bspline.Multiplicity(i) for i in range(1, bspline.NbKnots() + 1)]

        # Extract control points
        geom.control_points = []
        for i in range(1, bspline.NbPoles() + 1):
            pole = bspline.Pole(i)
            weight = bspline.Weight(i) if bspline.IsRational() else 1.0
            geom.control_points.append(ControlPoint(
                x=pole.X(), y=pole.Y(), z=pole.Z(),
                weight=weight,
                index=(i - 1,)  # 0-indexed
            ))

    elif curve_type == CurveType.BEZIER:
        bezier = adaptor.Bezier()
        geom.degree = bezier.Degree()

        # Extract control points
        geom.control_points = []
        for i in range(1, bezier.NbPoles() + 1):
            pole = bezier.Pole(i)
            weight = bezier.Weight(i) if bezier.IsRational() else 1.0
            geom.control_points.append(ControlPoint(
                x=pole.X(), y=pole.Y(), z=pole.Z(),
                weight=weight,
                index=(i - 1,)
            ))

    return geom


def extract_face_geometry(face: TopoDS_Face) -> FaceGeometry:
    """Extract geometry from a face (surface)."""
    adaptor = BRepAdaptor_Surface(face)

    occ_type = adaptor.GetType()
    surface_type = SURFACE_TYPE_MAP.get(occ_type, SurfaceType.OTHER)

    # Basic properties
    orientation = 1 if face.Orientation() == 0 else -1  # TopAbs_FORWARD = 0
    u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()
    is_u_closed = adaptor.IsUClosed()
    is_v_closed = adaptor.IsVClosed()

    geom = FaceGeometry(
        surface_type=surface_type,
        orientation=orientation,
        u_degree=1,
        v_degree=1,
        is_u_closed=is_u_closed,
        is_v_closed=is_v_closed,
        u_min=u_min,
        u_max=u_max,
        v_min=v_min,
        v_max=v_max,
    )

    # Type-specific parameters
    if surface_type == SurfaceType.PLANE:
        plane = adaptor.Plane()
        geom.u_degree = 1
        geom.v_degree = 1
        geom.plane_normal = _dir_to_tuple(plane.Axis().Direction())
        geom.plane_origin = _pnt_to_tuple(plane.Location())

    elif surface_type == SurfaceType.CYLINDER:
        cylinder = adaptor.Cylinder()
        geom.u_degree = 2
        geom.v_degree = 1
        geom.axis_direction = _dir_to_tuple(cylinder.Axis().Direction())
        geom.axis_origin = _pnt_to_tuple(cylinder.Location())
        geom.radius = cylinder.Radius()

    elif surface_type == SurfaceType.CONE:
        cone = adaptor.Cone()
        geom.u_degree = 2
        geom.v_degree = 1
        geom.axis_direction = _dir_to_tuple(cone.Axis().Direction())
        geom.axis_origin = _pnt_to_tuple(cone.Location())
        geom.radius = cone.RefRadius()
        geom.half_angle = cone.SemiAngle()

    elif surface_type == SurfaceType.SPHERE:
        sphere = adaptor.Sphere()
        geom.u_degree = 2
        geom.v_degree = 2
        geom.axis_direction = _dir_to_tuple(sphere.Position().Direction())
        geom.axis_origin = _pnt_to_tuple(sphere.Location())
        geom.radius = sphere.Radius()

    elif surface_type == SurfaceType.TORUS:
        torus = adaptor.Torus()
        geom.u_degree = 2
        geom.v_degree = 2
        geom.axis_direction = _dir_to_tuple(torus.Axis().Direction())
        geom.axis_origin = _pnt_to_tuple(torus.Location())
        geom.radius = torus.MajorRadius()
        geom.radius2 = torus.MinorRadius()

    elif surface_type == SurfaceType.BSPLINE:
        bspline = adaptor.BSpline()
        geom.u_degree = bspline.UDegree()
        geom.v_degree = bspline.VDegree()

        # Extract U knots
        geom.u_knots = [bspline.UKnot(i) for i in range(1, bspline.NbUKnots() + 1)]
        geom.u_multiplicities = [bspline.UMultiplicity(i) for i in range(1, bspline.NbUKnots() + 1)]

        # Extract V knots
        geom.v_knots = [bspline.VKnot(i) for i in range(1, bspline.NbVKnots() + 1)]
        geom.v_multiplicities = [bspline.VMultiplicity(i) for i in range(1, bspline.NbVKnots() + 1)]

        # Extract control points (2D grid flattened)
        geom.control_points = []
        for ui in range(1, bspline.NbUPoles() + 1):
            for vi in range(1, bspline.NbVPoles() + 1):
                pole = bspline.Pole(ui, vi)
                weight = bspline.Weight(ui, vi) if bspline.IsURational() or bspline.IsVRational() else 1.0
                geom.control_points.append(ControlPoint(
                    x=pole.X(), y=pole.Y(), z=pole.Z(),
                    weight=weight,
                    index=(ui - 1, vi - 1)  # 0-indexed
                ))

    elif surface_type == SurfaceType.BEZIER:
        bezier = adaptor.Bezier()
        geom.u_degree = bezier.UDegree()
        geom.v_degree = bezier.VDegree()

        # Extract control points (2D grid flattened)
        geom.control_points = []
        for ui in range(1, bezier.NbUPoles() + 1):
            for vi in range(1, bezier.NbVPoles() + 1):
                pole = bezier.Pole(ui, vi)
                weight = bezier.Weight(ui, vi) if bezier.IsURational() or bezier.IsVRational() else 1.0
                geom.control_points.append(ControlPoint(
                    x=pole.X(), y=pole.Y(), z=pole.Z(),
                    weight=weight,
                    index=(ui - 1, vi - 1)
                ))

    return geom
