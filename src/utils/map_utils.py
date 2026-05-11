import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.transform import Rotation
from shapely.ops import nearest_points
from dataclasses import dataclass

@dataclass
class EdgeProjectionResult:
    projected_point: Point
    projected_lat: float
    projected_lon: float
    distance_m: float
    edge_index: int
    highway_type: str
    edge_length: float
    edge_heading_rad: float
    is_oneway: bool

@dataclass
class EdgeProjectionResult:
    projected_point: Point
    projected_lat: float
    projected_lon: float
    distance_m: float
    edge_index: int
    highway_type: str
    edge_length: float
    edge_heading_rad: float
    is_oneway: bool

def edge_vector_to_yaw(edge_start_xy: np.ndarray, edge_end_xy: np.ndarray) -> float:
    """
    Compute yaw angle from an edge vector in local cartesian coordinates.
    
    Parameters
    ----------
    edge_start_xy : np.ndarray, shape (2,)
        [x, y] of edge start in local cartesian frame.
    edge_end_xy : np.ndarray, shape (2,)
        [x, y] of edge end in local cartesian frame.
    
    Returns
    -------
    yaw : float
        Heading angle in radians, measured from the x-axis (east),
        counterclockwise positive. Use atan2 for correct quadrant.
    """
    delta = edge_end_xy - edge_start_xy
    yaw = np.arctan2(delta[1], delta[0])
    return yaw


def yaw_to_quaternion_wxyz(yaw: float, current_quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Build a corrected quaternion by replacing yaw from the edge heading
    while preserving roll and pitch from the current state estimate.
    
    Parameters
    ----------
    yaw : float
        Edge-derived yaw angle in radians.
    current_quat_wxyz : np.ndarray, shape (4,)
        Current state quaternion [w, x, y, z].
    
    Returns
    -------
    corrected_quat_wxyz : np.ndarray, shape (4,)
        Quaternion with edge yaw, original roll/pitch. [w, x, y, z].
    """
    # scipy uses [x, y, z, w] internally
    current_quat_xyzw = np.array([
        current_quat_wxyz[1],
        current_quat_wxyz[2],
        current_quat_wxyz[3],
        current_quat_wxyz[0],
    ])

    rot = Rotation.from_quat(current_quat_xyzw)
    # intrinsic ZYX decomposition: [yaw, pitch, roll]
    euler = rot.as_euler('ZYX', degrees=False)

    # Replace only the yaw component with edge heading
    euler[0] = yaw

    corrected_rot = Rotation.from_euler('ZYX', euler)
    corrected_xyzw = corrected_rot.as_quat()  # [x, y, z, w]

    corrected_wxyz = np.array([
        corrected_xyzw[3],
        corrected_xyzw[0],
        corrected_xyzw[1],
        corrected_xyzw[2],
    ])
    return corrected_wxyz


def select_edge_by_heading(
    edges: list,
    vehicle_yaw: float,
    vehicle_pos_xy: np.ndarray,
    oneway_flags: list,
    heading_threshold_deg: float = 60.0,
):
    """
    Select the best edge considering distance AND heading compatibility.
    
    Parameters
    ----------
    edges : list of dict
        Each dict has keys:
          'start_xy': np.ndarray (2,), 'end_xy': np.ndarray (2,),
          'distance': float (distance from vehicle to edge),
          'oneway': bool
    vehicle_yaw : float
        Current vehicle heading in radians.
    vehicle_pos_xy : np.ndarray, shape (2,)
        Vehicle position in cartesian.
    oneway_flags : list of bool
        Whether each edge is one-way.
    heading_threshold_deg : float
        Max angular difference (degrees) to accept an edge.
    
    Returns
    -------
    best_edge : dict or None
    best_yaw : float or None
        The yaw to use for correction (may be flipped for two-way roads).
    """
    threshold_rad = np.deg2rad(heading_threshold_deg)
    candidates = []

    for edge in edges:
        edge_yaw = edge_vector_to_yaw(edge['start_xy'], edge['end_xy'])

        # Angular difference, wrapped to [-pi, pi]
        diff = _wrap_angle(edge_yaw - vehicle_yaw)

        if abs(diff) <= threshold_rad:
            # Heading aligns with edge direction
            candidates.append((edge, edge_yaw, abs(diff), edge['distance']))
        elif not edge['oneway']:
            # Two-way road: check the reverse direction
            reverse_yaw = _wrap_angle(edge_yaw + np.pi)
            diff_rev = _wrap_angle(reverse_yaw - vehicle_yaw)
            if abs(diff_rev) <= threshold_rad:
                candidates.append((edge, reverse_yaw, abs(diff_rev), edge['distance']))

    if not candidates:
        return None, None

    # Sort by distance first, then by heading agreement
    candidates.sort(key=lambda c: (c[3], c[2]))
    best_edge, best_yaw, _, _ = candidates[0]
    return best_edge, best_yaw


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ─── Kalman filter measurement update ───

def build_orientation_measurement(
    edge_yaw: float,
    current_state_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """
    Build the quaternion measurement vector for the KF update step.
    
    Returns the corrected quaternion that should be used as the
    measurement z in your Kalman filter's orientation update.
    """
    corrected_quat = yaw_to_quaternion_wxyz(edge_yaw, current_state_quat_wxyz)
    return corrected_quat

def _edge_heading_at_fraction(edge_geom, frac, eps=1e-4):
    """
    Compute the tangent heading of a LineString at a normalized fraction.
    Uses a small finite-difference step along the line to get the local direction.

    Returns yaw in radians (from east / x-axis, CCW positive) in the
    projected (UTM) coordinate system.
    """
    frac = np.clip(frac, 0.0, 1.0)
    f0 = max(frac - eps, 0.0)
    f1 = min(frac + eps, 1.0)

    p0 = edge_geom.interpolate(f0, normalized=True)
    p1 = edge_geom.interpolate(f1, normalized=True)

    dx = p1.x - p0.x
    dy = p1.y - p0.y

    return np.arctan2(dy, dx)


def _is_oneway(edge_row):
    """
    Determine if an OSM edge is one-way.
    osmnx stores 'oneway' as bool or string depending on version.
    """
    oneway = edge_row.get('oneway', False)
    if isinstance(oneway, bool):
        return oneway
    if isinstance(oneway, str):
        return oneway.lower() in ('yes', 'true', '1')
    return bool(oneway)


def _wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def quat_wxyz_to_yaw(quat_wxyz):
    """Extract yaw from state quaternion [w, x, y, z]."""
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return Rotation.from_quat(quat_xyzw).as_euler('XYZ')[2]

def yaw_to_quat_wxyz(yaw, current_quat_wxyz):
    """Replace yaw in quaternion, preserve roll/pitch."""
    quat_xyzw = np.array([
        current_quat_wxyz[1], current_quat_wxyz[2],
        current_quat_wxyz[3], current_quat_wxyz[0],
    ])
    euler = Rotation.from_quat(quat_xyzw).as_euler('ZYX')  # [yaw, pitch, roll]
    euler[0] = yaw
    corrected_xyzw = Rotation.from_euler('ZYX', euler).as_quat()
    new_quat = np.array([corrected_xyzw[3], corrected_xyzw[0],
                     corrected_xyzw[1], corrected_xyzw[2]])
    new_quat /= np.linalg.norm(new_quat)  # Normalize to unit quaternion
    return new_quat
                     
def project_to_nearest_edge(lat, lon, heading, edges_gdf, max_distance=5,
                            heading_threshold_deg=60.0, top_k=5) -> EdgeProjectionResult | None:
    """
    Project a vehicle position onto the nearest heading-compatible street edge.

    Args:
        lat, lon: Vehicle position
        heading: Vehicle heading in radians (from x-axis / east, CCW positive)
        edges_gdf: GeoDataFrame with street edges (OSM via osmnx)
        max_distance: Maximum search distance in meters
        heading_threshold_deg: Max angular difference to accept an edge
        top_k: Number of nearest candidates to evaluate for heading compatibility

    Returns:
        EdgeProjectionResult with projected point, nearest edge info, distance, and edge heading,
        or None if no compatible edge is found.
    """
    try:
        vehicle_point = Point(lon, lat)

        lon_min, lon_max = lon - 0.001, lon + 0.001
        lat_min, lat_max = lat - 0.0005, lat + 0.0005

        edges_map_local = edges_gdf.cx[lon_min:lon_max, lat_min:lat_max]

        if edges_map_local.empty:
            return None

        utm_crs = edges_map_local.estimate_utm_crs()
        edges_proj = edges_map_local.to_crs(utm_crs)
        vehicle_point_proj = (
            gpd.GeoSeries([vehicle_point], crs=edges_map_local.crs)
            .to_crs(utm_crs)
            .iloc[0]
        )

        # Compute distances and take top_k closest
        distances = edges_proj.geometry.distance(vehicle_point_proj)
        nearest_indices = distances.nsmallest(top_k).index

        threshold_rad = np.deg2rad(heading_threshold_deg)
        candidates = []

        for idx in nearest_indices:
            dist_m = distances[idx]
            if dist_m > max_distance:
                continue

            edge_geom_proj = edges_proj.loc[idx].geometry
            edge_row = edges_map_local.loc[idx]

            # ── Compute edge heading at the projection point ──
            # Project vehicle onto edge to find local tangent direction
            proj_frac = edge_geom_proj.project(vehicle_point_proj, normalized=True)
            edge_yaw = _edge_heading_at_fraction(edge_geom_proj, proj_frac)

            # ── Check heading compatibility ──
            is_oneway = _is_oneway(edge_row)
            diff = _wrap_angle(edge_yaw - heading)

            if abs(diff) <= threshold_rad:
                candidates.append((idx, dist_m, edge_yaw, abs(diff)))
            elif not is_oneway:
                reverse_yaw = _wrap_angle(edge_yaw + np.pi)
                diff_rev = _wrap_angle(reverse_yaw - heading)
                if abs(diff_rev) <= threshold_rad:
                    candidates.append((idx, dist_m, reverse_yaw, abs(diff_rev)))

        if not candidates:
            return None

        # Sort: distance first, heading agreement second
        candidates.sort(key=lambda c: (c[1], c[3]))
        best_idx, best_dist, best_edge_yaw, _ = candidates[0]

        # ── Project point onto the chosen edge ──
        best_edge_proj = edges_proj.loc[best_idx]
        nearest_point_proj = nearest_points(vehicle_point_proj, best_edge_proj.geometry)[1]
        nearest_point_geo = (
            gpd.GeoSeries([nearest_point_proj], crs=utm_crs)
            .to_crs(edges_map_local.crs)
            .iloc[0]
        )

        best_edge_row = edges_map_local.loc[best_idx]

        return EdgeProjectionResult(
            projected_point=nearest_point_geo,
            projected_lat=nearest_point_geo.y,
            projected_lon=nearest_point_geo.x,
            distance_m=best_dist,
            edge_index=best_idx,
            highway_type=best_edge_row.get('highway', None),
            edge_length=best_edge_row.get('length', None),
            edge_heading_rad=best_edge_yaw,
            is_oneway=_is_oneway(best_edge_row),
        )

    except Exception:
        return None
    

@dataclass
class EdgeProjectionResultExt:
    projected_point: Point
    top_n_nearest_edges: gpd.GeoDataFrame
    unique_road_count: int          # number of distinct physical roads in top_n_nearest_edges
    projected_lat: float
    projected_lon: float
    distance_m: float
    edge_index: int
    highway_type: str
    edge_length: float
    edge_heading_rad: float
    is_oneway: bool

def _endpoint_key(geom):
    """
    Return a frozenset of the first and last coordinate of a LineString.
    Two anti-parallel directed edges of the same physical road share this key.
    Coordinates are rounded to ~1 cm to absorb floating-point noise.
    """
    coords = list(geom.coords)
    p0 = tuple(round(v, 5) for v in coords[0])
    p1 = tuple(round(v, 5) for v in coords[-1])
    return frozenset([p0, p1])

def _dedup_edges_by_endpoints(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge anti-parallel directed edge pairs that represent the same physical road.

    For each group of edges sharing the same endpoint-pair key:
      - If any row has oneway=False (two-way), keep that row.
      - Otherwise keep the first row (arbitrary direction of a one-way pair).

    Returns a GeoDataFrame with one representative row per unique physical road,
    preserving the original index of the kept row.
    """
    keys = edges_gdf.geometry.map(_endpoint_key)
    kept_indices = []
    for key, group_idx in edges_gdf.groupby(keys).groups.items():
        group = edges_gdf.loc[group_idx]
        twoway = group[~group.apply(_is_oneway, axis=1)]
        kept_indices.append(twoway.index[0] if not twoway.empty else group.index[0])
    return edges_gdf.loc[kept_indices]

# Approximate meters-per-degree constants (WGS84)
_M_PER_DEG_LAT = 111_320.0          # nearly constant globally
_M_PER_DEG_LON_AT_EQ = 111_320.0    # scales by cos(lat)

def project_to_nearest_edge_width_n_candidates(
        lat,
        lon,
        heading,
        edges_gdf,
        max_candidates: int = 1,
        max_distance: float = 5.0,
        heading_threshold_deg: float = 60.0,
        top_k: int = 5):
    """
    Project a vehicle position onto the nearest heading-compatible street edge.

    After filtering by heading compatibility, candidate edges are deduplicated by
    physical road (anti-parallel directed pairs merged into one). The result field
    ``unique_road_count`` tells the caller how many distinct roads are nearby:
      - 1 → on a single road → position correction is safe to apply.
      - >1 → near an intersection → suppress correction.

    Args:
        lat, lon          : Vehicle position (degrees, WGS-84)
        heading           : Vehicle heading in radians (from east / x-axis, CCW positive)
        edges_gdf         : GeoDataFrame with street edges (OSM via osmnx, geographic CRS)
        max_candidates    : Number of top edges to return in
                            ``EdgeProjectionResult.top_n_nearest_edges``
        max_distance      : Maximum snap distance in metres
        heading_threshold_deg : Max angular difference (degrees) to accept an edge
        top_k             : Number of nearest edges to evaluate for heading compatibility

    Returns:
        EdgeProjectionResult for the best heading-compatible edge, or None.
    """
    try:
        vehicle_point = Point(lon, lat)
        geo_crs = edges_gdf.crs

        # ── Adaptive bounding-box pre-filter ──────────────────────────────────
        margin = max_distance * 1.2
        dlat = margin / _M_PER_DEG_LAT
        dlon = margin / (_M_PER_DEG_LON_AT_EQ * np.cos(np.deg2rad(lat)))
        edges_map_local = edges_gdf.cx[lon - dlon:lon + dlon, lat - dlat:lat + dlat]

        if edges_map_local.empty:
            return None

        # ── Project to UTM for metric distance / heading computation ──────────
        utm_crs = edges_map_local.estimate_utm_crs()
        edges_proj = edges_map_local.to_crs(utm_crs)
        vehicle_point_proj = (
            gpd.GeoSeries([vehicle_point], crs=geo_crs)
            .to_crs(utm_crs)
            .iloc[0]
        )

        # ── Evaluate the top_k closest edges for heading compatibility ────────
        distances = edges_proj.geometry.distance(vehicle_point_proj)
        nearest_indices = distances.nsmallest(top_k).index

        threshold_rad = np.deg2rad(heading_threshold_deg)
        # Each entry: (edge_idx, snap_dist_m, accepted_yaw_rad, heading_diff_rad)
        candidates = []

        for idx in nearest_indices:
            dist_m = distances[idx]
            if dist_m > max_distance:
                continue

            edge_geom_proj = edges_proj.loc[idx].geometry
            edge_row = edges_map_local.loc[idx]

            proj_frac = edge_geom_proj.project(vehicle_point_proj, normalized=True)
            edge_yaw = _edge_heading_at_fraction(edge_geom_proj, proj_frac)

            is_oneway = _is_oneway(edge_row)
            diff = _wrap_angle(edge_yaw - heading)

            if abs(diff) <= threshold_rad:
                candidates.append((idx, dist_m, edge_yaw, abs(diff)))
            elif not is_oneway:
                reverse_yaw = _wrap_angle(edge_yaw + np.pi)
                diff_rev = _wrap_angle(reverse_yaw - heading)
                if abs(diff_rev) <= threshold_rad:
                    candidates.append((idx, dist_m, reverse_yaw, abs(diff_rev)))

        if not candidates:
            return None

        # Sort: snap distance first, heading agreement second
        candidates.sort(key=lambda c: (c[1], c[3]))
        best_idx, best_dist, best_edge_yaw, _ = candidates[0]

        # ── Collect top-N candidate edges and deduplicate by physical road ────
        top_n_edge_indices = [idx for idx, *_ in candidates[:max_candidates]]
        top_n_nearest_edges = edges_map_local.loc[top_n_edge_indices]
        top_n_nearest_edges = _dedup_edges_by_endpoints(top_n_nearest_edges)
        unique_road_count = len(top_n_nearest_edges)

        # ── Project vehicle onto the best edge ────────────────────────────────
        nearest_point_proj = nearest_points(vehicle_point_proj, edges_proj.loc[best_idx].geometry)[1]
        nearest_point_geo = (
            gpd.GeoSeries([nearest_point_proj], crs=utm_crs)
            .to_crs(geo_crs)
            .iloc[0]
        )

        best_edge_row = edges_map_local.loc[best_idx]

        return EdgeProjectionResultExt(
            projected_point=nearest_point_geo,
            top_n_nearest_edges=top_n_nearest_edges,
            unique_road_count=unique_road_count,
            projected_lat=nearest_point_geo.y,
            projected_lon=nearest_point_geo.x,
            distance_m=best_dist,
            edge_index=best_idx,
            highway_type=best_edge_row.get('highway', None),
            edge_length=best_edge_row.get('length', None),
            edge_heading_rad=best_edge_yaw,
            is_oneway=_is_oneway(best_edge_row),
        )

    except Exception:
        return None
