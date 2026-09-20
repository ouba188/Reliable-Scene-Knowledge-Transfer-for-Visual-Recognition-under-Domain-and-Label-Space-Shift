"""Module-1 k_i features for the object table: local port relations per candidate ship.

Given a port's facility/coast/navigation geometry and a scene's object centroids (in the raster
CRS), produce per-object:
  d_coast_m, d_quay_m, d_anchorage_m, d_fairway_m   nearest distances (capped at RADIUS)
  on_fairway                                        within ON_FAIRWAY_M of a navigation line
  channel_angle_deg                                 acute angle between the OBB long axis and the
                                                    nearest navigation line's local direction (0-90)
  local_ships_500m / local_ships_1km                ship-group density around the object
  nn_distance_m                                     nearest-neighbour distance in the same scene
  heading_consistency_deg                           circular dispersion of neighbour orientations
  worldcover_class                                  land-cover value at the object (80 = water)
  semantic_area                                     water / built / land / unknown

Distances use shapely 2 vectorized STRtree queries (dwithin), so cost is ~O(n log m), not O(n·m).
"""
import math

import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

RADIUS = 2000.0          # distances beyond this are reported as the cap
ON_FAIRWAY_M = 100.0
WORLDCOVER_WATER = 80


def nearest_within(tree, geoms, points, radius=RADIUS):
    """Return (distance array, index array) for the nearest geometry to each point (cap if none)."""
    n = len(points)
    dist = np.full(n, radius, dtype=float)
    idx = np.full(n, -1, dtype=int)
    if tree is None or not len(geoms):
        return dist, idx
    hits = tree.query(points, predicate='dwithin', distance=radius)
    obj_idx, geom_idx = hits[0], hits[1]
    if len(obj_idx):
        uniq = {}
        for o, g in zip(obj_idx, geom_idx):
            d = points[o].distance(geoms[g])
            if d < dist[o]:
                dist[o] = d
                uniq[o] = g
        for o, g in uniq.items():
            idx[o] = g
    return dist, idx


def channel_angle(points, obb_long_deg, tree, geoms, seg_hint=0.0):
    """Acute angle (0-90) between each object's long axis and the nearest navigation line direction."""
    out = np.full(len(points), np.nan)
    if tree is None or not len(geoms):
        return out
    hits = tree.query(points, predicate='dwithin', distance=ON_FAIRWAY_M)
    for o, g in zip(hits[0], hits[1]):
        line = geoms[g]
        try:
            proj = line.project(points[o])
            delta = max(1.0, min(50.0, line.length / 10))
            a = np.array(line.interpolate(max(0.0, proj - delta)).coords[0])
            b = np.array(line.interpolate(min(line.length, proj + delta)).coords[0])
        except Exception:
            continue
        if np.allclose(a, b):
            continue
        line_deg = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        diff = abs((obb_long_deg[o] - line_deg + 90) % 180 - 90)
        out[o] = min(diff, 180 - diff) if diff > 90 else diff
    return out


def local_density(coords, k_neighbours=8):
    """Per-object neighbour counts (500 m / 1 km), nearest-neighbour distance, orientation spread.

    coords: (n, 2) world metres; orientations handled separately so this stays reusable.
    """
    from scipy.spatial import cKDTree
    n = len(coords)
    out = dict(local_ships_500m=np.zeros(n, int), local_ships_1km=np.zeros(n, int),
               nn_distance_m=np.full(n, np.nan))
    if n < 2:
        return out
    tree = cKDTree(coords)
    for radius, key in ((500.0, 'local_ships_500m'), (1000.0, 'local_ships_1km')):
        counts = tree.query_ball_point(coords, radius, workers=-1)
        out[key] = np.array([len(c) - 1 for c in counts])
    dd, _ = tree.query(coords, k=2)
    out['nn_distance_m'] = dd[:, 1]
    return out


def heading_consistency(coords, long_deg, radius=1000.0):
    """Circular dispersion of neighbour long-axis orientations (degrees, 0 = aligned)."""
    from scipy.spatial import cKDTree
    n = len(coords)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    tree = cKDTree(coords)
    doubled = np.radians(2.0 * np.nan_to_num(long_deg))
    neigh = tree.query_ball_point(coords, radius, workers=-1)
    for i, js in enumerate(neigh):
        if len(js) < 3:
            continue
        s = np.sin(doubled[js]).mean()
        c = np.cos(doubled[js]).mean()
        r = math.hypot(s, c)
        if r > 0:
            out[i] = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(min(1.0, r))))) / 2.0
    return out


def worldcover_at(points_world, raster_path):
    """Sample land-cover values at world coordinates (point sampling; never reads the whole tile)."""
    import rasterio
    vals = np.full(len(points_world), -1, dtype=int)
    try:
        with rasterio.open(raster_path) as ds:
            for i, v in enumerate(ds.sample(points_world)):
                vals[i] = int(v[0])
    except Exception:
        pass
    return vals


def semantic_area(values):
    """Collapse land-cover codes into the semantic areas the k_i vector uses."""
    out = []
    for v in values:
        if v < 0:
            out.append('unknown')
        elif v == WORLDCOVER_WATER:
            out.append('water')
        elif v in (50,):
            out.append('built')
        elif v == 0:
            out.append('nodata')
        else:
            out.append('land')
    return out
