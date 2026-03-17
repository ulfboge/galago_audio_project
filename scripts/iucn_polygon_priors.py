"""
Polygon-based location priors using per-species IUCN GeoJSON ranges.

Design goals:
- No heavyweight dependencies (no shapely/geopandas)
- Good enough for a small number of species (19) and moderate polygon complexity
- Cache loaded geometries for repeated calls

GeoJSON notes:
- Coordinates are expected in RFC7946: [lon, lat] in EPSG:4326.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Polygon:
    # Outer ring and optional holes are lists of (lon, lat) tuples.
    outer: List[Tuple[float, float]]
    holes: List[List[Tuple[float, float]]]
    bbox: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


def _ring_bbox(ring: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    min_lon = min(p[0] for p in ring)
    max_lon = max(p[0] for p in ring)
    min_lat = min(p[1] for p in ring)
    max_lat = max(p[1] for p in ring)
    return (min_lon, min_lat, max_lon, max_lat)


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> bool:
    # Collinearity + within bounding box; small epsilon for float noise.
    eps = 1e-12
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= eps


def _point_in_ring(lon: float, lat: float, ring: List[Tuple[float, float]]) -> bool:
    """
    Ray-casting point-in-polygon for a single ring.
    Treats boundary as inside.
    """
    if len(ring) < 3:
        return False

    inside = False
    x = lon
    y = lat

    # Ensure we iterate over edges (vi -> vj)
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]

        # Boundary check
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True

        # Ray casting: count crossings of horizontal ray to +inf in x direction.
        # Standard robust form: check edge straddles y and intersection x is to the right.
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1
        )
        if intersects:
            inside = not inside
    return inside


def _parse_geom_to_polygons(geom: Dict[str, Any]) -> List[_Polygon]:
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not gtype or coords is None:
        return []

    polys: List[_Polygon] = []

    def parse_polygon(poly_coords: Any) -> None:
        # poly_coords: [outer_ring, hole1, hole2, ...]
        if not isinstance(poly_coords, list) or not poly_coords:
            return
        rings: List[List[Tuple[float, float]]] = []
        for ring in poly_coords:
            if not isinstance(ring, list) or len(ring) < 3:
                continue
            pts: List[Tuple[float, float]] = []
            for p in ring:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                pts.append((float(p[0]), float(p[1])))
            if len(pts) >= 3:
                rings.append(pts)
        if not rings:
            return
        outer = rings[0]
        holes = rings[1:]
        bbox = _ring_bbox(outer)
        polys.append(_Polygon(outer=outer, holes=holes, bbox=bbox))

    if gtype == "Polygon":
        parse_polygon(coords)
    elif gtype == "MultiPolygon":
        if isinstance(coords, list):
            for poly in coords:
                parse_polygon(poly)
    elif gtype == "GeometryCollection":
        geoms = geom.get("geometries") or []
        for g in geoms:
            if isinstance(g, dict):
                polys.extend(_parse_geom_to_polygons(g))
    else:
        # Ignore Point/LineString etc.
        return []

    return polys


def _load_geojson_polygons(path: Path) -> List[_Polygon]:
    data = json.loads(path.read_text(encoding="utf-8"))

    features = []
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        features = data.get("features") or []
    elif isinstance(data, dict) and data.get("type") == "Feature":
        features = [data]
    else:
        return []

    out: List[_Polygon] = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry")
        if isinstance(geom, dict):
            out.extend(_parse_geom_to_polygons(geom))
    return out


class IucnPolygonPrior:
    """
    Loads the model-label -> GeoJSON mapping and performs point-in-polygon tests.

    Expected inputs:
      - data/iucn_geojson_index_by_label.json (tracked)
      - data/iucn/ranges_geojson/by_species/*.geojson (local, ignored by git)
    """

    def __init__(
        self,
        mapping_json: Optional[Path] = None,
    ) -> None:
        self.mapping_json = mapping_json or (PROJECT_ROOT / "data" / "iucn_geojson_index_by_label.json")
        self._label_to_geojson_relpath: Dict[str, str] = {}
        self._polys_by_label: Dict[str, List[_Polygon]] = {}
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.mapping_json.exists():
            # No mapping available; leave empty.
            self._loaded = True
            return
        data = json.loads(self.mapping_json.read_text(encoding="utf-8"))
        label_to = (data or {}).get("label_to_geojson", {})
        if isinstance(label_to, dict):
            for label, v in label_to.items():
                if not isinstance(label, str) or not isinstance(v, dict):
                    continue
                rel = v.get("geojson_relpath")
                if isinstance(rel, str) and rel.strip():
                    self._label_to_geojson_relpath[label] = rel.strip()
        self._loaded = True

    def is_available(self) -> bool:
        self._ensure_loaded()
        return bool(self._label_to_geojson_relpath)

    def _polys_for_label(self, label: str) -> List[_Polygon]:
        self._ensure_loaded()
        if label in self._polys_by_label:
            return self._polys_by_label[label]
        rel = self._label_to_geojson_relpath.get(label)
        if not rel:
            self._polys_by_label[label] = []
            return []
        path = (PROJECT_ROOT / rel).resolve()
        if not path.exists():
            self._polys_by_label[label] = []
            return []
        polys = _load_geojson_polygons(path)
        self._polys_by_label[label] = polys
        return polys

    def contains(self, label: str, lat: float, lon: float) -> bool:
        # GeoJSON is lon/lat order.
        x = float(lon)
        y = float(lat)
        polys = self._polys_for_label(label)
        for poly in polys:
            min_lon, min_lat, max_lon, max_lat = poly.bbox
            if x < min_lon or x > max_lon or y < min_lat or y > max_lat:
                continue
            if _point_in_ring(x, y, poly.outer):
                # Exclude holes
                in_hole = any(_point_in_ring(x, y, hole) for hole in poly.holes)
                if not in_hole:
                    return True
        return False

    def prior(self, label: str, lat: Optional[float], lon: Optional[float]) -> float:
        """
        Returns a location prior:
          - 1.0 if point is inside the species range polygon
          - 0.1 if outside
          - 0.5 if no point provided or polygon data unavailable
        """
        if lat is None or lon is None:
            return 0.5
        if not self.is_available():
            return 0.5
        try:
            inside = self.contains(label, lat=lat, lon=lon)
        except Exception:
            # Be robust: if geometry parsing fails, fall back to neutral.
            return 0.5
        return 1.0 if inside else 0.1

    def status(self, label: str, lat: Optional[float], lon: Optional[float]) -> str:
        if lat is None or lon is None or not self.is_available():
            return "Unknown"
        p = self.prior(label, lat=lat, lon=lon)
        if p >= 0.9:
            return "Most likely here"
        if p >= 0.5:
            return "Possible but uncommon"
        return "Unlikely here"

