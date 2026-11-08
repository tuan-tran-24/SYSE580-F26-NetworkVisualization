"""Load and prepare genuine Tempel et al. SDSS filament-spine coordinates."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ["ID", "IDpt", "x", "y", "z", "fden"]
EXPECTED_ROWS = 275_599
EXPECTED_FILAMENTS = 15_421


def load_filaments(path: str | Path, *, require_full: bool = True) -> pd.DataFrame:
    """Read the locally saved VizieR table2 CSV. Never downloads or simulates data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing real SDSS filament catalog: {path.resolve()}\n"
            "Place your existing VizieR table2 CSV in data/. The project is local-only."
        )

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(x).strip() for x in df.columns]
    missing = sorted(set(COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Filament CSV missing columns: {missing}. Found: {list(df.columns)}")
    df = df[COLUMNS].copy()
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["ID", "IDpt", "x", "y", "z"]].isna().any().any():
        raise ValueError("Filament CSV contains missing/non-numeric identifiers or coordinates")
    if df.duplicated(["ID", "IDpt"]).any():
        raise ValueError("Filament CSV contains duplicate (ID, IDpt) pairs")
    if require_full and (len(df) != EXPECTED_ROWS or df.ID.nunique() != EXPECTED_FILAMENTS):
        raise ValueError(
            f"Incomplete catalog: {len(df):,}/{EXPECTED_ROWS:,} points; "
            f"{df.ID.nunique():,}/{EXPECTED_FILAMENTS:,} filaments. "
            "Set strict_full_catalog=False only if you intentionally use a real partial extract."
        )
    df = df.sort_values(["ID", "IDpt"], kind="stable").reset_index(drop=True)
    return df


def make_traces(df: pd.DataFrame) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Each filament is an ordered chain of *cataloged* 3D ridge points."""
    traces = []
    for fid, group in df.groupby("ID", sort=False):
        if len(group) < 2:
            continue
        xyz = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
        fden = group["fden"].to_numpy(dtype=np.float64)
        traces.append((int(fid), xyz, fden))
    return traces


def density_normalizer(df: pd.DataFrame):
    values = df.fden.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No finite fden values exist in the catalog")
    low, high = np.percentile(values, [5, 95])
    if high <= low:
        high = low + 1.0

    def strength(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return np.clip((np.where(np.isfinite(arr), arr, low) - low) / (high - low), 0, 1)

    return strength


def footprint_limits(df: pd.DataFrame, width: int, height: int, *, padding: float = 1.05):
    """True physical XY aspect ratio, including the observer at the origin."""
    x0, x1 = min(float(df.x.min()), 0.0), max(float(df.x.max()), 0.0)
    y0, y1 = min(float(df.y.min()), 0.0), max(float(df.y.max()), 0.0)
    xc, yc = (x0+x1)/2, (y0+y1)/2
    sx = max(x1-x0, (y1-y0)*width/height, 1.0) * padding
    sy = sx*height/width
    return (xc-sx/2, xc+sx/2, yc-sy/2, yc+sy/2)


def community_colors(traces, *, sample_step=12, join_radius=2.0):
    """Optional graph-derived groups, NOT confirmed galaxy clusters.

    Connect FILAMENT IDs if observed ridge points approach within join_radius.
    """
    from scipy.spatial import cKDTree
    import networkx as nx

    pts = []
    ids = []
    for fid, coords, _ in traces:
        sampled = coords[::max(1, sample_step)]
        pts.append(sampled)
        ids.extend([fid] * len(sampled))
    if not pts:
        return {}
    pts = np.vstack(pts)
    ids = np.asarray(ids)
    graph = nx.Graph()
    graph.add_nodes_from(fid for fid, _, _ in traces)
    for a, b in cKDTree(pts).query_pairs(join_radius):
        if ids[a] != ids[b]:
            u, v = int(ids[a]), int(ids[b])
            if graph.has_edge(u, v):
                graph[u][v]["weight"] += 1
            else:
                graph.add_edge(u, v, weight=1)

    groups = sorted(nx.community.louvain_communities(graph, weight="weight", seed=42),
                    key=len, reverse=True)
    palette = ["#d7e4ee", "#c9dedf", "#f5d8b3", "#e3d4ed", "#cdd7ef"]
    return {fid: palette[k % len(palette)] for k, group in enumerate(groups)
            for fid in group}
