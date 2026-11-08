"""Observed SDSS galaxies and their cataloged nearest-filament associations.

Tempel et al. 2014, VizieR J/MNRAS/438/3465/table3.
No stellar positions, galaxies, connections, or CMB values are synthesized.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_GALAXIES = 576_493
GALAXY_COLUMNS = [
    "IDgal", "Ngal", "z", "RAJ2000", "DEJ2000", "Dcor",
    "umag", "gmag", "rmag", "imag", "zmag", "Lr", "w",
    "Dedg", "Dfil", "ID", "IDpt",
]
REQUIRED_GALAXY_COLUMNS = [
    "IDgal", "RAJ2000", "DEJ2000", "Dcor", "gmag", "rmag",
    "Lr", "Dfil", "ID", "IDpt",
]


def load_galaxies(path: str | Path, *, require_full: bool = True) -> pd.DataFrame:
    """Load the published 576,493-galaxy table from LOCAL CSV only."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing galaxy catalog: {path.resolve()}\n"
            "Place your existing VizieR table3 CSV in data/. No automatic download is performed."
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    missing = sorted(set(REQUIRED_GALAXY_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Galaxy CSV is missing required columns {missing}; found {list(df.columns)}")
    for col in GALAXY_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.IDgal.isna().any() or df.IDgal.duplicated().any():
        raise ValueError("Galaxy IDs are missing or duplicated; refusing unverified data")
    if require_full and len(df) != EXPECTED_GALAXIES:
        raise ValueError(
            f"Expected {EXPECTED_GALAXIES:,} real galaxies, found {len(df):,}. "
            "Set strict_full_catalog=False only for a deliberately downloaded real subset."
        )
    # Some measurements may be missing; keep objects with measured coordinates.
    usable = df[["RAJ2000", "DEJ2000", "Dcor"]].notna().all(axis=1)
    usable &= df.Dcor.gt(0) & df.RAJ2000.between(0, 360) & df.DEJ2000.between(-90, 90)
    if usable.sum() == 0:
        raise ValueError("No galaxies have valid observed positions and corrected distances")
    if (~usable).any():
        print(f"Excluding {(~usable).sum():,} catalog rows without usable observed spatial coordinates")
    return df.loc[usable].copy().reset_index(drop=True)


def add_sdss_xyz(galaxies: pd.DataFrame) -> pd.DataFrame:
    """Transform RA/Dec and published corrected comoving distance to Tempel X/Y/Z.

    Tempel et al. (2014) use SDSS survey longitude/latitude (lambda, eta),
    not the conventional equatorial Cartesian axes. This exactly equivalent
    rotated orthonormal basis preserves the true filament-table alignment.
    Coordinates remain in comoving h^-1 Mpc.
    """
    df = galaxies.copy()
    ra = np.deg2rad(df.RAJ2000.to_numpy(float))
    dec = np.deg2rad(df.DEJ2000.to_numpy(float))
    direction = np.column_stack([
        np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)
    ])
    # SDSS coordinate anchors: (lambda=0, eta=0) = (RA=185°, Dec=32.5°)
    # and (lambda=90°, eta=0) = (RA=275°, Dec=0°).
    ax = np.deg2rad(275.0)
    ay = np.deg2rad(185.0)
    dy = np.deg2rad(32.5)
    e_x = np.array([-np.cos(ax), -np.sin(ax), 0.0])
    e_y = np.array([np.cos(dy)*np.cos(ay), np.cos(dy)*np.sin(ay), np.sin(dy)])
    e_z = np.cross(e_x, e_y)
    basis = np.stack([e_x, e_y, e_z], axis=1)
    xyz = (direction @ basis) * df.Dcor.to_numpy(float)[:, None]
    df[["x_gal", "y_gal", "z_gal"]] = xyz
    return df


def attach_filament_associations(galaxies: pd.DataFrame,
                                  filament_points: pd.DataFrame,
                                  *, max_dfil: float = 1.0) -> pd.DataFrame:
    """Join galaxy -> nearest cataloged filament POINT by the published IDpt.

    An association means *nearest filament within max_dfil h^-1 Mpc*, not
    a gravitationally bound relation or a confirmed galaxy-group membership.
    All galaxies remain in the returned table, even without an association.
    """
    if not {"x_gal", "y_gal", "z_gal"}.issubset(galaxies.columns):
        raise ValueError("Call add_sdss_xyz(galaxies) before attaching associations")
    pts = filament_points[["ID", "IDpt", "x", "y", "z"]].copy()
    pts = pts.rename(columns={"ID": "filament_ID", "IDpt": "filament_point_ID",
                              "x": "filament_x", "y": "filament_y", "z": "filament_z"})
    # The paper specifies that IDpt is unique across the entire point catalog.
    if pts.filament_point_ID.duplicated().any():
        raise ValueError("Filament IDpt must be globally unique")
    df = galaxies.merge(pts, left_on="IDpt", right_on="filament_point_ID",
                        how="left", validate="many_to_one", sort=False)
    xyz1 = df[["x_gal", "y_gal", "z_gal"]].to_numpy(float)
    xyz2 = df[["filament_x", "filament_y", "filament_z"]].to_numpy(float)
    dist = np.linalg.norm(xyz1 - xyz2, axis=1)
    df["point_distance"] = dist
    id_match = df.ID.notna() & df.filament_ID.notna() & df.ID.eq(df.filament_ID)
    df["associated"] = (
        id_match & df.Dfil.notna() & df.Dfil.ge(0) & df.Dfil.le(max_dfil)
        & np.isfinite(dist)
    )
    # Guard against a wrong rotation / inconsistent source files: nearest
    # sampled spine point should be close to the published distance to axis.
    linked = df.loc[df.associated, "point_distance"]
    if len(linked) >= 50 and float(linked.median()) > max_dfil + 3.0:
        raise ValueError(
            f"Galaxy/filament coordinates do not align (median point offset {linked.median():.2f} "
            "h^-1 Mpc). Confirm table2 and table3 come from the same catalog release."
        )
    return df


def choose_display_links(associations: pd.DataFrame, *, limit: int = 15000,
                         seed: int = 42) -> pd.DataFrame:
    """Deterministically subsample REAL associations for legible rendering."""
    rows = associations.loc[associations.associated]
    if limit and len(rows) > limit:
        return rows.sample(n=limit, random_state=seed).sort_index()
    return rows


def show_counts(galaxies: pd.DataFrame, linked: pd.DataFrame) -> dict:
    return {
        "cataloged_galaxies_with_positions": len(galaxies),
        "linked_nearest_filament_within_cutoff": int(linked.associated.sum()),
        "different_filaments_linked": int(linked.loc[linked.associated, "ID"].nunique()),
    }


def export_galaxy_filament_network(galaxies: pd.DataFrame,
                                   filament_points: pd.DataFrame,
                                   output_dir: str | Path, *, hubble_h: float = 0.674) -> tuple[Path, Path]:
    """Optional machine-readable graph with real galaxy IDs and spine-point IDs.

    Node types: 'galaxy', 'filament_point'.
    Edges: published nearest-point association (only qualified rows) and
    geometric spine segments (consecutive cataloged points on a filament).
    The second type describes a reconstructed ridge, not a physical cable.
    """
    if hubble_h <= 0:
        raise ValueError("hubble_h must be positive")
    factor = 3.26156 / hubble_h
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    gal = galaxies[["IDgal", "x_gal", "y_gal", "z_gal", "z", "Lr"]].copy()
    gal = gal.rename(columns={"IDgal":"source_id", "x_gal":"x", "y_gal":"y", "z_gal":"zpos", "z":"redshift", "Lr":"luminosity"})
    gal.insert(0, "node_id", "galaxy:" + gal.source_id.astype("Int64").astype(str))
    gal["type"] = "galaxy"
    pts = filament_points[["IDpt","x","y","z"]].copy()
    pts = pts.rename(columns={"IDpt":"source_id", "z":"zpos"})
    pts.insert(0, "node_id", "filament_point:" + pts.source_id.astype("Int64").astype(str))
    pts["type"] = "filament_point"
    pts["redshift"] = np.nan
    pts["luminosity"] = np.nan
    nodes = pd.concat([gal,pts], ignore_index=True)
    nodes[['x','y','zpos']] *= factor
    nodes = nodes.rename(columns={'x':'x_million_ly', 'y':'y_million_ly', 'zpos':'z_million_ly'})

    assoc = galaxies.loc[galaxies.associated]
    a = pd.DataFrame({
        "source": "galaxy:" + assoc.IDgal.astype("Int64").astype(str),
        "target": "filament_point:" + assoc.IDpt.astype("Int64").astype(str),
        "relation": "nearest_reconstructed_filament_point",
        "distance_million_ly": assoc.Dfil.to_numpy(float)*factor,
    })
    # FILAMENT POINTS MUST BE ORDERED BY IDpt WITHIN A FILAMENT.
    spine = filament_points.sort_values(["ID","IDpt"],kind="stable")
    prev = spine.groupby("ID", sort=False).shift(1)
    valid = prev.IDpt.notna()
    p = spine.loc[valid, ["IDpt","x","y","z"]].to_numpy(float)
    q = prev.loc[valid, ["IDpt","x","y","z"]].to_numpy(float)
    s = pd.DataFrame({
        "source": [f"filament_point:{int(v)}" for v in q[:,0]],
        "target": [f"filament_point:{int(v)}" for v in p[:,0]],
        "relation": "consecutive_reconstructed_spine_points",
        "distance_million_ly": np.linalg.norm(p[:,1:] - q[:,1:],axis=1)*factor,
    })
    edges = pd.concat([a,s], ignore_index=True)
    path_nodes = outdir/"galaxy_filament_nodes.csv"
    path_edges = outdir/"galaxy_filament_edges.csv"
    nodes.to_csv(path_nodes,index=False)
    edges.to_csv(path_edges,index=False)
    return path_nodes,path_edges


def add_verified_galaxy_names(galaxies: pd.DataFrame, names_csv: str | Path) -> pd.DataFrame:
    """Attach VERIFIED common/catalog names from a LOCAL manual crossmatch.

    Format: IDgal,name. Missing entries stay explicitly unnamed; no catalogue
    designations or familiar galaxy names are invented from coordinates.
    """
    names_csv = Path(names_csv)
    result = galaxies.copy()
    result['display_name'] = 'Unnamed SDSS galaxy'
    result['has_verified_name'] = False
    if not names_csv.is_file():
        return result
    names = pd.read_csv(names_csv, dtype={'name':'string'})
    names.columns = names.columns.str.strip()
    if not {'IDgal','name'}.issubset(names.columns):
        raise ValueError('galaxy_names.csv must have IDgal,name columns')
    names = names[['IDgal','name']].copy()
    names['IDgal'] = pd.to_numeric(names.IDgal, errors='coerce')
    names['name'] = names['name'].astype('string').str.strip()
    names = names.dropna(subset=['IDgal','name'])
    names = names[names.name.str.len() > 0]
    if names.IDgal.duplicated().any():
        raise ValueError('galaxy_names.csv contains duplicate IDgal keys')
    # Map avoids duplicating rows and does not overwrite the survey ID.
    mapped = result.IDgal.map(names.set_index('IDgal').name)
    has_name = mapped.notna()
    result.loc[has_name, 'display_name'] = mapped[has_name].astype(str)
    result['has_verified_name'] = has_name
    return result
