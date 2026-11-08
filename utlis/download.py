"""Optional one-time download of TWO real public astronomy datasets.

Nothing in this module runs unless download_data(...) is called explicitly.
Every visualization afterwards reads only local CSV/FITS files.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .catalog import COLUMNS, EXPECTED_ROWS, EXPECTED_FILAMENTS

CATALOG_ID = "J/MNRAS/438/3465/table2"
VIZIER_HOSTS = ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu")
WMAP_URLS = (
    "https://lambda.gsfc.nasa.gov/data/map/dr5/dfp/ilc/wmap_ilc_9yr_v5.fits",
    "https://nasa-lambda.s3.amazonaws.com/wmap/dr5/dfp/ilc/wmap_ilc_9yr_v5.fits",
)


def _session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "CosmicWebViewer/2.0 (academic astronomy visualization)"})
    return session


def _clean_vizier_table(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if not set(COLUMNS).issubset(df.columns):
        raise ValueError(f"VizieR response is missing required columns: {set(COLUMNS)-set(df.columns)}")
    df = df[COLUMNS]
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # VizieR ASCII responses may contain a units and separator row.
    df = df.dropna(subset=["ID", "IDpt", "x", "y", "z"])
    if df.duplicated(["ID", "IDpt"]).any():
        raise ValueError("Duplicate catalog (ID, IDpt) identifiers; rejecting response")
    return df.sort_values(["ID", "IDpt"]).reset_index(drop=True)


def _check_complete(df):
    df = _clean_vizier_table(df)
    if len(df) != EXPECTED_ROWS or df.ID.nunique() != EXPECTED_FILAMENTS:
        raise ValueError(f"Incomplete result: {len(df):,}/{EXPECTED_ROWS:,} points, "
                         f"{df.ID.nunique():,}/{EXPECTED_FILAMENTS:,} filaments")
    return df


def _request_table(session, host, low=None, high=None):
    params = {"-source": CATALOG_ID, "-out": ",".join(COLUMNS), "-out.max": "unlimited"}
    if low is not None:
        params["ID"] = f"{low}..{high}"
    response = session.get(f"https://{host}/viz-bin/asu-tsv", params=params, timeout=(25, 180))
    response.raise_for_status()
    if "<html" in response.text[:1000].lower():
        raise ValueError("Catalog service returned an HTML page, not a TSV file")
    return _clean_vizier_table(pd.read_csv(StringIO(response.text), sep="\t", comment="#", dtype=str))


def download_filaments(session: requests.Session, csv_path: Path) -> Path:
    """Fetch complete VizieR table2 using mirrors and resumable ID range requests."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.is_file():
        try:
            df = _check_complete(pd.read_csv(csv_path, low_memory=False))
        except Exception as error:
            raise RuntimeError(f"Existing local catalog is invalid: {csv_path}: {error}") from error
        print(f"Existing complete SDSS catalog: {csv_path} ({len(df):,} points)")
        return csv_path

    errors = []
    for host in VIZIER_HOSTS:
        try:
            print(f"Downloading SDSS filament table from {host} ...")
            df = _check_complete(_request_table(session, host))
            temporary = csv_path.with_suffix(".csv.part")
            df.to_csv(temporary, index=False)
            temporary.replace(csv_path)
            print(f"Saved {csv_path} ({len(df):,} real filament points)")
            return csv_path
        except Exception as error:
            errors.append(f"{host}: {type(error).__name__}: {error}")
            print(f"  Complete-table request failed: {error}")

    print("Attempting smaller resumable catalog ranges...")
    chunks_dir = csv_path.parent / "_vizier_parts"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    chunk_size = 200
    # Include ID zero if the catalog uses zero-based identifiers.
    for low in range(0, EXPECTED_FILAMENTS + 1, chunk_size):
        high = min(EXPECTED_FILAMENTS, low + chunk_size - 1)
        chunk_path = chunks_dir / f"ID_{low:05d}_{high:05d}.csv"
        if chunk_path.exists():
            part = _clean_vizier_table(pd.read_csv(chunk_path))
        else:
            part = None
            for host in VIZIER_HOSTS:
                try:
                    candidate = _request_table(session, host, low, high)
                    if not candidate.empty and not candidate["ID"].between(low, high).all():
                        raise ValueError("Catalog server ignored ID range filter")
                    part = candidate
                    part.to_csv(chunk_path, index=False)
                    break
                except Exception as error:
                    errors.append(f"ID {low}-{high} / {host}: {error}")
            if part is None:
                raise RuntimeError("Could not download the SDSS catalog; cached chunks are kept. "
                                   "Rerun later or download table2 from VizieR manually.\n" + "\n".join(errors[-4:]))
            time.sleep(.15)
        chunks.append(part)
    full = _check_complete(pd.concat(chunks, ignore_index=True))
    temporary = csv_path.with_suffix(".csv.part")
    full.to_csv(temporary, index=False)
    temporary.replace(csv_path)
    print(f"Saved complete SDSS catalog: {csv_path} ({len(full):,} points)")
    return csv_path


def download_wmap(session: requests.Session, fits_path: Path) -> Path:
    """Download original NASA WMAP nine-year CMB FITS, never a generated texture."""
    fits_path.parent.mkdir(parents=True, exist_ok=True)
    if fits_path.exists():
        with fits_path.open("rb") as stream:
            signature = stream.read(80)
        if fits_path.stat().st_size < 10_000_000 or not signature.startswith(b"SIMPLE"):
            raise RuntimeError(f"Existing FITS file appears invalid: {fits_path}")
        print(f"Existing WMAP FITS: {fits_path}")
        return fits_path

    errors = []
    for url in WMAP_URLS:
        temporary = fits_path.with_suffix(".fits.part")
        try:
            print(f"Downloading real NASA WMAP nine-year CMB: {url}")
            with session.get(url, timeout=(25, 180), stream=True) as resp:
                resp.raise_for_status()
                with temporary.open("wb") as file:
                    for block in resp.iter_content(chunk_size=1024*1024):
                        if block:
                            file.write(block)
            with temporary.open("rb") as stream:
                signature = stream.read(80)
            if temporary.stat().st_size < 10_000_000 or not signature.startswith(b"SIMPLE"):
                raise ValueError("Downloaded content is not the expected WMAP FITS")
            temporary.replace(fits_path)
            print(f"Saved WMAP map: {fits_path}")
            return fits_path
        except Exception as error:
            errors.append(f"{url}: {type(error).__name__}: {error}")
            temporary.unlink(missing_ok=True)
    raise RuntimeError("All WMAP download mirrors failed:\n" + "\n".join(errors))


def download_data(data_dir: str | Path = "data", *, filaments: bool = True, cmb: bool = True):
    """One-time explicit download, suitable for calling from a Jupyter cell."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    with _session() as session:
        if filaments:
            result["filaments"] = download_filaments(session, data_dir / "tempel_sdss_filament_points.csv")
        if cmb:
            result["cmb"] = download_wmap(session, data_dir / "wmap_ilc_9yr_v5.fits")
    return result


# ---------------------------------------------------------------------------
# Real SDSS galaxies (Tempel et al. 2014, VizieR table3)
# ---------------------------------------------------------------------------
from .galaxies import GALAXY_COLUMNS, EXPECTED_GALAXIES

GALAXY_CATALOG_ID = "J/MNRAS/438/3465/table3"


def _request_galaxy_table(session: requests.Session, host: str,
                          low: int | None = None, high: int | None = None) -> pd.DataFrame:
    params = {
        "-source": GALAXY_CATALOG_ID,
        "-out": ",".join(GALAXY_COLUMNS),
        "-out.max": "unlimited",
    }
    if low is not None:
        params["IDgal"] = f"{low}..{high}"
    resp = session.get(f"https://{host}/viz-bin/asu-tsv",
                       params=params, timeout=(25, 180))
    resp.raise_for_status()
    if "<html" in resp.text[:1500].lower():
        raise ValueError("VizieR returned HTML instead of its tab-separated galaxy table")
    raw = pd.read_csv(StringIO(resp.text), sep="\t", comment="#", dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = set(GALAXY_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Galaxy VizieR table missing columns: {sorted(missing)}")
    raw = raw[GALAXY_COLUMNS].copy()
    for col in GALAXY_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["IDgal"])
    if raw.IDgal.duplicated().any():
        raise ValueError("Duplicated galaxy IDs in VizieR response")
    if low is not None and not raw.IDgal.between(low, high).all():
        raise ValueError("VizieR ignored the requested IDgal range")
    return raw


def download_galaxies(data_dir: str | Path = "data") -> Path:
    """Download the complete 576,493-galaxy *observed* SDSS table once.

    Saves original numeric fields as a local CSV. If the full-table request is
    blocked, downloads resumable IDgal chunks from CDS or Harvard mirrors.
    No invented records, and NO incomplete cache masquerading as complete.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "tempel_sdss_galaxies.csv"
    if csv_path.exists():
        from .galaxies import load_galaxies
        # Also checks the original raw file row count.
        galaxies = load_galaxies(csv_path, require_full=True)
        print(f"Using existing real galaxy catalog: {csv_path} ({len(galaxies):,} usable positions)")
        return csv_path

    def validate(df):
        if len(df) != EXPECTED_GALAXIES:
            raise ValueError(f"Partial catalog ({len(df):,}/{EXPECTED_GALAXIES:,} galaxies)")
        if df.IDgal.duplicated().any():
            raise ValueError("Galaxy catalog has duplicated IDs")
        return df.sort_values("IDgal", kind="stable").reset_index(drop=True)

    errors = []
    with _session() as s:
        for host in VIZIER_HOSTS:
            try:
                print(f"Downloading entire real galaxy catalog from {host}...")
                df = validate(_request_galaxy_table(s, host))
                tmp = csv_path.with_suffix(".csv.part")
                df.to_csv(tmp, index=False)
                tmp.replace(csv_path)
                print(f"Saved {len(df):,} SDSS galaxies: {csv_path}")
                return csv_path
            except Exception as err:
                errors.append(f"{host}: {type(err).__name__}: {err}")
                print("Complete request failed; trying alternate mirror/ranges.", err)

        # Keep partial chunks between interrupted runs; download IDs in bounded sets.
        parts_dir = data_dir / "_vizier_galaxy_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        batch = 20_000
        for low in range(1, EXPECTED_GALAXIES+1, batch):
            high = min(low+batch-1, EXPECTED_GALAXIES)
            part_file = parts_dir / f"IDgal_{low:06d}_{high:06d}.csv"
            if part_file.exists():
                part = pd.read_csv(part_file)
            else:
                part = None
                for host in VIZIER_HOSTS:
                    try:
                        part = _request_galaxy_table(s, host, low, high)
                        part.to_csv(part_file, index=False)
                        break
                    except Exception as err:
                        errors.append(f"{host} IDs {low}-{high}: {err}")
                if part is None:
                    raise RuntimeError(
                        f"Failed galaxy table3 range {low}–{high}. Already downloaded chunks "
                        f"remain in {parts_dir}; rerun this function to resume. "
                        "Or download table3 manually from VizieR as a CSV.\n"
                        + "\n".join(errors[-4:])
                    )
                time.sleep(.10)
            parts.append(part)
            print(f"Galaxy records: {low:,}–{high:,}")
        df = validate(pd.concat(parts, ignore_index=True))
        temp = csv_path.with_suffix(".csv.part")
        df.to_csv(temp, index=False)
        temp.replace(csv_path)
        print(f"Saved complete real galaxy catalog: {csv_path}")
        return csv_path
