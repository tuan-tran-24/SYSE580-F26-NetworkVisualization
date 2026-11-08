"""Read genuine WMAP/Planck HEALPix CMB files and prepare sky projections."""
from pathlib import Path
import warnings
import numpy as np
from .config import VizConfig


def _healpy():
    try:
        import healpy as hp
    except ImportError as err:
        raise ImportError(
            "CMB rendering requires healpy. On Windows install in a writable "
            "Conda env using: conda install -c conda-forge healpy"
        ) from err
    return hp


def load_cmb(cfg: VizConfig) -> np.ndarray:
    """Return observed CMB temperature anisotropy map in millikelvin."""
    hp = _healpy()
    if cfg.cmb_source == "WMAP9":
        path = cfg.wmap_fits
        multiplier = 1.0  # WMAP ILC thermodynamic mK
    elif cfg.cmb_source == "PLANCK_LOCAL":
        path = cfg.planck_fits
        multiplier = 1000.0  # Planck SMICA thermodynamic K -> mK
    else:
        raise ValueError("cmb_source must be 'WMAP9' or 'PLANCK_LOCAL'")

    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Missing CMB FITS: {Path(path).resolve()}\n"
            "Copy your existing WMAP/Planck FITS to data/; no automatic download occurs."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp = hp.read_map(str(path), field=0, dtype=np.float32, nest=False)
    temp = np.asarray(temp, dtype=np.float32)
    valid = np.isfinite(temp) & (temp > -1e29)  # remove HEALPix UNSEEN before rescaling
    if not valid.any():
        raise ValueError(f"CMB map has no valid samples: {path}")
    temp[~valid] = hp.UNSEEN
    temp[valid] *= multiplier
    return temp


def _interpolate_healpix(cmb_mk: np.ndarray, theta, phi):
    """Sample observed CMB and mask missing pixels *before* interpolating.

    Reject any interpolation with appreciable influence from unseen input
    pixels; no synthetic temperature values or inpainting are introduced.
    """
    hp = _healpy()
    measured = np.asarray(cmb_mk, dtype=np.float32)
    valid = np.isfinite(measured) & (measured > -1e29)
    if not np.any(valid):
        raise ValueError("CMB map contains no valid measured pixels")
    safe = np.where(valid, measured, 0.0)
    values = hp.get_interp_val(safe, theta, phi, nest=False)
    valid_fraction = hp.get_interp_val(valid.astype(np.float32), theta, phi, nest=False)
    missing = (~np.isfinite(values)) | (valid_fraction < 0.999)
    return np.ma.array(values, mask=missing)


def sky_texture(cmb_mk: np.ndarray, *, nx=1024, ny=512):
    """Full-sky temperature reprojected onto a 2D equirectangular texture."""
    lon = np.linspace(0, 2*np.pi, nx, endpoint=True)
    lat = np.linspace(-np.pi/2, np.pi/2, ny)
    ll, bb = np.meshgrid(lon, lat)
    return _interpolate_healpix(cmb_mk, np.pi/2-bb, ll)


def sphere_surface(cmb_mk: np.ndarray, cfg: VizConfig):
    """Full-sky CMB mapped onto a unit sphere with a CLOSED longitude seam.

    Phi includes both 0 and 2π, giving a full 360-degree surface. The
    number of display vertices is separate from the underlying FITS pixels.
    """
    if cfg.cmb_theta_resolution < 20 or cfg.cmb_phi_resolution < 40:
        raise ValueError("CMB display resolution is too low")
    theta = np.linspace(0, np.pi, cfg.cmb_theta_resolution)
    phi = np.linspace(0, 2*np.pi, cfg.cmb_phi_resolution)
    PHI, THETA = np.meshgrid(phi, theta)
    temp = _interpolate_healpix(cmb_mk, THETA, PHI)
    xyz = (np.sin(THETA)*np.cos(PHI),
           np.sin(THETA)*np.sin(PHI),
           np.cos(THETA))
    return xyz, temp
