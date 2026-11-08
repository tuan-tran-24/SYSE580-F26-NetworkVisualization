"""Central visualization configuration; ALL input data come from local files."""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VizConfig:
    data_dir: Path = Path('data')
    output_dir: Path = Path('outputs')
    width: int = 1920
    height: int = 1080
    dpi: int = 120

    # Astronomical distances: input coords are comoving h^-1 Mpc, display Mly.
    hubble_h: float = 0.674
    strict_full_catalog: bool = True
    max_galaxy_filament_distance: float = 1.0  # in h^-1 Mpc, as published

    # Real CMB: either local WMAP9 or local Planck temperature FITS.
    cmb_source: str = 'WMAP9'      # 'WMAP9' or 'PLANCK_LOCAL'
    planck_filename: str = 'COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits'
    cmb_temperature_range_mk: float = 0.30
    cmb_colorscale: str = 'Turbo' # false-color for measured Delta T, NOT visible-light colors
    cmb_surface_opacity: float = 0.78  # lower values may cause 3D sorting artifacts
    cmb_theta_resolution: int = 241
    cmb_phi_resolution: int = 481
    show_cmb_backdrop: bool = True    # decorative only; NOT spatially co-registered
    cmb_backdrop_alpha: float = 0.13
    show_illustrative_cmb_shell_3d: bool = True  # arbitrary shell; not a spatially aligned cosmic distance
    cmb_shell_opacity: float = 0.16

    background_color: str = '#000207'
    max_links_2d: int = 0
    max_galaxies_3d: int = 0
    max_links_3d: int = 0
    max_3d_filaments: int = 0
    galaxy_labels: bool = True     
    show_galaxy_colorbar: bool = True
    show_local_window: bool = True
    local_window_width_mly: float = 10
    show_scale_guide: bool = True
    show_title: bool = True

    # Optional real-coordinate crop, input in h^-1 Mpc; None = full survey.
    view_width_mpc_h: float | None = None
    view_center_x: float | None = None
    view_center_y: float | None = None

    @property
    def filament_csv(self) -> Path:
        return Path(self.data_dir) / 'tempel_sdss_filament_points.csv'

    @property
    def galaxy_csv(self) -> Path:
        return Path(self.data_dir) / 'tempel_sdss_galaxies.csv'

    @property
    def galaxy_names_csv(self) -> Path:
        return Path(self.data_dir) / 'galaxy_names.csv'

    @property
    def wmap_fits(self) -> Path:
        return Path(self.data_dir) / 'wmap_ilc_9yr_v5.fits'

    @property
    def planck_fits(self) -> Path:
        return Path(self.data_dir) / self.planck_filename

    @property
    def figure_size(self) -> tuple[float,float]:
        return (self.width / self.dpi, self.height / self.dpi)

    @property
    def mly_per_hinv_mpc(self) -> float:
        if self.hubble_h <= 0:
            raise ValueError('hubble_h must be positive')
        return 3.26156 / self.hubble_h

    def ensure_output_dir(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
