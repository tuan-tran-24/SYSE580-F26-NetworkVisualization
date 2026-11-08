"""2D SDSS filament map, optional CMB backdrop, Milky Way locator, and scale."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from pathlib import Path

from .config import VizConfig
from .catalog import (make_traces, density_normalizer,
                      footprint_limits, community_colors)
from .cmb import sky_texture


def _add_filaments(ax, traces, strength, color, *, alpha_scale=1., width_scale=1., zorder=3, color_by_id=None):
    segments, segment_rgba, linewidths = [], [], []
    for fid, coords, dens in traces:
        if len(coords) < 2:
            continue
        line_segments = np.stack([coords[:-1, :2], coords[1:, :2]], axis=1)
        scores = strength((dens[:-1] + dens[1:]) / 2)
        base = to_rgba(color_by_id.get(fid, color) if color_by_id else color)
        segments.append(line_segments)
        segment_rgba.extend([(base[0], base[1], base[2], float((.035+.40*q)*alpha_scale))
                             for q in scores])
        linewidths.extend((.30 + .80*scores)*width_scale)
    if segments:
        ax.add_collection(LineCollection(np.concatenate(segments), colors=segment_rgba,
                                         linewidths=linewidths, zorder=zorder))


def _draw_scale(ax, limits, wanted=100.0):
    xmin, xmax, ymin, ymax = limits
    span_x = xmax-xmin
    length = wanted
    if length > .26 * span_x:
        length = min([1, 2, 5, 10, 20, 50, 100, 200, 500],
                     key=lambda n: abs(n - .13*span_x))
    x, y = xmin+.055*span_x, ymin+.075*(ymax-ymin)
    tick = .007*(ymax-ymin)
    ax.plot([x,x+length],[y,y],color="#e6e8eb",lw=2,zorder=30)
    ax.plot([x,x],[y-tick,y+tick],color="#e6e8eb",lw=1.4,zorder=30)
    ax.plot([x+length,x+length],[y-tick,y+tick],color="#e6e8eb",lw=1.4,zorder=30)
    ax.text(x,y+2.4*tick,f"{length:g} h⁻¹ Mpc",color="white",
            fontsize=10,ha="left",va="bottom",zorder=30)


def _milky_way_inset(fig, traces, cfg, strength):
    """True catalog coordinates within a 3D radius of the observer."""
    ax = fig.add_axes([.76, .65, .215, .27], facecolor="#050b13")
    r = cfg.locator_half_width
    ax.set_xlim(-r,r)
    ax.set_ylim(-r*.706,r*.706)
    ax.set_aspect("equal",adjustable="box")
    pieces = []
    for fid, pts, den in traces:
        inside = np.linalg.norm(pts, axis=1) <= r
        if np.count_nonzero(inside) < 2:
            continue
        for idx in np.split(np.arange(len(inside)), np.flatnonzero(np.diff(inside.astype(int)))+1):
            if len(idx) >= 2 and inside[idx[0]]:
                pieces.append((fid,pts[idx],den[idx]))
    if pieces:
        _add_filaments(ax,pieces,strength,cfg.filament_color,alpha_scale=.90,width_scale=.80,zorder=1)
    ax.scatter([0],[0],s=135,marker="*",color="#ffdea0",zorder=10)
    ax.text(.5,1.06,"MILKY WAY · LOCAL VIEW",color="#e6e9ee",fontsize=9,
            ha="center",transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_edgecolor("#687382")
    ax.set_xticks([]); ax.set_yticks([])


def render_cosmic_web_2d(df: pd.DataFrame, cmb_mk: np.ndarray, cfg: VizConfig) -> Path:
    """Export a PNG of full real SDSS filament spine geometry (no synthetic points)."""
    cfg.ensure_output_dir()
    traces = make_traces(df)
    strength = density_normalizer(df)
    limits = footprint_limits(df,cfg.width,cfg.height)
    xmin,xmax,ymin,ymax = limits

    # Scientific point locations; community colors are a graph-derived visual choice.
    colors = community_colors(traces) if cfg.show_communities else None
    fig, ax = plt.subplots(figsize=cfg.figure_size,dpi=cfg.dpi,facecolor=cfg.background_color)
    ax.set_facecolor(cfg.background_color)
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.subplots_adjust(0,0,1,1)

    if cfg.show_cmb_backdrop:
        texture = sky_texture(cmb_mk)
        ax.imshow(texture,origin="lower",extent=limits,cmap="RdBu_r",
                  vmin=-cfg.cmb_temperature_range_mk,vmax=cfg.cmb_temperature_range_mk,
                  alpha=cfg.cmb_backdrop_alpha,interpolation="bilinear",zorder=0)

    _add_filaments(ax,traces,strength,cfg.filament_color,alpha_scale=.34,
                   width_scale=4.2,zorder=2,color_by_id=colors)  # glow
    _add_filaments(ax,traces,strength,cfg.filament_color,alpha_scale=.94,
                   width_scale=.85,zorder=3,color_by_id=colors)  # ridge cores

    if cfg.show_nodes:
        anchor = df.groupby("ID",sort=False).head(1)
        subset = pd.concat([df.iloc[::max(1,cfg.sample_step)],anchor],ignore_index=True)
        subset = subset.drop_duplicates(["ID", "IDpt"])
        scores = strength(subset.fden.to_numpy())
        sizes = .7 + 4*scores
        xx,yy = subset.x.to_numpy(),subset.y.to_numpy()
        ax.scatter(xx,yy,s=sizes*9,color=cfg.node_color,alpha=.012,
                   linewidths=0,rasterized=True,zorder=5)
        shades = [colors.get(int(fid),cfg.node_color) for fid in subset.ID] if colors else cfg.node_color
        ax.scatter(xx,yy,s=sizes,color=shades,alpha=.55,
                   linewidths=0,rasterized=True,zorder=6)

    if cfg.show_milky_way_locator:
        _milky_way_inset(fig,traces,cfg,strength)
    _draw_scale(ax,limits,cfg.scale_length)

    out = Path(cfg.output_dir) / "cosmic_web_1920x1080.png"
    fig.savefig(out,facecolor=cfg.background_color,dpi=cfg.dpi,pad_inches=0)
    plt.close(fig)
    from PIL import Image
    with Image.open(out) as im:
        if im.size != (cfg.width,cfg.height):
            raise RuntimeError(f"Saved image size {im.size}, expected {(cfg.width,cfg.height)}")
    return out
