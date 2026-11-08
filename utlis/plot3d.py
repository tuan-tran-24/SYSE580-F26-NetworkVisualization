"""Full-sky CMB as a false-color HEALPix sphere; local files only.

The CMB has measured ANGULAR temperature fluctuations. A display sphere
radius is NOT the physical distance to last scattering. A composite shell
around SDSS is decorative, not physically co-registered cosmology.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .cmb import sphere_surface
from .config import VizConfig


def _cmb_colorbar(cfg):
    return dict(title=dict(text='ΔT (mK)'), thickness=14, len=.67)


def make_cmb_shell_trace(cmb_mk,cfg:VizConfig,*,radius:float):
    """Decorative transparent CMB backdrop around real SDSS data.

    radius is a chosen DISPLAY distance, not a measured CMB radius.
    Original galaxy and filament coordinates are NOT modified.
    """
    import plotly.graph_objects as go
    (sx,sy,sz),temp=sphere_surface(cmb_mk,cfg)
    vals=np.asarray(np.ma.filled(temp,np.nan),dtype=np.float32)
    if np.isfinite(vals).mean()<.95:
        raise ValueError('CMB coverage below 95%; check local FITS map')
    return go.Surface(x=sx*radius,y=sy*radius,z=sz*radius,surfacecolor=vals,
        colorscale=cfg.cmb_colorscale,cmin=-cfg.cmb_temperature_range_mk,
        cmax=cfg.cmb_temperature_range_mk,opacity=cfg.cmb_shell_opacity,
        showscale=False,showlegend=True,name='CMB shell (illustrative, NOT to scale)',
        hovertemplate='Observed CMB ΔT %{surfacecolor:.4f} mK<extra></extra>',
        lighting=dict(ambient=1.0,diffuse=0,specular=0,roughness=1.0))


def render_cmb_plotly(cmb_mk,traces,cfg:VizConfig,*,return_figure=True):
    """Interactive CMB sky sphere. `traces` ignored, not co-registered."""
    import plotly.graph_objects as go
    cfg.ensure_output_dir()
    (sx,sy,sz),temp=sphere_surface(cmb_mk,cfg)
    vals=np.asarray(np.ma.filled(temp,np.nan),dtype=np.float32)
    coverage=np.isfinite(vals).mean()
    if coverage<.95: raise ValueError(f'CMB full-sky valid mesh coverage: {coverage:.1%}')
    fig=go.Figure(go.Surface(
        x=sx,y=sy,z=sz,surfacecolor=vals,
        colorscale=cfg.cmb_colorscale,cmin=-cfg.cmb_temperature_range_mk,
        cmax=cfg.cmb_temperature_range_mk,opacity=cfg.cmb_surface_opacity,
        showscale=True,colorbar=_cmb_colorbar(cfg),
        name=f'{cfg.cmb_source} observed ΔT',showlegend=False,
        lighting=dict(ambient=1.0,diffuse=0,specular=0,roughness=1.0),
        hovertemplate='CMB ΔT %{surfacecolor:.4f} mK<extra></extra>'))
    fig.update_layout(
        width=cfg.width,height=cfg.height,paper_bgcolor=cfg.background_color,
        font=dict(color='#F0F3F5'),margin=dict(l=0,r=0,t=0,b=0),
        scene=dict(bgcolor=cfg.background_color,aspectmode='cube',
                   xaxis=dict(visible=False,range=[-1.04,1.04]),
                   yaxis=dict(visible=False,range=[-1.04,1.04]),
                   zaxis=dict(visible=False,range=[-1.04,1.04]),
                   camera=dict(eye=dict(x=1.6,y=1.35,z=.85)),dragmode='orbit'))
    path=Path(cfg.output_dir)/'cosmic_web_cmb_3d.html'
    fig.write_html(str(path),include_plotlyjs=True,full_html=True)
    return (path,fig) if return_figure else path


def render_sdss_filaments_plotly(traces,cfg:VizConfig,*,return_figure=False):
    """Standalone original 3D reconstructed filaments in comoving million ly."""
    import plotly.graph_objects as go
    cfg.ensure_output_dir()
    chosen=sorted(traces,key=lambda t:len(t[1]),reverse=True)
    if cfg.max_3d_filaments>0: chosen=chosen[:cfg.max_3d_filaments]
    xx,yy,zz=[],[],[]
    for _,xyz,_ in chosen:
        xyz=np.asarray(xyz,dtype=float)*cfg.mly_per_hinv_mpc
        xx.extend(xyz[:,0].tolist()+[None]);yy.extend(xyz[:,1].tolist()+[None]);zz.extend(xyz[:,2].tolist()+[None])
    fig=go.Figure(go.Scatter3d(x=xx,y=yy,z=zz,mode='lines',
                               name='SDSS filament spines',line=dict(color='#B7D5E6',width=2),
                               hoverinfo='skip'))
    fig.update_layout(width=cfg.width,height=cfg.height,
        paper_bgcolor=cfg.background_color,font=dict(color='#EFF4FA'),
        margin=dict(l=0,r=0,t=0,b=0),
        scene=dict(bgcolor=cfg.background_color,aspectmode='data',
                   xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)))
    path=Path(cfg.output_dir)/'cosmic_web_sdss_filaments_3d.html'
    fig.write_html(str(path),include_plotlyjs=True,full_html=True)
    return (path,fig) if return_figure else path


def render_cmb_static(cmb_mk,traces,cfg:VizConfig)->Path:
    """Standalone CMB sphere PNG at exact user-selected pixel dimensions."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from PIL import Image
    cfg.ensure_output_dir()
    (sx,sy,sz),temp=sphere_surface(cmb_mk,cfg)
    vals=np.asarray(np.ma.filled(temp,np.nan),dtype=float)
    if np.isfinite(vals).mean()<.95:
        raise ValueError('CMB texture coverage below 95%')
    norm=Normalize(-cfg.cmb_temperature_range_mk,cfg.cmb_temperature_range_mk)
    rgba=colormaps[cfg.cmb_colorscale.lower()](norm(vals))
    # Static 3D surfaces are opaque for correct depth rendering, whereas the
    # interactive Plotly sphere can be partially transparent.
    rgba[...,3]=1.0
    fig=plt.figure(figsize=cfg.figure_size,dpi=cfg.dpi,facecolor=cfg.background_color)
    ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    ax.set_facecolor(cfg.background_color); ax.set_axis_off()
    ax.set_box_aspect((1,1,1));fig.subplots_adjust(0,0,1,1)
    ax.plot_surface(sx,sy,sz,facecolors=rgba,rstride=1,cstride=1,
                    linewidth=0,antialiased=False,shade=False,zorder=1)
    ax.view_init(elev=22,azim=132)
    ax.set(xlim=(-1.06,1.06),ylim=(-1.06,1.06),zlim=(-1.06,1.06))
    sm=ScalarMappable(norm=norm,cmap=cfg.cmb_colorscale.lower());sm.set_array([])
    cax=fig.add_axes([.88,.23,.015,.53]);cb=fig.colorbar(sm,cax=cax)
    cb.set_label('CMB ΔT (mK)',color='#E4E8EF',size=10)
    cb.ax.tick_params(colors='#E4E8EF',labelsize=9);cb.outline.set_edgecolor('#67717D')
    out=Path(cfg.output_dir)/'cosmic_web_cmb_3d.png'
    fig.savefig(out,dpi=cfg.dpi,facecolor=cfg.background_color,pad_inches=0)
    plt.close(fig)
    with Image.open(out) as im:
        if im.size!=(cfg.width,cfg.height):
            raise RuntimeError(f'PNG size {im.size}, expected {(cfg.width,cfg.height)}')
    return out
