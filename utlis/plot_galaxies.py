"""Render observed SDSS galaxies and reconstructed Tempel cosmic filaments.

The 2D rendering is a true 1:1 Cartesian XY projection in comoving
h^-1 Mpc. The physical scale bar is linear and labelled in million ly.
All observations come from already-loaded local catalogs; no downloads or
artificial galaxy positions are used.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .catalog import make_traces, density_normalizer
from .galaxies import choose_display_links
from .config import VizConfig
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Actual LaTeX typesetting: on Windows use MiKTeX's latex.exe, dvipng.exe,
# and supporting programs. This deliberately does NOT silently fall back to
# Matplotlib mathtext or the bundled cmr10 font.
def _configure_fonts():
    import os
    import shutil
    import matplotlib as mpl

    # Optional for Windows installations not registered on PATH:
    #   os.environ["MIKTEX_BIN"] = r"C:\Program Files\MiKTeX\miktex\bin\x64"
    miktex_bin = os.environ.get('MIKTEX_BIN', '').strip()
    if miktex_bin:
        if not Path(miktex_bin).is_dir():
            raise FileNotFoundError(f'MIKTEX_BIN does not exist: {miktex_bin}')
        os.environ['PATH'] = miktex_bin + os.pathsep + os.environ.get('PATH', '')

    missing = [program for program in ('latex', 'dvipng') if shutil.which(program) is None]
    if missing:
        raise RuntimeError(
            'Actual LaTeX rendering requires MiKTeX. Missing: '
            + ', '.join(missing)
            + '. Install MiKTeX, enable missing-package installation, '
              'and add its bin folder to PATH (or set MIKTEX_BIN before running the notebook). '
              'Restart the Jupyter kernel afterward.'
        )

    mpl.rcParams.update({
        'text.usetex': True,               # Run the real latex.exe engine
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'axes.unicode_minus': False,
        'axes.formatter.use_mathtext': True,
    })


def _latex_escape(value) -> str:
    """Render catalog names safely as literal text through MiKTeX."""
    import re
    import unicodedata
    s = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode('ascii')
    # Escape TeX metacharacters, preserving the actual catalog text.
    return re.sub(r'([\\#$%&_{}])', lambda m: r'\textbackslash{}' if m.group(1) == '\\'
                  else '\\' + m.group(1), s)


def _limits(galaxies: pd.DataFrame, cfg: VizConfig):
    x = galaxies.x_gal.to_numpy(float)
    y = galaxies.y_gal.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        raise ValueError('No valid observed galaxy positions')
    x, y = x[valid], y[valid]
    if cfg.view_width_mpc_h is not None:
        cx = cfg.view_center_x if cfg.view_center_x is not None else float(np.median(x))
        cy = cfg.view_center_y if cfg.view_center_y is not None else float(np.median(y))
        w = float(cfg.view_width_mpc_h)
        if w <= 0:
            raise ValueError('view_width_mpc_h must be positive')
    else:
        xmin, xmax, ymin, ymax = float(x.min()), float(x.max()), float(y.min()), float(y.max())
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        w = max(xmax-xmin, (ymax-ymin)*cfg.width/cfg.height, 1) * 1.035
    h = w * cfg.height / cfg.width
    return (cx-w/2, cx+w/2, cy-h/2, cy+h/2)


def _galaxy_sizes(galaxies: pd.DataFrame):
    luminosity = galaxies.Lr.to_numpy(float)
    luminosity = np.where(np.isfinite(luminosity) & (luminosity > 0), luminosity, 0)
    return np.clip(0.75 + 1.75*np.log1p(luminosity), 0.8, 11.0)


def _galaxy_colors(galaxies: pd.DataFrame):
    """Measured g-r optical index mapped to illustrative display RGB."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    gr = (galaxies.gmag - galaxies.rmag).to_numpy(float)
    cmap = LinearSegmentedColormap.from_list(
        'galaxy_optical_color', ['#74AEE0','#C0D4E4','#F5F0DD','#E9B47D','#C9825F'])
    norm = Normalize(vmin=.15, vmax=1.10, clip=True)
    colors = cmap(norm(np.where(np.isfinite(gr), gr, .55)))
    colors[~np.isfinite(gr)] = [.72,.74,.76,1]
    return colors, cmap, norm


def _spine_segments(filaments: pd.DataFrame):
    normalizer = density_normalizer(filaments)
    parts, strengths = [], []
    for _, xyz, fden in make_traces(filaments):
        if len(xyz) < 2:
            continue
        parts.append(np.stack((xyz[:-1,:2], xyz[1:,:2]), axis=1))
        strengths.append(normalizer((fden[:-1] + fden[1:])/2))
    if not parts:
        raise ValueError('No reconstructed filament segments found in local catalog')
    return np.concatenate(parts), np.concatenate(strengths)


def _draw_spines(ax, segments, strengths, *, zorder=2, glow=True):
    from matplotlib.collections import LineCollection
    colors = np.tile([.75,.86,.93,1.], (len(strengths),1))
    colors[:,3] = .09 + .37*strengths
    if glow:
        halo = colors.copy()
        halo[:,3] *= .14
        ax.add_collection(LineCollection(segments, colors=halo, linewidths=2.7,
                                         zorder=zorder, rasterized=True))
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=.45,
                                     zorder=zorder+1, rasterized=True))


def _draw_scale(ax, bounds, cfg: VizConfig):
    r"""One truly linear ruler with ticks at 100, 200, and 500 Mly."""
    if not cfg.show_scale_guide:
        return
    span_mly = (bounds[1]-bounds[0]) * cfg.mly_per_hinv_mpc
    if not np.isfinite(span_mly) or span_mly <= 0:
        raise ValueError('Invalid physical scale bounds')
    mark_values = (100.0, 200.0, 500.0)
    usable = [v for v in mark_values if v <= span_mly * 0.58]
    if not usable:
        return
    end = usable[-1]
    x0, y0 = 0.045, 0.100
    col = '#F4F1E8'
    t = ax.transAxes
    length_frac = end / span_mly
    ax.text(x0, y0+.025,
            r'\textbf{million light-years}',
            color=col, fontsize=12,
            ha='left', va='bottom', transform=t, zorder=120)
    ax.plot([x0, x0+length_frac], [y0, y0], transform=t,
            color=col, lw=1.6, solid_capstyle='butt', zorder=120)
    ax.plot([x0,x0], [y0-.007,y0+.007], transform=t, color=col, lw=1.25, zorder=120)
    for value in usable:
        xp = x0 + value/span_mly
        ax.plot([xp,xp], [y0-.007,y0+.007], transform=t,
                color=col, lw=1.25, zorder=120)
        ax.text(xp, y0-.018, rf'$\,{int(value):d}$', transform=t,
                color=col, fontsize=10, ha='center', va='top', zorder=121)

def _local_center(galaxies: pd.DataFrame, bounds, cfg: VizConfig):
    """Pick a dense REAL region inside the same area as the separate 2D closeup."""
    x = galaxies.x_gal.to_numpy(float)
    y = galaxies.y_gal.to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[valid], y[valid]
    center_x = float(cfg.view_center_x if cfg.view_center_x is not None else np.median(xx))
    center_y = float(cfg.view_center_y if cfg.view_center_y is not None else np.median(yy))
    closeup_width = cfg.view_width_mpc_h or 100.0
    inside = ((np.abs(xx-center_x) <= closeup_width/2) &
              (np.abs(yy-center_y) <= closeup_width/2))
    if not np.any(inside):
        inside = ((xx>=bounds[0])&(xx<=bounds[1])&
                  (yy>=bounds[2])&(yy<=bounds[3]))
    if not np.any(inside):
        raise ValueError('No observed galaxies inside the local zoom region')
    xx, yy = xx[inside], yy[inside]
    h, xe, ye = np.histogram2d(xx, yy, bins=(16,12))
    i,j = np.unravel_index(int(np.argmax(h)), h.shape)
    in_cell = (xx>=xe[i]) & (xx<=xe[i+1]) & (yy>=ye[j]) & (yy<=ye[j+1])
    return (float(np.median(xx[in_cell])), float(np.median(yy[in_cell])))


def _local_inset(ax, galaxies, segments, strengths, bounds, cfg, colors, sizes, cmb_mk=None):
    """A close-in real-data view matching the close-up style, with CMB backdrop."""
    from matplotlib.patches import Rectangle
    cx,cy = _local_center(galaxies, bounds, cfg)
    # Existing cfg default is 350 Mly. The new default effective zoom
    # is 160 Mly, and can be tightened further via cfg.local_window_width_mly.
    width_mly = min(float(cfg.local_window_width_mly), 60.0)
    if width_mly <= 0:
        raise ValueError('local_window_width_mly must be positive')
    # The inset has a 16:9-ish shape determined by its actual screen footprint.
    screen_box = (.715, .060, .257, .290)  # bottom right
    physical_ratio = (screen_box[3]*cfg.height) / (screen_box[2]*cfg.width)
    w = width_mly/cfg.mly_per_hinv_mpc
    h = w*physical_ratio
    l,r,b,t = cx-w/2, cx+w/2, cy-h/2, cy+h/2

    inset = ax.inset_axes(screen_box, facecolor='#03070B', zorder=42)
    inset.set(xlim=(l,r), ylim=(b,t))
    inset.set_aspect('equal', adjustable='box')
    inset.set_xticks([])
    inset.set_yticks([])
    if cfg.show_cmb_backdrop and cmb_mk is not None:
        from .cmb import sky_texture
        tex = sky_texture(cmb_mk, nx=640, ny=320)
        inset.imshow(tex, origin='lower', extent=(l,r,b,t),
                     cmap=cfg.cmb_colorscale.lower(),
                     vmin=-cfg.cmb_temperature_range_mk,
                     vmax=cfg.cmb_temperature_range_mk,
                     alpha=min(0.18, cfg.cmb_backdrop_alpha + 0.03),
                     zorder=0, interpolation='bilinear')
    mask = ((segments[:,:,0].max(axis=1)>=l) &
            (segments[:,:,0].min(axis=1)<=r) &
            (segments[:,:,1].max(axis=1)>=b) &
            (segments[:,:,1].min(axis=1)<=t))
    if np.any(mask):
        _draw_spines(inset, segments[mask], strengths[mask], zorder=2, glow=True)
    gx = galaxies.x_gal.to_numpy(float)
    gy = galaxies.y_gal.to_numpy(float)
    valid = np.isfinite(gx) & np.isfinite(gy)
    visible = valid & (gx>=l)&(gx<=r)&(gy>=b)&(gy<=t)
    if np.any(visible):
        inset.scatter(gx[visible],gy[visible],s=np.minimum(sizes[visible]*2.4,22),
                      c=colors[visible],alpha=.95,linewidths=0,
                      rasterized=True,zorder=5)
    inset.text(.045,.955,r'\textbf{Close-up region}',
               transform=inset.transAxes,va='top',ha='left',
               fontsize=13,color='#FCF7F0',zorder=40)
    # inset border: a rectangle in axes coordinates remains visible even
    # if the axes spines are obscured by Matplotlib aspect adjustments.
    inset.add_patch(Rectangle((0,0),1,1,transform=inset.transAxes,fill=False,
                              edgecolor='#D94B4B',linewidth=1.4,
                              zorder=80,clip_on=False))
    # Main-frame selection: SAME physical l/r/b/t as the zoom window.
    ax.add_patch(Rectangle((l,b),w,h,fill=False,lw=1.35,
                           ec='#D94B4B',alpha=0.95,zorder=38))
    ax.text(l, t + h*.08, r'\textbf{Zoom region}',
            fontsize=10, color='#D94B4B',
            ha='left',va='bottom',zorder=38)


def render_galaxy_filament_2d(galaxies: pd.DataFrame, filaments: pd.DataFrame,
                               cfg: VizConfig, *, cmb_mk=None, filename=None) -> Path:
    """Write 1920x1080 with real MiKTeX (text.usetex=True) typography."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from PIL import Image
    _configure_fonts()
    if 'associated' not in galaxies.columns:
        raise ValueError('Call attach_filament_associations before rendering')
    cfg.ensure_output_dir()
    bounds = _limits(galaxies, cfg)
    xmin, xmax, ymin, ymax = bounds
    fig, ax = plt.subplots(figsize=cfg.figure_size, dpi=cfg.dpi,
                           facecolor=cfg.background_color)
    ax.set_facecolor(cfg.background_color)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    try:
        if cfg.show_cmb_backdrop:
            if cmb_mk is None:
                raise ValueError('CMB backdrop requested; supply your locally loaded CMB map')
            from .cmb import sky_texture
            tex = sky_texture(cmb_mk, nx=1280, ny=640)
            ax.imshow(tex, origin='lower', extent=bounds,
                      cmap=cfg.cmb_colorscale.lower(),
                      vmin=-cfg.cmb_temperature_range_mk,
                      vmax=cfg.cmb_temperature_range_mk,
                      alpha=cfg.cmb_backdrop_alpha,zorder=0,
                      interpolation='bilinear')
        segments, strengths = _spine_segments(filaments)
        _draw_spines(ax, segments, strengths, zorder=2)

        gx,gy = galaxies.x_gal.to_numpy(float),galaxies.y_gal.to_numpy(float)
        visible = (np.isfinite(gx)&np.isfinite(gy)&
                   (gx>=xmin)&(gx<=xmax)&(gy>=ymin)&(gy<=ymax))
        sizes = _galaxy_sizes(galaxies)
        colors, cmap, norm = _galaxy_colors(galaxies)
        ax.scatter(gx[visible],gy[visible],s=sizes[visible]*3.5,
                   c=colors[visible],alpha=.018,linewidths=0,
                   rasterized=True,zorder=5)
        ax.scatter(gx[visible],gy[visible],s=sizes[visible],
                   c=colors[visible],alpha=.68,linewidths=0,
                   rasterized=True,zorder=6)
        links=choose_display_links(galaxies, limit=cfg.max_links_2d)
        if not links.empty:
            lines=np.stack((links[['x_gal','y_gal']].to_numpy(float),
                            links[['filament_x','filament_y']].to_numpy(float)),axis=1)
            ax.add_collection(LineCollection(lines, colors=[(.95,.78,.53,.22)],
                                             linewidths=.38,zorder=7,rasterized=True))
        if cfg.galaxy_labels and 'has_verified_name' in galaxies.columns:
            named = galaxies.loc[visible & galaxies.has_verified_name].nlargest(12,'Lr')
            for row in named.itertuples():
                ax.annotate(_latex_escape(row.display_name),(row.x_gal,row.y_gal),
                            xytext=(3,4),textcoords='offset points',
                            fontsize=9,color='#F5F2EA',zorder=30)
        if cfg.show_title:
            ax.text(.028,.964,r'\textbf{Cosmic Web}',transform=ax.transAxes,
                    ha='left',va='top',fontsize=38,color='#F7F5F0',
                    zorder=100)
            ax.text(.030,.906,r'\textit{Observed galaxies and reconstructed filaments}',
                    transform=ax.transAxes,ha='left',va='top',fontsize=15,
                    color='#D6E1E7',zorder=100)
        _draw_scale(ax,bounds,cfg)
        if cfg.show_local_window:
            _local_inset(ax,galaxies,segments,strengths,bounds,cfg,colors,sizes,cmb_mk=cmb_mk)
        if cfg.show_galaxy_colorbar:
            sm=ScalarMappable(norm=norm,cmap=cmap)
            cax=fig.add_axes([.931,.685,.010,.225])
            cb=fig.colorbar(sm,cax=cax)
            cb.ax.tick_params(colors='#E5EDF3',labelsize=9,length=2)
            for tick_label in cb.ax.get_yticklabels():
                tick_label.set_fontfamily('serif')
            cb.outline.set_visible(False)
            cb.set_label(r'\textbf{Galaxy Color}',
                         color='#E5EDF3',size=12)
        legend_handles = [
            Line2D(
                [0], [0],
                marker='o',
                linestyle='None',
                markersize=7,
                markerfacecolor='#E9D7B0',
                markeredgewidth=0,
                alpha=0.9,
                label=r'\textbf{Nodes: Galaxies}'
            ),
            Line2D(
                [0], [0],
                color='#F5C98F',
                lw=1.2,
                alpha=0.5,
                label=r'\textbf{Edges: Galaxy--Filament}'
            ),
            Patch(
                facecolor='#8E63D2',
                edgecolor='none',
                alpha=0.30,
                label=r'\textbf{CMB Temperature Map}'
            ),
        ]
        
        legend = ax.legend(
            handles=legend_handles,
            loc='lower left',
            bbox_to_anchor=(0.045, 0.16),   # move if needed
            frameon=False,                  
            fontsize=10,
            handlelength=2.4,
            handletextpad=0.8,
            borderpad=0.0,
            labelspacing=0.7
        )
        for txt in legend.get_texts():
            txt.set_color('#F3F1EC')        
        name=(filename or ('galaxy_filament_closeup_1920x1080.png'
                           if cfg.view_width_mpc_h else 'galaxy_filament_full_1920x1080.png'))
        out=Path(cfg.output_dir)/name
        fig.savefig(out,dpi=cfg.dpi,facecolor=cfg.background_color,pad_inches=0)
    finally:
        plt.close(fig)
    with Image.open(out) as image:
        if image.size!=(cfg.width,cfg.height):
            raise RuntimeError(f'PNG size {image.size}, expected {(cfg.width,cfg.height)}')
    return out


def render_galaxy_filament_plotly(galaxies:pd.DataFrame,filaments:pd.DataFrame,
                                  cfg:VizConfig,*,cmb_mk=None,return_figure=False):
    """True-coordinate Plotly HTML; browser fonts are NOT typeset by MiKTeX."""
    import plotly.graph_objects as go
    cfg.ensure_output_dir()
    traces=sorted(make_traces(filaments),key=lambda t:len(t[1]),reverse=True)
    if cfg.max_3d_filaments>0: traces=traces[:cfg.max_3d_filaments]
    f=cfg.mly_per_hinv_mpc
    fx,fy,fz=[],[],[]
    for _,xyz,_ in traces:
        vals=xyz*f
        fx.extend(vals[:,0].tolist()+[None]);fy.extend(vals[:,1].tolist()+[None]);fz.extend(vals[:,2].tolist()+[None])
    fig=go.Figure()
    layer_indices = {}  # One dedicated Plotly trace per independently switchable layer.
    if cfg.show_illustrative_cmb_shell_3d:
        if cmb_mk is None:
            raise ValueError('3D CMB shell enabled, but no local CMB map was supplied')
        from .plot3d import make_cmb_shell_trace
        radius=float(np.nanmax(np.linalg.norm(galaxies[['x_gal','y_gal','z_gal']].to_numpy(float),axis=1)))*f*1.07
        cmb_trace = make_cmb_shell_trace(cmb_mk,cfg,radius=radius)
        cmb_trace.update(name='CMB background (illustrative)',
                         showlegend=True, legendgroup='cmb')
        layer_indices['CMB'] = len(fig.data)
        fig.add_trace(cmb_trace)
    layer_indices['Filaments'] = len(fig.data)
    fig.add_trace(go.Scatter3d(x=fx,y=fy,z=fz,mode='lines',
                               name='Reconstructed filaments',legendgroup='filaments',showlegend=True,
                               line=dict(color='#9BBBD0',width=2),hoverinfo='skip'))
    displayed=galaxies
    if cfg.max_galaxies_3d and len(displayed)>cfg.max_galaxies_3d:
        displayed=displayed.sample(n=cfg.max_galaxies_3d,random_state=42)
    gr=(displayed.gmag-displayed.rmag).to_numpy(float)
    gr=np.where(np.isfinite(gr),gr,.55)
    names=(displayed.display_name.astype(str).tolist() if 'display_name' in displayed
           else ['Unnamed SDSS galaxy']*len(displayed))
    custom=np.column_stack((names,displayed.IDgal.astype('Int64').astype(str).to_numpy(),
                             displayed.z.to_numpy(float),displayed.Dfil.to_numpy(float)*f))
    layer_indices['Galaxies'] = len(fig.data)
    fig.add_trace(go.Scatter3d(
        x=displayed.x_gal.to_numpy(float)*f,y=displayed.y_gal.to_numpy(float)*f,
        z=displayed.z_gal.to_numpy(float)*f,mode='markers',
        name=f'Observed galaxies ({len(displayed):,})',
        legendgroup='galaxies', showlegend=True,
        marker=dict(size=2,opacity=.67,color=gr,cmin=.15,cmax=1.10,
                    colorscale=[[0,'#74AEE0'],[.3,'#C0D4E4'],[.55,'#F5F0DD'],[1,'#C9825F']],
                    showscale=True,colorbar=dict(title='Galaxy Color',thickness=12)),
        customdata=custom,
        hovertemplate='<b>%{customdata[0]}</b><br>Catalog ID: %{customdata[1]}'
                      '<br>Redshift: %{customdata[2]:.4f}'
                      '<br>Filament distance: %{customdata[3]:.2f} million ly'
                      '<br>Comoving x: %{x:.1f} million ly'
                      '<br>Comoving y: %{y:.1f} million ly'
                      '<br>Comoving z: %{z:.1f} million ly<extra></extra>'
    ))
    links=choose_display_links(galaxies,limit=cfg.max_links_3d)
    if not links.empty:
        p=links[['x_gal','y_gal','z_gal']].to_numpy(float)*f
        q=links[['filament_x','filament_y','filament_z']].to_numpy(float)*f
        joined=np.stack((p,q,np.full_like(p,np.nan)),axis=1).reshape(-1,3)
        layer_indices['Links'] = len(fig.data)
        fig.add_trace(go.Scatter3d(x=joined[:,0],y=joined[:,1],z=joined[:,2],
                                   mode='lines',name='Galaxy–spine associations',
                                   legendgroup='links',showlegend=True,
                                   line=dict(color='rgba(245,201,143,0.35)',width=1),hoverinfo='skip'))
    # Four independent two-state buttons (where available). Clicking a layer
    # button once hides that layer; clicking the SAME button again shows it.
    # This uses Plotly's documented `args` / `args2` two-state mechanism.
    # Legend entries provide a second, built-in independent toggle.
    toggle_menus = []
    for position, label in enumerate(('Galaxies', 'Filaments', 'Links', 'CMB')):
        if label not in layer_indices:
            continue  # An optional layer was not drawn in this figure.
        trace_index = layer_indices[label]
        toggle_menus.append(dict(
            type='buttons', direction='right', showactive=True, active=-1,
            x=0.014, y=0.89 - position * 0.062,
            xanchor='left', yanchor='top',
            bgcolor='rgba(12,22,32,0.84)', bordercolor='rgba(190,210,225,0.55)',
            borderwidth=1, pad=dict(l=4, r=4, t=3, b=3),
            font=dict(color='#F0F4F7', size=12),
            buttons=[dict(
                label=f'{label}: ON / OFF', method='restyle',
                args=[{'visible': False}, [trace_index]],
                args2=[{'visible': True}, [trace_index]]
            )]
        ))
    fig.update_layout(
        updatemenus=toggle_menus,
        width=cfg.width,height=cfg.height,paper_bgcolor=cfg.background_color,
        font=dict(family='Latin Modern Roman, CMU Serif, Georgia, serif', color='#E9EDF1'),
        margin=dict(l=0,r=0,b=0,t=0),
        scene=dict(bgcolor=cfg.background_color,aspectmode='data',
                   xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)),
        legend=dict(x=.985,y=.99,xanchor='right',yanchor='top',
                    bgcolor='rgba(0,0,0,0.35)',
                    itemclick='toggle', itemdoubleclick='toggleothers'))
    out=Path(cfg.output_dir)/'galaxy_filament_network_3d.html'
    fig.write_html(str(out),include_plotlyjs=True,full_html=True)
    return (out,fig) if return_figure else out
