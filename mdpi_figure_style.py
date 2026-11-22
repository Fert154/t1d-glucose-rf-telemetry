import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_PAGELLA_DIRS = [
    '/usr/share/texmf/fonts/opentype/public/tex-gyre/',
    '/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/',
    '/usr/share/fonts/opentype/tex-gyre/',
]
for d in _PAGELLA_DIRS:
    for s in ['regular', 'bold', 'italic', 'bolditalic']:
        f = os.path.join(d, f'texgyrepagella-{s}.otf')
        if os.path.exists(f):
            fm.fontManager.addfont(f)
FONT = ['TeX Gyre Pagella', 'Palatino Linotype', 'Book Antiqua', 'Palatino', 'serif']

CM = 1 / 2.54
W, H = 12 * CM, 6.5 * CM
SQ = 8 * CM

BLUE   = '#2a78d6'
ORANGE = '#eb6834'
AQUA   = '#1baf7a'
INK    = '#1f1f1e'
MUTED  = '#6b6b66'
GRID   = '#d9d9d6'
BLUE_L, ORANGE_L, GRAY_L = '#e3eefc', '#fde3d8', '#efefec'

plt.rcParams.update({
    'font.family': FONT,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'TeX Gyre Pagella',
    'mathtext.it': 'TeX Gyre Pagella:italic',
    'mathtext.bf': 'TeX Gyre Pagella:bold',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'axes.linewidth': 0.6,
    'axes.edgecolor': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.0,
    'pdf.fonttype': 42,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

def legend_top(ax, ncol=2, **kw):
    return ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0,
                     handlelength=1.8, ncol=ncol, **kw)

def save(fig, name, outdir='.'):
    fig.savefig(os.path.join(outdir, f'{name}.pdf'))
    fig.savefig(os.path.join(outdir, f'{name}.png'), dpi=200)
    plt.close(fig)

