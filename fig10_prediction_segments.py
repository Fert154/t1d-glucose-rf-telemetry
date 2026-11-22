import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

SEED = 42
INPUT_LENGTH = 12
SAMPLING_MIN = 5
HYPO, HYPER = 70.0, 180.0
RF_PARAMS = dict(n_estimators=200, max_depth=15, min_samples_split=20, min_samples_leaf=20,
                 max_features=0.5, bootstrap=True, n_jobs=-1, random_state=SEED)

try:
    from mdpi_figure_style import W, H, BLUE, ORANGE, MUTED, legend_top, save
except ImportError:
    from matplotlib import font_manager as fm
    for d in ['/usr/share/texmf/fonts/opentype/public/tex-gyre/',
              '/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/',
              '/usr/share/fonts/opentype/tex-gyre/']:
        for st in ['regular', 'bold', 'italic', 'bolditalic']:
            f = os.path.join(d, f'texgyrepagella-{st}.otf')
            if os.path.exists(f):
                fm.fontManager.addfont(f)
    CM = 1 / 2.54
    W, H = 12 * CM, 6.5 * CM
    BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1f1f1e', '#6b6b66', '#d9d9d6'
    plt.rcParams.update({
        'font.family': ['TeX Gyre Pagella', 'Palatino Linotype', 'Book Antiqua', 'Palatino', 'serif'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'TeX Gyre Pagella',
        'mathtext.it': 'TeX Gyre Pagella:italic', 'mathtext.bf': 'TeX Gyre Pagella:bold',
        'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4, 'lines.linewidth': 1.0,
        'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })

    def legend_top(ax, ncol=2, **kw):
        return ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0,
                         handlelength=1.8, ncol=ncol, **kw)

    def save(fig, name, outdir='.'):
        fig.savefig(os.path.join(outdir, f'{name}.pdf'))
        fig.savefig(os.path.join(outdir, f'{name}.png'), dpi=200)
        plt.close(fig)

_DATOS_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "combined_data.xlsx")
_DATOS_POR_DEFECTO = _DATOS_REPO if os.path.exists(_DATOS_REPO) else "combined_data.xlsx"

def build_windows(df, steps):
    patient_col = df.columns[0]
    values = df.drop(columns=[patient_col]).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    patients = df[patient_col].to_numpy()
    X_l, y_l, g_l, r_l, s_l = [], [], [], [], []
    for row_id, (pid, series) in enumerate(zip(patients, values)):
        n = len(series) - INPUT_LENGTH - (steps - 1)
        if n <= 0:
            continue
        windows = sliding_window_view(series, INPUT_LENGTH)[:n]
        targets = series[INPUT_LENGTH + steps - 1:]
        ok = ~np.isnan(windows).any(axis=1) & ~np.isnan(targets)
        X_l.append(windows[ok])
        y_l.append(targets[ok])
        g_l.append(np.full(ok.sum(), pid, dtype=object))
        r_l.append(np.full(ok.sum(), row_id))
        s_l.append(np.flatnonzero(ok))
    return (np.vstack(X_l), np.concatenate(y_l), np.concatenate(g_l),
            np.concatenate(r_l), np.concatenate(s_l))

def delta_features(X):
    t = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2.0
    mean = X.mean(axis=1, keepdims=True)
    slope = ((X - mean) @ t / (t @ t))[:, None]
    return np.hstack([X, np.diff(X, axis=1), mean, X.std(axis=1, keepdims=True), slope])

def find_segments(y, row_id, start, seg_len, n_wanted):
    found = []
    for rid in np.unique(row_id):
        idx = np.flatnonzero(row_id == rid)
        idx = idx[np.argsort(start[idx])]
        best = None
        for a in range(0, len(idx) - seg_len + 1, 12):
            seg = idx[a:a + seg_len]
            if np.any(np.diff(start[seg]) != 1):
                continue
            n_hypo = int(np.sum(y[seg] < HYPO))
            n_hyper = int(np.sum(y[seg] > HYPER))
            score = min(n_hypo, n_hyper)
            if score >= 6 and (best is None or score > best[0]):
                best = (score, seg, n_hypo, n_hyper)
        if best:
            found.append(best)
    found.sort(key=lambda b: -b[0])
    return found[:n_wanted]

def draw(ax, t_h, real, pred, legend=True):
    ax.plot(t_h, real, color=BLUE, label="Real value")
    ax.plot(t_h, pred, color=ORANGE, label="Prediction")
    ax.axhline(HYPER, color=MUTED, ls="--", label="Hyperglycemia (180 mg/dL)")
    ax.axhline(HYPO, color=MUTED, ls=":", label="Hypoglycemia (70 mg/dL)")
    ax.set_xlim(t_h[0], t_h[-1])
    ax.set_ylim(min(40.0, float(min(real.min(), pred.min())) - 8),
                max(300.0, float(max(real.max(), pred.max())) + 15))
    ax.set(xlabel="Time [h]", ylabel="Glucose (mg/dL)")
    if legend:
        legend_top(ax, ncol=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=_DATOS_POR_DEFECTO)
    ap.add_argument("--sheet", default="Combined Data")
    ap.add_argument("--out", default="fig10_segments")
    ap.add_argument("--horizonte", type=int, default=45)
    ap.add_argument("--pliegue", type=int, default=1)
    ap.add_argument("--horas", type=float, default=24)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--trees", type=int, default=None, help="solo para pruebas rapidas")
    args = ap.parse_args()
    if args.trees:
        RF_PARAMS["n_estimators"] = args.trees
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_excel(args.file, sheet_name=args.sheet)
    steps = args.horizonte // SAMPLING_MIN
    X, y, groups, row_id, start = build_windows(df, steps)
    tr, te = list(GroupKFold(n_splits=5).split(X, y, groups))[args.pliegue - 1]

    print(f"Pliegue {args.pliegue}: {len(np.unique(groups[tr]))} pacientes de entrenamiento, "
          f"{len(np.unique(groups[te]))} de prueba. Entrenando Random Forest...")
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(delta_features(X[tr]), y[tr] - X[tr][:, -1])
    pred = X[te][:, -1] + model.predict(delta_features(X[te]))
    y_te, rid_te, st_te, g_te = y[te], row_id[te], start[te], groups[te]
    mae_fold = float(np.mean(np.abs(pred - y_te)))

    seg_len = int(args.horas * 60 / SAMPLING_MIN)
    segs = find_segments(y_te, rid_te, st_te, seg_len, args.n)
    if not segs:
        print("No se encontraron tramos con hipo e hiperglucemia; prueba con --horas mas grande u otro --pliegue.")
        return

    rows = []
    ncols = 2
    nrows = int(np.ceil(len(segs) / ncols))
    fig_all, axes = plt.subplots(nrows, ncols, figsize=(2 * W + 0.4, (H + 0.5) * nrows), squeeze=False)
    for k, (_, seg, n_hypo, n_hyper) in enumerate(segs, 1):
        t_min = (st_te[seg] + INPUT_LENGTH - 1 + steps) * SAMPLING_MIN
        t_h = (t_min - t_min[0]) / 60.0
        real, p = y_te[seg], pred[seg]
        mae_seg = float(np.mean(np.abs(p - real)))
        rows.append({"candidato": k, "paciente": g_te[seg][0], "sesion (fila del Excel)": int(rid_te[seg][0]) + 2,
                     "lecturas hipo": n_hypo, "lecturas hiper": n_hyper,
                     "MAE tramo": round(mae_seg, 2), "MAE pliegue": round(mae_fold, 2)})

        fig, ax = plt.subplots(figsize=(W, H))
        draw(ax, t_h, real, p)
        save(fig, f"fig10_candidato_{k}", outdir=args.out)

        ax = axes[(k - 1) // ncols][(k - 1) % ncols]
        draw(ax, t_h, real, p, legend=False)
        ax.set_title(f"Candidato {k}: paciente {g_te[seg][0]}, MAE {mae_seg:.1f} mg/dL")
    for j in range(len(segs), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig_all.savefig(os.path.join(args.out, "resumen_candidatos.png"), dpi=150)
    plt.close(fig_all)

    tabla = pd.DataFrame(rows)
    tabla.to_csv(os.path.join(args.out, "candidatos.csv"), index=False)
    print("\n" + tabla.to_string(index=False))
    print(f"\nMAE de todo el pliegue {args.pliegue} a {args.horizonte} min: {mae_fold:.2f} mg/dL")
    print(f"Abre '{args.out}/resumen_candidatos.png' para ver todas las opciones juntas.")

if __name__ == "__main__":
    main()
