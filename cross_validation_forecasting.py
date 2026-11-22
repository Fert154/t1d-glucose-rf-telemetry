import argparse
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import chi2, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterSampler

try:
    import xgboost
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SEED = 42
INPUT_LENGTH = 12
SAMPLING_MIN = 5
HORIZONS_MIN = [15, 30, 45, 60]
N_SPLITS = 5
HYPO, HYPER = 70.0, 180.0
CLASS_NAMES = ["hypo", "normo", "hyper"]
TAU_GRID = np.arange(70.0, 121.0, 1.0)
CALIB_FRACTION = 0.25

RF_PARAMS = dict(
    n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=2,
    max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=SEED,
)
RF_DELTA_PARAMS = dict(
    n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=5,
    max_features=0.5, bootstrap=True, n_jobs=-1, random_state=SEED,
)
XGB_PARAMS = dict(
    objective="reg:squarederror", n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED,
)

RF_SPACE = {
    "max_depth": [10, 15, 20, 30],
    "min_samples_leaf": [2, 5, 10, 20, 50],
    "min_samples_split": [2, 10, 20],
    "max_features": [0.33, 0.5, 0.75, 1.0],
}
XGB_SPACE = {
    "n_estimators": [200, 400, 800],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 20],
    "reg_lambda": [1.0, 5.0],
}
INNER_SPLITS = 3
SEARCH = dict(n_rf=12, n_xgb=20, stride=4, rf_trees=60)

_DATOS_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "combined_data.xlsx")
_DATOS_POR_DEFECTO = _DATOS_REPO if os.path.exists(_DATOS_REPO) else "combined_data.xlsx"

def get_models(tune=False):
    models = {
        "Persistence": None,
        "RF": dict(cls=RandomForestRegressor, params=RF_PARAMS, delta=False),
        "RF-delta": dict(cls=RandomForestRegressor, params=RF_DELTA_PARAMS, delta=True),
    }
    if HAS_XGB:
        models["XGB-delta"] = dict(cls=XGBRegressor, params=XGB_PARAMS, delta=True)
    if tune:
        models["RF-delta-tuned"] = dict(cls=RandomForestRegressor, params=RF_DELTA_PARAMS, delta=True,
                                        space=RF_SPACE, n_iter=SEARCH["n_rf"],
                                        search_override={"n_estimators": SEARCH["rf_trees"]})
        if HAS_XGB:
            models["XGB-delta-tuned"] = dict(cls=XGBRegressor, params=XGB_PARAMS, delta=True,
                                             space=XGB_SPACE, n_iter=SEARCH["n_xgb"], search_override={})
    return models

def build_windows(df, input_length, steps, corrected=True):
    shift = steps - 1 if corrected else steps
    patient_col = df.columns[0]
    values = df.drop(columns=[patient_col]).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    patients = df[patient_col].to_numpy()

    X_l, y_l, g_l, r_l, s_l = [], [], [], [], []
    for row_id, (pid, series) in enumerate(zip(patients, values)):
        n = len(series) - input_length - shift
        if n <= 0:
            continue
        windows = sliding_window_view(series, input_length)[:n]
        targets = series[input_length + shift:]
        ok = ~np.isnan(windows).any(axis=1) & ~np.isnan(targets)
        if not ok.any():
            continue
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

def fit_predict(spec, X_tr, y_tr, X_te, params=None):
    if spec is None:
        return X_te[:, -1].copy(), None, 0.0
    model = spec["cls"](**(params or spec["params"]))
    t0 = time.perf_counter()
    if spec["delta"]:
        model.fit(delta_features(X_tr), y_tr - X_tr[:, -1])
        predict = lambda Z: Z[:, -1] + model.predict(delta_features(Z))
    else:
        model.fit(X_tr, y_tr)
        predict = lambda Z: model.predict(Z)
    fit_s = time.perf_counter() - t0
    predict.model = model
    return predict(X_te), predict, fit_s

def to_class(g, tau=HYPO):
    g = np.asarray(g, dtype=float)
    return np.where(g < tau, 0, np.where(g > HYPER, 2, 1))

def tune_params(spec, X_tr, y_tr, g_tr):
    Xs, ys, gs = X_tr[::SEARCH["stride"]], y_tr[::SEARCH["stride"]], g_tr[::SEARCH["stride"]]
    inner = list(GroupKFold(n_splits=INNER_SPLITS).split(Xs, ys, gs))
    cands = [{}] + list(ParameterSampler(spec["space"], n_iter=max(spec["n_iter"] - 1, 0), random_state=SEED))
    scores = []
    for cand in cands:
        p = {**spec["params"], **cand, **spec["search_override"]}
        maes = [mean_absolute_error(ys[b], fit_predict(spec, Xs[a], ys[a], Xs[b], p)[0]) for a, b in inner]
        scores.append(float(np.mean(maes)))
    best = int(np.argmin(scores))
    return {**spec["params"], **cands[best]}, cands[best], scores[best], scores[0]

def calibrate_tau(spec, X_tr, y_tr, g_tr, params=None):
    gss = GroupShuffleSplit(n_splits=1, test_size=CALIB_FRACTION, random_state=SEED)
    fit_idx, cal_idx = next(gss.split(X_tr, y_tr, g_tr))
    pred, _, _ = fit_predict(spec, X_tr[fit_idx], y_tr[fit_idx], X_tr[cal_idx], params)
    is_hypo = y_tr[cal_idx] < HYPO
    best_tau, best_f1 = HYPO, -1.0
    for tau in TAU_GRID:
        flagged = pred < tau
        denom = flagged.sum() + is_hypo.sum()
        f1 = 2.0 * np.sum(flagged & is_hypo) / denom if denom else 0.0
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), f1
    return best_tau

def fold_metrics(y_true, y_pred, tau=HYPO):
    yt, yp = to_class(y_true), to_class(y_pred, tau)
    prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2], zero_division=0)
    out = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mard_pct": float(np.mean(np.abs(y_pred - y_true) / np.clip(y_true, 1e-6, None)) * 100),
        "accuracy": float(np.mean(yt == yp)),
        "precision_macro": float(prec.mean()),
        "recall_macro": float(rec.mean()),
        "f1_macro": float(f1.mean()),
    }
    for k, name in enumerate(CLASS_NAMES):
        out[f"precision_{name}"] = float(prec[k])
        out[f"recall_{name}"] = float(rec[k])
        out[f"f1_{name}"] = float(f1[k])
    is_hypo, pred_hypo = yt == 0, yp == 0
    tn, fp = np.sum(~is_hypo & ~pred_hypo), np.sum(~is_hypo & pred_hypo)
    out["specificity_hypo"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    out["aucpr_hypo"] = float(average_precision_score(is_hypo, -y_pred)) if is_hypo.any() else np.nan
    out["prevalence_hypo"] = float(is_hypo.mean())
    out["tau"] = float(tau)
    return out

def mcnemar(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return b, c, 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(stat), float(chi2.sf(stat, 1))

def latency_ms(predict, X_test, n_calls=200):
    model = predict.model
    old = model.get_params().get("n_jobs")
    model.set_params(n_jobs=1)
    idx = np.random.default_rng(SEED).integers(0, len(X_test), n_calls)
    t0 = time.perf_counter()
    for i in idx:
        predict(X_test[i:i + 1])
    dt = (time.perf_counter() - t0) / n_calls * 1000.0
    model.set_params(n_jobs=old)
    return dt

def plot_confusion(cm, title, path):
    rown = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n({rown[i, j] * 100:.1f}%)", ha="center", va="center",
                    fontsize=9, color="white" if rown[i, j] > 0.5 else "black")
    labels = ["Hypo", "Normo", "Hyper"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="Row-normalized rate")
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)

def plot_trace(y_true, preds, row_id, start, horizon, path, n_points=288):
    best = None
    for rid in np.unique(row_id):
        sel = np.flatnonzero(row_id == rid)
        breaks = np.flatnonzero(np.diff(start[sel]) != 1)
        end = breaks[0] + 1 if len(breaks) else len(sel)
        if best is None or end > best[1]:
            best = (sel[:end], end)
        if end >= n_points:
            break
    sel = best[0][:n_points]
    t = np.arange(len(sel)) * SAMPLING_MIN / 60.0
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t, y_true[sel], label="Real value", lw=1.6, color="black")
    for name, p in preds.items():
        ax.plot(t, p[sel], label=name, lw=1.1, ls="--")
    ax.axhline(HYPER, color="tab:red", ls=":", lw=1)
    ax.axhline(HYPO, color="tab:blue", ls=":", lw=1)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"{horizon}-min prediction horizon, one held-out session", fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=_DATOS_POR_DEFECTO)
    ap.add_argument("--sheet", default="Combined Data")
    ap.add_argument("--out", default="cv_results")
    ap.add_argument("--afinar", action="store_true", default=True, help="busqueda de hiperparametros con validacion anidada (activada por defecto)")
    ap.add_argument("--n-rf", type=int, default=12, help="candidatos para Random Forest")
    ap.add_argument("--n-xgb", type=int, default=20, help="candidatos para XGBoost")
    ap.add_argument("--submuestra", type=int, default=4, help="la busqueda usa 1 de cada N ventanas")
    ap.add_argument("--arboles-busqueda", type=int, default=60, help="arboles de RF durante la busqueda")
    ap.add_argument("--desfase-original", action="store_true")
    ap.add_argument("--solo", nargs="+", default=["RF-delta", "RF-delta-tuned", "XGB-delta", "XGB-delta-tuned"], help="modelos a correr (Persistence siempre se incluye)")
    ap.add_argument("--trees", type=int, default=None, help="solo para pruebas rapidas")
    args = ap.parse_args()

    if args.trees:
        RF_PARAMS["n_estimators"] = RF_DELTA_PARAMS["n_estimators"] = XGB_PARAMS["n_estimators"] = args.trees
    SEARCH.update(n_rf=args.n_rf, n_xgb=args.n_xgb, stride=max(args.submuestra, 1),
                  rf_trees=min(args.arboles_busqueda, RF_DELTA_PARAMS["n_estimators"]))
    corrected = not args.desfase_original
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_excel(args.file, sheet_name=args.sheet)
    patient_col = df.columns[0]
    n_readings = int(df.drop(columns=[patient_col]).apply(pd.to_numeric, errors="coerce").notna().sum().sum())
    n_patients = df[patient_col].nunique()
    per_patient = df[patient_col].value_counts()

    print("=" * 78)
    print(f"Archivo            : {args.file}  (hoja: {args.sheet})")
    print(f"Sesiones (filas)   : {df.shape[0]}   |  lecturas por sesion: {df.shape[1] - 1}")
    print(f"Pacientes unicos   : {n_patients}  (sesiones por paciente: min {per_patient.min()}, "
          f"mediana {per_patient.median():.0f}, max {per_patient.max()})")
    print(f"Lecturas validas   : {n_readings:,}  (~{n_readings * SAMPLING_MIN / 1440:.0f} pacientes-dia)")
    print(f"Desfase corregido  : {corrected}")
    if not HAS_XGB:
        print("AVISO: XGBoost no esta instalado; se omite XGB-delta (python -m pip install xgboost).")

    models = get_models(tune=args.afinar)
    if args.solo:
        models = {k: v for k, v in models.items() if k == "Persistence" or k in args.solo}
    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_rows, size_rows, test_rows, cm_rows, time_rows, hp_rows = [], [], [], [], [], []

    for h_min in HORIZONS_MIN:
        steps = h_min // SAMPLING_MIN
        X, y, groups, row_id, start = build_windows(df, INPUT_LENGTH, steps, corrected)
        yt_c = to_class(y)
        print("\n" + "=" * 78)
        print(f"HORIZONTE {h_min} min ({steps} pasos)  |  ventanas: {len(y):,}")

        oof = {name: np.full(len(y), np.nan) for name in models}
        oof_cls = {name: np.zeros(len(y), dtype=int) for name in models}

        for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
            cls_te = yt_c[te]
            size_rows.append({
                "horizon_min": h_min, "fold": fold,
                "patients_train": len(np.unique(groups[tr])), "patients_test": len(np.unique(groups[te])),
                "windows_train": len(tr), "windows_test": len(te),
                "test_hypo": int((cls_te == 0).sum()), "test_normo": int((cls_te == 1).sum()),
                "test_hyper": int((cls_te == 2).sum()),
            })
            for name, spec in models.items():
                params, tune_s = None, 0.0
                if spec is not None and spec.get("space"):
                    t0 = time.perf_counter()
                    params, chosen, mae_best, mae_def = tune_params(spec, X[tr], y[tr], groups[tr])
                    tune_s = time.perf_counter() - t0
                    hp_rows.append({"model": name, "horizon_min": h_min, "fold": fold,
                                    "inner_mae_chosen": mae_best, "inner_mae_default": mae_def,
                                    "is_default": not chosen, "search_s": tune_s,
                                    **{k: params[k] for k in spec["space"]}})
                    print(f"  pliegue {fold} | {name:<15} busqueda {tune_s:6.0f} s | MAE interno "
                          f"{mae_def:.2f} -> {mae_best:.2f} | {chosen if chosen else 'se queda el defecto'}")
                tau = calibrate_tau(spec, X[tr], y[tr], groups[tr], params)
                pred, predict, fit_s = fit_predict(spec, X[tr], y[tr], X[te], params)
                oof[name][te] = pred
                oof_cls[name][te] = to_class(pred, tau)

                for decision, t_use in (("default", HYPO), ("calibrated", tau)):
                    m = fold_metrics(y[te], pred, t_use)
                    m.update({"model": name, "decision": decision, "horizon_min": h_min, "fold": fold})
                    fold_rows.append(m)
                    if decision == "default":
                        m0 = m
                    else:
                        m1 = m
                time_rows.append({
                    "model": name, "horizon_min": h_min, "fold": fold, "fit_s": fit_s, "search_s": tune_s,
                    "single_sample_latency_ms": latency_ms(predict, X[te]) if predict else 0.0,
                })
                print(f"  pliegue {fold} | {name:<15} RMSE {m0['rmse']:6.2f}  MAE {m0['mae']:6.2f} | "
                      f"sens. hipo {m0['recall_hypo']:.3f} -> {m1['recall_hypo']:.3f} "
                      f"(tau {tau:.0f}), F1 hipo {m0['f1_hypo']:.3f} -> {m1['f1_hypo']:.3f}")

        for name in models:
            for decision, cls in (("default", to_class(oof[name])), ("calibrated", oof_cls[name])):
                cm = confusion_matrix(yt_c, cls, labels=[0, 1, 2])
                for i, tname in enumerate(CLASS_NAMES):
                    cm_rows.append({"model": name, "decision": decision, "horizon_min": h_min,
                                    "true_class": tname,
                                    **{f"pred_{p}": int(cm[i, j]) for j, p in enumerate(CLASS_NAMES)}})
                if name != "Persistence":
                    plot_confusion(cm, f"{name}, {h_min} min, {decision} threshold (5-fold by patient)",
                                   os.path.join(args.out, f"fig_confusion_{name}_{h_min}min_{decision}.png"))
        plot_trace(y, {n: p for n, p in oof.items() if n != "Persistence"}, row_id, start, h_min,
                   os.path.join(args.out, f"fig_trace_{h_min}min.png"))

        pat = groups
        mae_base = pd.Series(np.abs(oof["Persistence"] - y)).groupby(pat).mean()
        for name in models:
            if name == "Persistence":
                continue
            mae_m = pd.Series(np.abs(oof[name] - y)).groupby(pat).mean().loc[mae_base.index]
            w_stat, w_p = wilcoxon(mae_m, mae_base) if not np.allclose(mae_m, mae_base) else (np.nan, 1.0)
            row = {"horizon_min": h_min, "comparison": f"{name} vs Persistence", "n_patients": len(mae_m),
                   "mae_patient_mean_model": float(mae_m.mean()),
                   "mae_patient_mean_persistence": float(mae_base.mean()),
                   "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p)}
            for decision, cls in (("default", to_class(oof[name])), ("calibrated", oof_cls[name])):
                ref = to_class(oof["Persistence"]) if decision == "default" else oof_cls["Persistence"]
                b, c, s, p = mcnemar(cls == yt_c, ref == yt_c)
                row.update({f"mcnemar_{decision}_model_only_correct": b,
                            f"mcnemar_{decision}_persistence_only_correct": c,
                            f"mcnemar_{decision}_p": p})
            test_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    times = pd.DataFrame(time_rows)
    keys = ["model", "decision", "horizon_min"]
    metric_cols = [c for c in folds.columns if c not in keys + ["fold"]]
    agg = folds.groupby(keys, sort=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()

    def pm(df_, col, dec=2):
        return (df_[f"{col}_mean"].map(lambda v: f"{v:.{dec}f}") + " ± "
                + df_[f"{col}_std"].map(lambda v: f"{v:.{dec}f}"))

    base = agg[agg["decision"] == "default"]
    tabA = base[["model", "horizon_min"]].copy()
    for c in ("rmse", "mae", "mard_pct"):
        tabA[c] = pm(base, c, 2)

    rowsB = []
    for _, r in agg.iterrows():
        for name in CLASS_NAMES + ["macro"]:
            rowsB.append({"model": r["model"], "decision": r["decision"], "horizon_min": int(r["horizon_min"]),
                          "class": name,
                          "precision": f"{r[f'precision_{name}_mean']:.3f} ± {r[f'precision_{name}_std']:.3f}",
                          "recall": f"{r[f'recall_{name}_mean']:.3f} ± {r[f'recall_{name}_std']:.3f}",
                          "f1": f"{r[f'f1_{name}_mean']:.3f} ± {r[f'f1_{name}_std']:.3f}"})
    tabB = pd.DataFrame(rowsB)

    tabC = agg[keys].copy()
    tabC["tau"] = agg["tau_mean"].map(lambda v: f"{v:.1f}")
    for c in ("recall_hypo", "specificity_hypo", "precision_hypo", "f1_hypo", "f1_macro"):
        tabC[c] = pm(agg, c, 3)
    tabC["aucpr_hypo"] = pm(agg, "aucpr_hypo", 3)
    tabC = tabC.rename(columns={"recall_hypo": "sensitivity_hypo"})

    out = lambda name: os.path.join(args.out, name)
    folds.to_csv(out("metricas_por_pliegue.csv"), index=False)
    agg.to_csv(out("resumen_media_de.csv"), index=False)
    tabA.to_csv(out("tabla_A_regresion.csv"), index=False)
    tabB.to_csv(out("tabla_B_clases.csv"), index=False)
    tabC.to_csv(out("tabla_C_hipoglucemia.csv"), index=False)
    pd.DataFrame(size_rows).to_csv(out("pliegues_tamanos.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(out("pruebas_pareadas.csv"), index=False)
    pd.DataFrame(cm_rows).to_csv(out("matrices_confusion.csv"), index=False)
    times.to_csv(out("tiempos.csv"), index=False)
    if hp_rows:
        pd.DataFrame(hp_rows).to_csv(out("hiperparametros_elegidos.csv"), index=False)

    tsum = times.groupby("model", sort=False)[["fit_s", "single_sample_latency_ms"]].mean()
    env = [
        f"Python            : {sys.version.split()[0]}",
        f"NumPy             : {np.__version__}",
        f"pandas            : {pd.__version__}",
        f"scikit-learn      : {sklearn.__version__}",
        f"SciPy             : {scipy.__version__}",
        f"XGBoost           : {xgboost.__version__ if HAS_XGB else 'no instalado'}",
        f"Sistema           : {platform.platform()}",
        f"Procesador        : {platform.processor() or platform.machine()}",
        f"Nucleos logicos   : {os.cpu_count()}",
        f"Semilla           : {SEED}",
        f"Ventana           : {INPUT_LENGTH} lecturas, paso de 1 muestra, sin normalizacion",
        f"Desfase corregido : {corrected}",
        f"Validacion        : GroupKFold por paciente, k = {N_SPLITS}",
        f"Calibracion tau   : {int(CALIB_FRACTION * 100)} % de los pacientes de entrenamiento, "
        f"rejilla {TAU_GRID[0]:.0f}-{TAU_GRID[-1]:.0f} mg/dL, criterio F1 de hipoglucemia",
        f"RF                : {RF_PARAMS}",
        f"RF-delta          : {RF_DELTA_PARAMS}",
        f"XGB-delta         : {XGB_PARAMS if HAS_XGB else 'no instalado'}",
        f"Busqueda (--afinar): {'si' if args.afinar else 'no'}"
        + (f" | GroupKFold interno k = {INNER_SPLITS}, criterio MAE, RF {SEARCH['n_rf']} candidatos "
           f"({SEARCH['rf_trees']} arboles en la busqueda), XGB {SEARCH['n_xgb']} candidatos, "
           f"1 de cada {SEARCH['stride']} ventanas" if args.afinar else ""),
        f"Espacio RF        : {RF_SPACE}",
        f"Espacio XGB       : {XGB_SPACE}",
        f"Entradas delta    : 12 lecturas + 11 diferencias + media + DE + pendiente (26); objetivo G(t+H) - G(t)",
        f"Sesiones          : {df.shape[0]}  |  Pacientes: {n_patients}  |  Lecturas validas: {n_readings}",
    ] + [f"{m:<15} entrenamiento {r.fit_s:6.1f} s por pliegue | latencia lote 1: {r.single_sample_latency_ms:.2f} ms"
         for m, r in tsum.iterrows()]
    with open(out("entorno.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(env) + "\n")

    pd.set_option("display.width", 220)
    print("\n" + "=" * 78 + "\nTABLA A. Regresion (media ± DE entre pliegues)\n")
    print(tabA.to_string(index=False))
    print("\nTABLA C. Hipoglucemia y F1 macro, con umbral 70 (default) y umbral calibrado\n")
    print(tabC.to_string(index=False))
    if hp_rows:
        hp = pd.DataFrame(hp_rows)
        print("\nHIPERPARAMETROS ELEGIDOS POR PLIEGUE\n")
        print(hp.drop(columns=["search_s"]).to_string(index=False))
    print("\nENTORNO\n" + "\n".join(env))
    print(f"\nTodo quedo guardado en la carpeta '{args.out}'.")

if __name__ == "__main__":
    main()
