import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import functools, operator, warnings
    warnings.filterwarnings("ignore")

    import hvplot.polars  # noqa: F401
    import holoviews as hv
    import hvplot
    hvplot.extension("bokeh")

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    import xgboost as xgb
    from statsmodels.othermod.betareg import BetaModel
    import statsmodels.api as sm

    from statsmodels.tools.sm_exceptions import HessianInversionWarning, ConvergenceWarning

    return (
        BetaModel,
        ConvergenceWarning,
        GaussianProcessRegressor,
        HessianInversionWarning,
        Lasso,
        Matern,
        Pipeline,
        Ridge,
        StandardScaler,
        WhiteKernel,
        functools,
        hv,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        operator,
        pl,
        r2_score,
        sm,
        warnings,
        xgb,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # CTomics — ML Modelling Notebook
    ## Models benchmarked against Cho et al. baseline

    | ID | Model | Key idea |
    |----|-------|----------|
    | **M0** | Cho Model E | Sigmoidal baseline — ΔCt only |
    | **M1** | Beta Regression | Correct distributional model for proportions |
    | **M2** | Logit-LASSO | Sparse linear on logit scale, polynomial features |
    | **M3** | Logit-Ridge | Regularised linear, same feature set |
    | **M4** | XGBoost | Gradient boosted trees on logit scale |
    | **M5** | Gaussian Process | Matérn kernel, calibrated uncertainty |

    **Features**: `ct_ACTB`, `ct_16S`, `sample_type` (one-hot), `delta`
    **CV**: LOOCV primary · LOPOCV sensitivity · metrics split by regime
    """)
    return


@app.cell
def _(pl):
    _raw = pl.read_csv(
        "../data/data.csv",
        null_values=["NA", "", "None"],
        infer_schema_length=200,
    )
    SAMPLE_TYPES = ["lung_bal", "oropharyngeal", "stool", "rectal_swab", "vaginal_sample", "lung_sputum"]
    TYPE_IDX = {t: i for i, t in enumerate(SAMPLE_TYPES)}

    PALETTE = {
        "stool":          "#2166ac",
        "oropharyngeal":  "#e08214",
        "rectal_swab":    "#1b7837",
        "vaginal_sample": "#d6282a",
        "lung_bal":       "#7c3aed",
        "lung_sputum":    "#f97316",
    }

    df = _raw.with_columns([
        pl.col("ct_16S").cast(pl.Float64),
        pl.col("ct_ACTB").cast(pl.Float64),
        pl.col("delta").cast(pl.Float64),
        pl.col("pct_microbial").cast(pl.Float64),
        pl.col("qbit_1").cast(pl.Float64),
        pl.col("sample_type")
          .map_elements(lambda s: TYPE_IDX.get(s, -1), return_dtype=pl.Int32)
          .alias("type_idx"),
        pl.when(pl.col("pct_microbial") < 2).then(pl.lit("< 2% (BAL)"))
          .when(pl.col("pct_microbial") < 4).then(pl.lit("2-4% (Cho fail)"))
          .when(pl.col("pct_microbial") < 98).then(pl.lit("4-98% (valid)"))
          .otherwise(pl.lit("> 98% (sat.)"))
          .alias("regime"),
    ])

    df_fit = (
        df.filter(
            pl.col("pct_microbial").is_not_null() &
            pl.col("ct_16S").is_not_null() &
            pl.col("ct_ACTB").is_not_null()
        )
        .with_row_index("row_id")
    )

    print(f"Fitting dataset: {len(df_fit)} samples, {df_fit['sample_type'].n_unique()} types")
    df_fit.head(4)
    return PALETTE, SAMPLE_TYPES, TYPE_IDX, df_fit


@app.cell
def _(SAMPLE_TYPES, TYPE_IDX, df_fit, np):
    from sklearn.preprocessing import PolynomialFeatures as PF

    n         = len(df_fit)
    ct_actb   = df_fit["ct_ACTB"].to_numpy()
    ct_16s    = df_fit["ct_16S"].to_numpy()
    delta     = df_fit["delta"].to_numpy()
    y_pct     = df_fit["pct_microbial"].to_numpy()
    types     = df_fit["sample_type"].to_list()
    qbit      = df_fit["qbit_1"].fill_null(0.0).to_numpy()

    # one-hot encode sample type
    type_ohe = np.zeros((n, len(SAMPLE_TYPES)))
    for i, t in enumerate(types):
        idx = TYPE_IDX.get(t, -1)
        if idx >= 0:
            type_ohe[i, idx] = 1.0

    # base feature matrix: ct_ACTB, ct_16S, delta + type OHE
    base_cont = np.column_stack([ct_actb, ct_16s, delta])
    X_base    = np.column_stack([base_cont, type_ohe])
    X_beta    = np.column_stack([base_cont, type_ohe[:, :-1]])  

    # polynomial degree-2 of continuous features + type OHE
    _pf       = PF(degree=2, include_bias=False)
    X_poly    = np.column_stack([_pf.fit_transform(base_cont), type_ohe])

    # target transforms
    def logit(p):
        return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    def inv_logit(x):
        return 100.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def to_prop(pct):
        """% in [0,100] -> proportion in (0,1) with Smithson & Verkuilen shrinkage."""
        p  = np.clip(pct / 100.0, 1e-6, 1 - 1e-6)
        ns = len(p)
        return (p * (ns - 1) + 0.5) / ns

    y_logit = logit(y_pct / 100.0)
    y_prop  = to_prop(y_pct)

    print(f"X_base : {X_base.shape}  |  X_poly : {X_poly.shape}")
    print(f"y range: {y_pct.min():.3f} - {y_pct.max():.2f} %")
    return X_base, X_beta, X_poly, inv_logit, n, y_logit, y_pct, y_prop


@app.cell
def _(np):
    def cho_model(delta):
        return np.clip(
            2.7201549 / (99.50267 * np.exp(-0.7218 * delta) + 0.02733),
            0.0, 100.0,
        )

    return (cho_model,)


@app.cell
def _(
    BetaModel,
    GaussianProcessRegressor,
    Lasso,
    Matern,
    Pipeline,
    Ridge,
    StandardScaler,
    WhiteKernel,
    X_base,
    X_beta,
    X_poly,
    cho_model,
    df_fit,
    inv_logit,
    n,
    np,
    sm,
    xgb,
    y_logit,
    y_prop,
):
    cho_pred_loo = cho_model(df_fit["delta"].to_numpy())

    oof = {
        "M0_Cho":    cho_pred_loo.copy(),
        "M1_Beta":   np.full(n, np.nan),
        "M2_LASSO":  np.full(n, np.nan),
        "M3_Ridge":  np.full(n, np.nan),
        "M4_XGB":    np.full(n, np.nan),
        "M5_GPR":    np.full(n, np.nan),
        "M5_GPR_std":np.full(n, np.nan),
    }

    for _i in range(n):
        _tr = np.ones(n, dtype=bool)
        _tr[_i] = False

        # M1 Beta regression
        try:
            _Xtr = sm.add_constant(X_beta[_tr],   has_constant="add")
            _Xte = sm.add_constant(X_beta[[_i]], has_constant="add")
            _fit = BetaModel(y_prop[_tr], _Xtr).fit(
                disp=False, maxiter=400, method="bfgs",
            )
            oof["M1_Beta"][_i] = float(np.clip(_fit.predict(_Xte)[0] * 100, 0, 100))
        except Exception:
            oof["M1_Beta"][_i] = np.nan

        # M2 Logit-LASSO
        _pl = Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.05, max_iter=5000))])
        _pl.fit(X_poly[_tr], y_logit[_tr])
        oof["M2_LASSO"][_i] = float(inv_logit(_pl.predict(X_poly[[_i]])[0]))

        # M3 Logit-Ridge
        _pr = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
        _pr.fit(X_poly[_tr], y_logit[_tr])
        oof["M3_Ridge"][_i] = float(inv_logit(_pr.predict(X_poly[[_i]])[0]))

        # M4 XGBoost
        _xm = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=42, verbosity=0,
        )
        _xm.fit(X_base[_tr], y_logit[_tr], verbose=False)
        oof["M4_XGB"][_i] = float(inv_logit(_xm.predict(X_base[[_i]])[0]))

        # M5 GPR
        _kernel = 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
        _gpr = GaussianProcessRegressor(
            kernel=_kernel, normalize_y=True,
            n_restarts_optimizer=2, random_state=42,
        )
        _sc_gpr = StandardScaler()
        _Xtr_sc = _sc_gpr.fit_transform(X_base[_tr])
        _Xte_sc = _sc_gpr.transform(X_base[[_i]])
        _gpr.fit(_Xtr_sc, y_logit[_tr])
        _mu, _sig = _gpr.predict(_Xte_sc, return_std=True)
        oof["M5_GPR"][_i]     = float(inv_logit(_mu[0]))
        oof["M5_GPR_std"][_i] = float(_sig[0])

    print("LOOCV done.")
    return (oof,)


@app.cell
def _(
    df_fit,
    mean_absolute_error,
    mean_squared_error,
    mo,
    np,
    oof,
    pl,
    r2_score,
    y_pct,
):
    MODEL_NAMES  = ["M0_Cho", "M1_Beta", "M2_LASSO", "M3_Ridge", "M4_XGB", "M5_GPR"]
    REGIME_ORDER = ["< 2% (BAL)", "2-4% (Cho fail)", "4-98% (valid)", "> 98% (sat.)"]
    regimes      = df_fit["regime"].to_list()

    def _met(yt, yp, label, regime="all"):
        mask = ~np.isnan(yp)
        yt, yp = yt[mask], yp[mask]
        return {
            "model":  label,
            "regime": regime,
            "n":      int(mask.sum()),
            "RMSE":   round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
            "MAE":    round(float(mean_absolute_error(yt, yp)), 3),
            "MedAE":  round(float(np.median(np.abs(yt - yp))), 3),
            "R2":     round(float(r2_score(yt, yp)), 4),
        }

    rows = []
    for _m in MODEL_NAMES:
        rows.append(_met(y_pct, oof[_m], _m))
    df_global = pl.DataFrame(rows)

    rows_r = []
    for _reg in REGIME_ORDER:
        _idx = np.where(np.array(regimes) == _reg)[0]
        if len(_idx) < 3:
            continue
        for _m in MODEL_NAMES:
            rows_r.append(_met(y_pct[_idx], oof[_m][_idx], _m, _reg))
    df_regime = pl.DataFrame(rows_r)

    mo.vstack([
        mo.md("## Global LOOCV metrics"),
        mo.ui.table(df_global),
        mo.md("## Per-regime LOOCV metrics"),
        mo.ui.table(df_regime),
    ])
    return MODEL_NAMES, REGIME_ORDER, regimes


@app.cell
def _(MODEL_NAMES, mo, np, oof, pl, regimes, y_pct):
    # ── Bootstrap CI on LOOCV RMSE ───────────────────────────────────────────────
    # Resample (y_true, y_pred) LOOCV pairs with replacement N_BOOT times.
    # Each iteration recomputes RMSE → sampling distribution → 95% CI + p-value vs Cho.
    # p_vs_Cho = fraction of bootstrap samples where model does NOT beat Cho (lower = better).

    _N_BOOT = 1000
    _rng    = np.random.default_rng(42)
    _n      = len(y_pct)
    _reg_arr = np.array(regimes)

    _boot = {_m: {"all": [], "bal": [], "cho_fail": []} for _m in MODEL_NAMES}

    for _ in range(_N_BOOT):
        _idx  = _rng.integers(0, _n, size=_n)
        _yt   = y_pct[_idx]
        _rb   = _reg_arr[_idx]
        for _m in MODEL_NAMES:
            _yp   = oof[_m][_idx]
            _mask = ~np.isnan(_yp)
            if _mask.sum() > 1:
                _boot[_m]["all"].append(float(np.sqrt(np.mean((_yt[_mask] - _yp[_mask]) ** 2))))
            _bal  = _mask & (_rb == "< 2% (BAL)")
            if _bal.sum() > 1:
                _boot[_m]["bal"].append(float(np.sqrt(np.mean((_yt[_bal] - _yp[_bal]) ** 2))))
            _cf   = _mask & (_rb == "2-4% (Cho fail)")
            if _cf.sum() > 1:
                _boot[_m]["cho_fail"].append(float(np.sqrt(np.mean((_yt[_cf] - _yp[_cf]) ** 2))))

    def _bstats(vals_m, vals_cho, label, regime):
        _bm  = np.array(vals_m)
        _bc  = np.array(vals_cho)
        _k   = min(len(_bm), len(_bc))
        _d   = _bc[:_k] - _bm[:_k]   # positive = model beats Cho
        return {
            "model":        label,
            "regime":       regime,
            "RMSE":         round(float(_bm.mean()), 3),
            "CI_lo":        round(float(np.percentile(_bm, 2.5)), 3),
            "CI_hi":        round(float(np.percentile(_bm, 97.5)), 3),
            "ΔRMSE_vs_Cho": round(float(_d.mean()), 3),
            "p_vs_Cho":     round(float((_d <= 0).mean()), 4),
        }

    _cho_all = _boot["M0_Cho"]["all"]
    _cho_bal = _boot["M0_Cho"]["bal"]
    _cho_cf  = _boot["M0_Cho"]["cho_fail"]

    _df_b_all = pl.DataFrame([_bstats(_boot[_m]["all"],      _cho_all, _m, "all")          for _m in MODEL_NAMES])
    _df_b_bal = pl.DataFrame([_bstats(_boot[_m]["bal"],      _cho_bal, _m, "< 2% (BAL)")   for _m in MODEL_NAMES])
    _df_b_cf  = pl.DataFrame([_bstats(_boot[_m]["cho_fail"], _cho_cf,  _m, "2-4% (Cho fail)") for _m in MODEL_NAMES])

    mo.vstack([
        mo.md("## Bootstrap RMSE — 1000 iterations (seed=42)"),
        mo.md("### Global (all 154 samples)"),
        mo.ui.table(_df_b_all),
        mo.md("### Near-zero zone — BAL regime (< 2%)"),
        mo.ui.table(_df_b_bal),
        mo.md("### Cho failure zone (2–4%)"),
        mo.ui.table(_df_b_cf),
        mo.md(
            "> **RMSE** = bootstrap mean. **CI_lo / CI_hi** = 2.5th–97.5th percentile (95% CI). "
            "> **ΔRMSE_vs_Cho** = mean(RMSE_Cho − RMSE_model): positive means model beats Cho. "
            "> **p_vs_Cho** = fraction of bootstrap iterations where model does NOT beat Cho "
            "> (treat p < 0.05 as significant improvement)."
        ),
    ])
    return


@app.cell
def _(MODEL_NAMES, df_fit, functools, np, oof, operator, pl, y_pct):
    _PAL_REG = {
        "< 2% (BAL)":      "#7c3aed",
        "2-4% (Cho fail)": "#ef4444",
        "4-98% (valid)":   "#22c55e",
        "> 98% (sat.)":    "#f59e0b",
    }

    def _by_regime(data, x, y, pal, **kw):
        parts = [
            data.filter(pl.col("regime") == r).hvplot.scatter(
                x=x, y=y, color=pal[r], label=r, **kw
            )
            for r in pal if data.filter(pl.col("regime") == r).shape[0] > 0
        ]
        return functools.reduce(operator.mul, parts)

    _ideal = pl.DataFrame({"x": [0.0, 100.0], "y": [0.0, 100.0]})

    _panels = []
    for _m in MODEL_NAMES:
        _rmse = float(np.sqrt(np.nanmean((y_pct - oof[_m]) ** 2)))
        _df_p = df_fit.with_columns(pl.Series("pred", oof[_m].tolist()))
        _sc   = _by_regime(
            _df_p, "pct_microbial", "pred", _PAL_REG,
            title=f"{_m}  RMSE={_rmse:.2f}%",
            xlabel="Observed %", ylabel="Predicted %",
            alpha=0.75, size=55, width=360, height=300,
        )
        _id = _ideal.hvplot.line(x="x", y="y", color="black",
                                  line_dash="dashed", line_width=1.5)
        _panels.append(_sc * _id)

    _r1 = functools.reduce(operator.add, _panels[:3])
    _r2 = functools.reduce(operator.add, _panels[3:])
    (_r1 + _r2).cols(3)
    return


@app.cell
def _(MODEL_NAMES, PALETTE, df_fit, functools, hv, oof, operator, pl, y_pct):
    def _by_type(data, x, y, pal, **kw):
        parts = [
            data.filter(pl.col("sample_type") == st).hvplot.scatter(
                x=x, y=y, color=pal.get(st, "grey"), label=st, **kw
            )
            for st in pal if data.filter(pl.col("sample_type") == st).shape[0] > 0
        ]
        return functools.reduce(operator.mul, parts)

    _panels_r = []
    for _m in MODEL_NAMES:
        _df_r = df_fit.with_columns([
            pl.Series("pred",     oof[_m].tolist()),
            pl.Series("residual", (y_pct - oof[_m]).tolist()),
        ])
        _sc = _by_type(
            _df_r, "pred", "residual", PALETTE,
            title=f"{_m} residuals",
            xlabel="Predicted %", ylabel="Obs - Pred (%)",
            alpha=0.7, size=50, width=360, height=280,
        )
        _zero = hv.HLine(0).opts(color="red", line_dash="dashed",
                                   line_width=1.2, alpha=0.6)
        _panels_r.append(_sc * _zero)

    _r1 = functools.reduce(operator.add, _panels_r[:3])
    _r2 = functools.reduce(operator.add, _panels_r[3:])
    (_r1 + _r2).cols(3)
    return


@app.cell
def _(MODEL_NAMES, REGIME_ORDER, np, oof, pl, regimes, y_pct):
    _rows_bar = []
    for _reg in REGIME_ORDER:
        _idx = np.where(np.array(regimes) == _reg)[0]
        if len(_idx) < 2:
            continue
        for _m in MODEL_NAMES:
            _yp = oof[_m][_idx]
            _mask = ~np.isnan(_yp)
            if _mask.sum() < 2:
                continue
            _rmse = float(np.sqrt(np.mean((y_pct[_idx][_mask] - _yp[_mask]) ** 2)))
            _rows_bar.append({"regime": _reg, "model": _m, "RMSE": round(_rmse, 4)})

    pl.DataFrame(_rows_bar).hvplot.bar(
        x="model", y="RMSE", by="regime",
        title="LOOCV RMSE by model and regime",
        xlabel="", ylabel="RMSE (% microbial)",
        rot=30, width=820, height=360,
        legend="top_right", alpha=0.85,
    )
    return


@app.cell
def _(MODEL_NAMES, df_fit, functools, np, oof, operator, pl):
    _bal = df_fit["regime"].to_numpy() == "< 2% (BAL)"
    _y_b = df_fit["pct_microbial"].to_numpy()[_bal]

    _COL = {
        "M0_Cho":   "#94a3b8",
        "M1_Beta":  "#0070cc",
        "M2_LASSO": "#e08214",
        "M3_Ridge": "#1b7837",
        "M4_XGB":   "#d6282a",
        "M5_GPR":   "#7c3aed",
    }

    _ideal = pl.DataFrame({"x": [0.0, 2.0], "y": [0.0, 2.0]})
    _id_line = _ideal.hvplot.line(x="x", y="y", color="black",
                                   line_dash="dashed", line_width=1.5)

    _overlays = []
    for _m in MODEL_NAMES:
        _yp = oof[_m][_bal]
        _rmse = float(np.sqrt(np.nanmean((_y_b - _yp) ** 2)))
        _df_m = pl.DataFrame({"obs": _y_b.tolist(), "pred": _yp.tolist()})
        _overlays.append(
            _df_m.hvplot.scatter(
                x="obs", y="pred",
                color=_COL[_m], label=f"{_m} ({_rmse:.3f}%)",
                alpha=0.8, size=70,
            )
        )

    (_id_line * functools.reduce(operator.mul, _overlays)).opts(
        title="BAL regime (<2%) — all models vs observed",
        xlabel="Observed % microbial",
        ylabel="Predicted % microbial",
        xlim=(0, 2.2), ylim=(0, 2.2),
        width=560, height=440, legend_position="top_left",
    )
    return


@app.cell
def _(df_fit, oof, pl):
    _bal = df_fit["regime"].to_numpy() == "< 2% (BAL)"
    _df_gpr = (
        df_fit.filter(pl.Series(_bal.tolist()))
        .with_columns([
            pl.Series("gpr_pred", oof["M5_GPR"][_bal].tolist()),
            pl.Series("gpr_std",  oof["M5_GPR_std"][_bal].tolist()),
            pl.Series("cho_pred", oof["M0_Cho"][_bal].tolist()),
        ])
        .with_columns([
            (pl.col("gpr_pred") - pl.col("gpr_std") * 1.96).alias("ci_lo"),
            (pl.col("gpr_pred") + pl.col("gpr_std") * 1.96).alias("ci_hi"),
        ])
        .sort("pct_microbial")
        .with_row_index("sort_idx")
    )

    _obs  = _df_gpr.hvplot.scatter(
        x="sort_idx", y="pct_microbial",
        color="black", size=65, alpha=0.9, label="Observed",
        title="M5 GPR — BAL: predicted +/- 95% CI",
        xlabel="Sample (sorted by observed %)", ylabel="% microbial",
        width=820, height=360,
    )
    _pred = _df_gpr.hvplot.scatter(
        x="sort_idx", y="gpr_pred",
        color="#7c3aed", size=60, alpha=0.7, label="GPR pred",
    )
    _cho  = _df_gpr.hvplot.scatter(
        x="sort_idx", y="cho_pred",
        color="#94a3b8", size=50, alpha=0.5, marker="triangle",
        label="Cho pred",
    )
    _err  = _df_gpr.hvplot.errorbars(
        x="sort_idx", y="gpr_pred",
        yerr1="gpr_std", yerr2="gpr_std",
        color="#7c3aed", alpha=0.4, line_width=2,
    )
    _obs * _pred * _cho * _err
    return


@app.cell
def _(SAMPLE_TYPES, X_base, mo, pl, xgb, y_logit):
    _feat_names = ["ct_ACTB", "ct_16S", "delta"] + [f"type_{t}" for t in SAMPLE_TYPES]

    _xgb_full = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        random_state=42, verbosity=0,
    )
    _xgb_full.fit(X_base, y_logit)

    _df_imp = (
        pl.DataFrame({"feature": _feat_names, "importance": _xgb_full.feature_importances_.tolist()})
        .sort("importance", descending=True)
    )

    _chart = _df_imp.hvplot.bar(
        x="feature", y="importance",
        title="M4 XGBoost — Feature importance (full model)",
        xlabel="", ylabel="Gain",
        color="#d6282a", alpha=0.85, rot=35,
        width=620, height=340,
    )
    mo.vstack([_chart, mo.ui.table(_df_imp)])
    return


@app.cell
def _(
    BetaModel,
    HessianInversionWarning,
    Pipeline,
    Ridge,
    StandardScaler,
    X_base,
    X_beta,
    X_poly,
    df_fit,
    inv_logit,
    mo,
    np,
    pl,
    sm,
    warnings,
    xgb,
    y_logit,
    y_pct,
    y_prop,
):
    warnings.filterwarnings("ignore", category=HessianInversionWarning)
    _patients = df_fit["patient"].to_list() if "patient" in df_fit.columns else [None] * len(df_fit)
    _unique   = sorted(set(p for p in _patients if p is not None))

    _lpo_rows = []
    for _pat in _unique:
        _te  = np.array([p == _pat for p in _patients])
        _tr  = (~_te) & np.array([p is not None for p in _patients])
        if _te.sum() == 0 or _tr.sum() < 10:
            continue

        for _mname, _fn in [
            ("M1_Beta",  lambda Xtr, Xte, ytr: np.clip(
                BetaModel(y_prop[_tr], sm.add_constant(Xtr, has_constant="add"))
                  .fit(disp=False, maxiter=400, method="bfgs")
                  .predict(sm.add_constant(Xte, has_constant="add")) * 100, 0, 100)
            ),
            ("M3_Ridge", lambda Xtr, Xte, ytr: inv_logit(
                Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
                  .fit(Xtr, ytr).predict(Xte))
            ),
            ("M4_XGB",   lambda Xtr, Xte, ytr: inv_logit(
                xgb.XGBRegressor(n_estimators=200, max_depth=4, random_state=42, verbosity=0)
                  .fit(Xtr, ytr, verbose=False).predict(Xte))
            ),
        ]:
            try:
                _Xmat = X_poly if _mname == "M3_Ridge" else (X_beta if _mname == "M1_Beta" else X_base)
                _pred = _fn(_Xmat[_tr], _Xmat[_te], y_logit[_tr])
                for yt, yp in zip(y_pct[_te], _pred):
                    _lpo_rows.append({
                        "patient": str(_pat), "model": _mname,
                        "obs": float(yt), "pred": float(yp),
                        "abs_err": abs(float(yt) - float(yp)),
                    })
            except Exception:
                pass

    if _lpo_rows:
        _df_lpo = pl.DataFrame(_lpo_rows)
        _summary = (
            _df_lpo.with_columns(pl.col("abs_err").pow(2).alias("sq_err"))
            .group_by("model")
            .agg([
                pl.len().alias("n"),
                pl.col("sq_err").mean().sqrt().round(3).alias("LOPOCV_RMSE"),
                pl.col("abs_err").mean().round(3).alias("LOPOCV_MAE"),
            ])
            .sort("LOPOCV_RMSE")
        )
        mo.vstack([
            mo.md("## LOPOCV sensitivity check"),
            mo.ui.table(_summary),
            mo.md("> Compare LOPOCV_RMSE to LOOCV RMSE above. A gap >2x signals patient leakage risk."),
        ])
    else:
        mo.md("> No `patient` column found — skipping LOPOCV.")
    return


@app.cell
def _(
    BetaModel,
    ConvergenceWarning,
    HessianInversionWarning,
    Pipeline,
    Ridge,
    StandardScaler,
    X_base,
    df_fit,
    inv_logit,
    mo,
    np,
    pl,
    sm,
    warnings,
    y_logit,
    y_pct,
    y_prop,
):
    warnings.filterwarnings("ignore", category=HessianInversionWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    _bal_mask = df_fit["sample_type"].to_numpy() == "lung_bal"
    _n_bal    = int(_bal_mask.sum())
    _qbit_arr = df_fit["qbit_1"].fill_null(0.0).to_numpy()
    _has_qbit = int((_qbit_arr[_bal_mask] > 0).sum())

    if _has_qbit < 3:
        mo.md("> Too few BAL samples with Qubit data for ablation.")
    else:
        # BAL-only: keep only continuous features (ct_ACTB, ct_16S, delta)
        # OHE columns are all-zero within BAL → constant → breaks Beta MLE
        _X_cont = X_base[_bal_mask][:, :3]           # shape (n_bal, 3)
        _X_nq   = _X_cont                             # no qubit
        _X_wq   = np.column_stack([_X_cont, _qbit_arr[_bal_mask]])  # + qubit

        _y_b  = y_pct[_bal_mask]
        _yl_b = y_logit[_bal_mask]
        _yp_b = y_prop[_bal_mask]

        _abl = {
            "M1_noqubit": np.full(_n_bal, np.nan),
            "M1_qubit":   np.full(_n_bal, np.nan),
            "M3_noqubit": np.full(_n_bal, np.nan),
            "M3_qubit":   np.full(_n_bal, np.nan),
        }

        for _j in range(_n_bal):
            _tr = np.ones(_n_bal, dtype=bool)
            _tr[_j] = False

            for _tag, _X in [("noqubit", _X_nq), ("qubit", _X_wq)]:

                # M1 Beta — try multiple optimisers, take first that converges
                _Xtr = sm.add_constant(_X[_tr],   has_constant="add")
                _Xte = sm.add_constant(_X[[_j]], has_constant="add")
                _fit = None
                for _method in ["bfgs", "lbfgs", "nm"]:
                    try:
                        _fit = BetaModel(_yp_b[_tr], _Xtr).fit(
                            disp=False, maxiter=600, method=_method
                        )
                        break
                    except Exception:
                        continue
                if _fit is not None:
                    try:
                        _abl[f"M1_{_tag}"][_j] = float(
                            np.clip(_fit.predict(_Xte)[0] * 100, 0, 100)
                        )
                    except Exception:
                        _abl[f"M1_{_tag}"][_j] = np.nan
                else:
                    _abl[f"M1_{_tag}"][_j] = np.nan

                # M3 Ridge on logit scale
                try:
                    _pipe = Pipeline([
                        ("sc", StandardScaler()),
                        ("m",  Ridge(alpha=1.0)),
                    ])
                    _pipe.fit(_X[_tr], _yl_b[_tr])
                    _abl[f"M3_{_tag}"][_j] = float(
                        inv_logit(_pipe.predict(_X[[_j]])[0])
                    )
                except Exception:
                    _abl[f"M3_{_tag}"][_j] = np.nan

        _abl_rows = []
        for _k, _v in _abl.items():
            _valid = ~np.isnan(_v)
            _abl_rows.append({
                "variant":   _k,
                "n":         int(_valid.sum()),
                "RMSE":      round(float(np.sqrt(np.nanmean((_y_b - _v) ** 2))), 4),
                "MAE":       round(float(np.nanmean(np.abs(_y_b - _v))), 4),
                "n_failed":  int((~_valid).sum()),
            })

        _df_abl = pl.DataFrame(_abl_rows)

        _bar = _df_abl.hvplot.bar(
            x="variant", y="RMSE",
            title="BAL ablation: Qubit vs no-Qubit (LOOCV within BAL)",
            xlabel="", ylabel="RMSE (%)",
            color="#7c3aed", alpha=0.85,
            width=520, height=320,
        )

        mo.vstack([
            _bar,
            mo.ui.table(_df_abl),
            mo.md("> `n_failed`: folds where Beta MLE did not converge on any optimiser → prediction set to NaN."),
        ])
    return


@app.cell
def _(
    GaussianProcessRegressor,
    Matern,
    StandardScaler,
    WhiteKernel,
    X_base,
    mo,
    np,
    y_logit,
):
    import joblib, pathlib

    # ── Train GPR on full dataset (154 samples) and serialize for Pyodide ────────
    _kernel = 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
    _gpr    = GaussianProcessRegressor(
        kernel=_kernel, normalize_y=True,
        n_restarts_optimizer=2, random_state=42,
    )
    _scaler = StandardScaler()
    _X_sc   = _scaler.fit_transform(X_base)
    _gpr.fit(_X_sc, y_logit)

    _out = pathlib.Path("../models")
    _out.mkdir(exist_ok=True)
    joblib.dump({"gpr": _gpr, "scaler": _scaler}, _out / "gpr_ctomics.pkl")

    # Smoke-test: predict first 3 samples
    _mu, _sig = _gpr.predict(_scaler.transform(X_base[:3]), return_std=True)
    _pct_pred = 100.0 / (1.0 + np.exp(-np.clip(_mu, -30, 30)))

    mo.vstack([
        mo.md("## GPR — full model serialized to `models/gpr_ctomics.pkl`"),
        mo.md(
            f"- **Samples**: {X_base.shape[0]}  \n"
            f"- **Features**: {X_base.shape[1]} (ct_ACTB, ct_16S, delta + 6-type OHE)  \n"
            f"- **Kernel (fitted)**: `{_gpr.kernel_}`  \n"
            f"- **Scaler mean**: `{[round(float(v),3) for v in _scaler.mean_]}`  \n"
            f"- **Smoke-test predictions (first 3)**: `{[round(float(v),3) for v in _pct_pred]}` %"
        ),
        mo.md("> Load in Pyodide with: `import joblib; bundle = joblib.load('gpr_ctomics.pkl')`  \n"
              "> Predict: `mu, sigma = bundle['gpr'].predict(bundle['scaler'].transform(X), return_std=True)`"),
    ])
    return


if __name__ == "__main__":
    app.run()
