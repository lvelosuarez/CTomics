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
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

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
    # CTomics — OOD Validation: `lung_sputum`
    ## Testing trained models on new out-of-distribution sample type

    | Step | Description |
    |------|-------------|
    | **S1** | Load training data (M0–M5 trained on original 141 samples) |
    | **S2** | Load `lung_sputum` samples — never seen during training |
    | **S3** | Feature-space visualisation: where does sputum sit vs training cloud? |
    | **S4** | Apply all 6 models to sputum — no cross-validation, full-train prediction |
    | **S5** | GPR uncertainty analysis — does σ correctly signal OOD? |
    | **S6** | Metrics table: RMSE / MAE / MedAE per model on sputum |
    | **S7** | Regime breakdown and per-sample prediction table |
    | **S8** | Decision: retrain recommendation |

    **Key question**: Do models extrapolate gracefully to `lung_sputum`, or do they fail similarly to Cho on BAL?
    """)
    return


@app.cell
def _(pl):
    _raw = pl.read_csv(
        "../data/data_original.csv",
        null_values=["NA", "", "None"],
        infer_schema_length=200,
    )

    SAMPLE_TYPES = ["lung_bal", "oropharyngeal", "stool", "rectal_swab", "vaginal_sample"]
    TYPE_IDX = {t: i for i, t in enumerate(SAMPLE_TYPES)}

    PALETTE = {
        "stool":          "#2166ac",
        "oropharyngeal":  "#e08214",
        "rectal_swab":    "#1b7837",
        "vaginal_sample": "#d6282a",
        "lung_bal":       "#7c3aed",
        "lung_sputum":    "#f97316",   # new type — orange
    }

    MODEL_COL = {
        "M0_Cho":   "#94a3b8",
        "M1_Beta":  "#0070cc",
        "M2_LASSO": "#e08214",
        "M3_Ridge": "#1b7837",
        "M4_XGB":   "#d6282a",
        "M5_GPR":   "#7c3aed",
    }

    df_train = (
        _raw
        .with_columns([
            pl.col("ct_16S").cast(pl.Float64),
            pl.col("ct_ACTB").cast(pl.Float64),
            pl.col("delta").cast(pl.Float64),
            pl.col("pct_microbial").cast(pl.Float64),
            pl.col("qbit_1").cast(pl.Float64),
            pl.when(pl.col("pct_microbial") < 2).then(pl.lit("< 2% (BAL)"))
              .when(pl.col("pct_microbial") < 4).then(pl.lit("2–4% (Cho fail)"))
              .when(pl.col("pct_microbial") < 98).then(pl.lit("4–98% (valid)"))
              .otherwise(pl.lit("> 98% (sat.)"))
              .alias("regime"),
        ])
        .filter(
            pl.col("pct_microbial").is_not_null() &
            pl.col("ct_16S").is_not_null() &
            pl.col("ct_ACTB").is_not_null()
        )
        .with_row_index("row_id")
    )

    print(f"Training set: {len(df_train)} samples | types: {sorted(df_train['sample_type'].unique().to_list())}")
    df_train.head(3)
    return MODEL_COL, PALETTE, SAMPLE_TYPES, TYPE_IDX, df_train


@app.cell
def _(pl):
    _raw_sp = pl.read_csv(
        "../data/data_sputum.csv",
        null_values=["NA", "", "None"],
        infer_schema_length=200,
    )

    df_sputum = (
        _raw_sp
        .with_columns([
            pl.col("ct_16S").cast(pl.Float64),
            pl.col("ct_ACTB").cast(pl.Float64),
            pl.col("delta").cast(pl.Float64),
            pl.col("pct_microbial").cast(pl.Float64),
            pl.col("qbit_1").cast(pl.Float64),
            pl.lit("lung_sputum").alias("sample_type"),
            pl.when(pl.col("pct_microbial") < 2).then(pl.lit("< 2% (BAL)"))
              .when(pl.col("pct_microbial") < 4).then(pl.lit("2–4% (Cho fail)"))
              .when(pl.col("pct_microbial") < 98).then(pl.lit("4–98% (valid)"))
              .otherwise(pl.lit("> 98% (sat.)"))
              .alias("regime"),
        ])
        .filter(
            pl.col("pct_microbial").is_not_null() &
            pl.col("ct_16S").is_not_null() &
            pl.col("ct_ACTB").is_not_null()
        )
        .with_row_index("row_id")
    )

    print(f"Sputum OOD set: {len(df_sputum)} samples")
    print(f"pct_microbial: {df_sputum['pct_microbial'].min():.3f}% – {df_sputum['pct_microbial'].max():.3f}%  (median {df_sputum['pct_microbial'].median():.2f}%)")
    print(f"Qubit available: {(df_sputum['qbit_1'].is_not_null()).sum()} / {len(df_sputum)}")
    df_sputum
    return (df_sputum,)


@app.cell
def _(np):
    def logit(p):
        return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    def inv_logit(x):
        return 100.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def to_prop(pct):
        """% in [0,100] -> proportion in (0,1) with Smithson & Verkuilen shrinkage."""
        p  = np.clip(pct / 100.0, 1e-6, 1 - 1e-6)
        ns = len(p)
        return (p * (ns - 1) + 0.5) / ns

    def cho_model(delta):
        return np.clip(
            2.7201549 / (99.50267 * np.exp(-0.7218 * delta) + 0.02733),
            0.0, 100.0,
        )

    def build_features(df, sample_types, type_idx, degree=2):
        """Build X_base, X_poly, X_beta for a given dataframe."""
        from sklearn.preprocessing import PolynomialFeatures as PF
        n         = len(df)
        ct_actb   = df["ct_ACTB"].to_numpy()
        ct_16s    = df["ct_16S"].to_numpy()
        delta     = df["delta"].to_numpy()
        types     = df["sample_type"].to_list()

        # one-hot encode — unknown types map to all-zeros (OOD signal)
        type_ohe = np.zeros((n, len(sample_types)))
        for i, t in enumerate(types):
            idx = type_idx.get(t, -1)
            if idx >= 0:
                type_ohe[i, idx] = 1.0

        base_cont = np.column_stack([ct_actb, ct_16s, delta])
        X_base    = np.column_stack([base_cont, type_ohe])
        X_beta    = np.column_stack([base_cont, type_ohe[:, :-1]])
        _pf       = PF(degree=degree, include_bias=False)
        X_poly    = np.column_stack([_pf.fit_transform(base_cont), type_ohe])
        return X_base, X_poly, X_beta, ct_actb, ct_16s, delta

    return build_features, cho_model, inv_logit, logit, to_prop


@app.cell
def _(
    SAMPLE_TYPES,
    TYPE_IDX,
    build_features,
    df_sputum,
    df_train,
    logit,
    to_prop,
):

    # Training features
    X_tr_base, X_tr_poly, X_tr_beta, _, _, _ = build_features(
        df_train, SAMPLE_TYPES, TYPE_IDX
    )
    y_tr_pct   = df_train["pct_microbial"].to_numpy()
    y_tr_logit = logit(y_tr_pct / 100.0)
    y_tr_prop  = to_prop(y_tr_pct)
    n_tr       = len(df_train)

    # Sputum features — type_lung_sputum encodes as ALL ZEROS (OOD)
    X_sp_base, X_sp_poly, X_sp_beta, sp_actb, sp_16s, sp_delta = build_features(
        df_sputum, SAMPLE_TYPES, TYPE_IDX
    )
    y_sp_pct = df_sputum["pct_microbial"].to_numpy()
    n_sp     = len(df_sputum)

    # Confirm OHE is all-zeros for sputum (expected OOD behaviour)
    ohe_cols = X_sp_base[:, 3:]
    n_nonzero = int((ohe_cols.sum(axis=1) > 0).sum())
    print(f"Training set: {n_tr} samples | X_base: {X_tr_base.shape}")
    print(f"Sputum OOD:   {n_sp} samples | X_base: {X_sp_base.shape}")
    print(f"OHE non-zero rows in sputum: {n_nonzero} (expected 0 — pure OOD)")
    print(f"y_sp range: {y_sp_pct.min():.3f}% – {y_sp_pct.max():.3f}%")
    return (
        X_sp_base,
        X_sp_beta,
        X_sp_poly,
        X_tr_base,
        X_tr_beta,
        X_tr_poly,
        sp_delta,
        y_sp_pct,
        y_tr_logit,
        y_tr_prop,
    )


@app.cell
def _(PALETTE, df_sputum, df_train, functools, hv, mo, operator, pl):
    mo.md("## S3 — Feature space: sputum position vs training cloud")

    _TYPES_TR = sorted(df_train["sample_type"].unique().to_list())
    _TYPES_SP = ["lung_sputum"]

    # (ct_ACTB, ct_16S) scatter — training data
    _parts = []
    for _st in _TYPES_TR:
        _sub = df_train.filter(pl.col("sample_type") == _st)
        if _sub.shape[0] == 0:
            continue
        _parts.append(
            _sub.hvplot.scatter(
                x="ct_ACTB", y="ct_16S",
                color=PALETTE.get(_st, "grey"), label=_st,
                alpha=0.55, size=45,
            )
        )

    # Sputum — larger, distinct marker
    _sp_plot = df_sputum.hvplot.scatter(
        x="ct_ACTB", y="ct_16S",
        color=PALETTE["lung_sputum"], label="lung_sputum (OOD)",
        alpha=0.95, size=120, marker="star",
    )
    _parts.append(_sp_plot)

    # Add ct_16S = 33 detection-limit line
    _dl = hv.HLine(33).opts(
        color="red", line_dash="dashed", line_width=1.2, alpha=0.6
    )

    _p1 = functools.reduce(operator.mul, _parts) * _dl
    _p1 = _p1.opts(
        title="E3 — (ct_ACTB, ct_16S) feature plane: training vs sputum",
        xlabel="ct_ACTB", ylabel="ct_16S",
        width=680, height=420,
        legend_position="top_right",
    )

    # ΔCt vs pct_microbial with sputum highlighted
    _parts2 = []
    for _st in _TYPES_TR:
        _sub = df_train.filter(pl.col("sample_type") == _st)
        if _sub.shape[0] == 0:
            continue
        _parts2.append(
            _sub.hvplot.scatter(
                x="delta", y="pct_microbial",
                color=PALETTE.get(_st, "grey"), label=_st,
                alpha=0.45, size=40,
            )
        )
    _parts2.append(
        df_sputum.hvplot.scatter(
            x="delta", y="pct_microbial",
            color=PALETTE["lung_sputum"], label="lung_sputum (OOD)",
            alpha=0.95, size=120, marker="star",
        )
    )

    _p2 = functools.reduce(operator.mul, _parts2)
    _p2 = _p2.opts(
        title="ΔCt vs pct_microbial: training vs sputum",
        xlabel="ΔCt (ct_ACTB − ct_16S)", ylabel="% microbial",
        width=680, height=420,
        legend_position="top_left",
    )

    (_p1 + _p2).cols(2)
    return


@app.cell
def _(
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
    X_sp_base,
    X_sp_beta,
    X_sp_poly,
    X_tr_base,
    X_tr_beta,
    X_tr_poly,
    cho_model,
    inv_logit,
    mo,
    np,
    sm,
    sp_delta,
    warnings,
    xgb,
    y_tr_logit,
    y_tr_prop,
):
    warnings.filterwarnings("ignore", category=HessianInversionWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    n_sp_local = X_sp_base.shape[0]
    sp_pred = {
        "M0_Cho":     np.full(n_sp_local, np.nan),
        "M1_Beta":    np.full(n_sp_local, np.nan),
        "M2_LASSO":   np.full(n_sp_local, np.nan),
        "M3_Ridge":   np.full(n_sp_local, np.nan),
        "M4_XGB":     np.full(n_sp_local, np.nan),
        "M5_GPR":     np.full(n_sp_local, np.nan),
        "M5_GPR_std": np.full(n_sp_local, np.nan),
    }

    # M0 Cho — no training needed
    sp_pred["M0_Cho"] = cho_model(sp_delta)

    # M1 Beta regression — full train
    try:
        _Xtr = sm.add_constant(X_tr_beta, has_constant="add")
        _Xte = sm.add_constant(X_sp_beta, has_constant="add")
        _fit_beta = BetaModel(y_tr_prop, _Xtr).fit(
            disp=False, maxiter=400, method="bfgs"
        )
        sp_pred["M1_Beta"] = np.clip(_fit_beta.predict(_Xte) * 100, 0, 100)
    except Exception as _e:
        print(f"M1 Beta failed: {_e}")

    # M2 Logit-LASSO — full train
    _pl_lasso = Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.05, max_iter=5000))])
    _pl_lasso.fit(X_tr_poly, y_tr_logit)
    sp_pred["M2_LASSO"] = inv_logit(_pl_lasso.predict(X_sp_poly))

    # M3 Logit-Ridge — full train
    _pl_ridge = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    _pl_ridge.fit(X_tr_poly, y_tr_logit)
    sp_pred["M3_Ridge"] = inv_logit(_pl_ridge.predict(X_sp_poly))

    # M4 XGBoost — full train
    _xm = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        random_state=42, verbosity=0,
    )
    _xm.fit(X_tr_base, y_tr_logit, verbose=False)
    sp_pred["M4_XGB"] = inv_logit(_xm.predict(X_sp_base))

    # M5 GPR — full train
    _kernel = 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
    _gpr = GaussianProcessRegressor(
        kernel=_kernel, normalize_y=True,
        n_restarts_optimizer=2, random_state=42,
    )
    _sc_gpr = StandardScaler()
    _Xtr_sc = _sc_gpr.fit_transform(X_tr_base)
    _Xte_sc = _sc_gpr.transform(X_sp_base)
    _gpr.fit(_Xtr_sc, y_tr_logit)
    _mu_sp, _sig_sp = _gpr.predict(_Xte_sc, return_std=True)
    sp_pred["M5_GPR"]     = inv_logit(_mu_sp)
    sp_pred["M5_GPR_std"] = _sig_sp

    mo.md(f"> **Models trained on all {X_tr_base.shape[0]} training samples** and applied to {n_sp_local} sputum OOD samples. No CV — this is a strict train→test transfer.")
    return (sp_pred,)


@app.cell
def _(MODEL_COL, functools, mo, np, operator, pl, sp_pred, y_sp_pct):
    mo.md("## S4 — Observed vs Predicted: all models on sputum")

    MODEL_NAMES = ["M0_Cho", "M1_Beta", "M2_LASSO", "M3_Ridge", "M4_XGB", "M5_GPR"]
    _ideal = pl.DataFrame({"x": [0.0, 25.0], "y": [0.0, 25.0]})
    _id_line = _ideal.hvplot.line(
        x="x", y="y", color="black", line_dash="dashed", line_width=1.5
    )

    _panels = []
    for _m in MODEL_NAMES:
        _yp   = sp_pred[_m]
        _mask = ~np.isnan(_yp)
        _rmse = float(np.sqrt(np.nanmean((y_sp_pct - _yp) ** 2)))
        _df_m = pl.DataFrame({
            "obs":  y_sp_pct[_mask].tolist(),
            "pred": _yp[_mask].tolist(),
        })
        _sc = _df_m.hvplot.scatter(
            x="obs", y="pred",
            color=MODEL_COL[_m],
            #label=f"{_m} (RMSE={_rmse:.2f}%)",
            alpha=0.85, size=80,
            title=f"{_m}  RMSE={_rmse:.2f}%",
            xlabel="Observed %", ylabel="Predicted %",
            width=340, height=300,
        )
        _panels.append(_sc * _id_line)

    _r1 = functools.reduce(operator.add, _panels[:3])
    _r2 = functools.reduce(operator.add, _panels[3:])
    (_r1 + _r2).cols(3)
    return (MODEL_NAMES,)


@app.cell
def _(df_sputum, df_train, hv, mo, pl, sp_pred):
    mo.md("## S5 — GPR uncertainty: does σ correctly flag OOD?")

    # Compare GPR std on sputum vs BAL training samples
    _bal_mask = df_train["sample_type"].to_numpy() == "lung_bal"
    # We need the GPR std from the original LOOCV — proxy: use training GPR std
    # (we only have sp_pred here, so show sputum distribution and annotate)

    _df_gpr_sp = (
        df_sputum
        .with_columns([
            pl.Series("gpr_pred",    sp_pred["M5_GPR"].tolist()),
            pl.Series("gpr_std",     sp_pred["M5_GPR_std"].tolist()),
            pl.Series("cho_pred",    sp_pred["M0_Cho"].tolist()),
            pl.Series("ridge_pred",  sp_pred["M3_Ridge"].tolist()),
        ])
        .with_columns([
            (pl.col("gpr_pred") - pl.col("gpr_std") * 1.96).alias("ci_lo"),
            (pl.col("gpr_pred") + pl.col("gpr_std") * 1.96).alias("ci_hi"),
            (pl.col("gpr_std") > 0.5).cast(pl.Utf8).alias("high_uncertainty"),
        ])
        .sort("pct_microbial")
        .with_row_index("sort_idx")
    )

    _n_flagged = int((_df_gpr_sp["gpr_std"] > 0.5).sum())
    _mean_std  = float(_df_gpr_sp["gpr_std"].mean())

    _obs = _df_gpr_sp.hvplot.scatter(
        x="sort_idx", y="pct_microbial",
        color="black", size=70, alpha=0.9, label="Observed",
        title=f"M5 GPR — sputum OOD: predicted ± 95% CI  (mean σ={_mean_std:.3f}, flagged={_n_flagged})",
        xlabel="Sample (sorted by observed %)", ylabel="% microbial",
        width=860, height=380,
    )
    _pred = _df_gpr_sp.hvplot.scatter(
        x="sort_idx", y="gpr_pred",
        color="#7c3aed", size=65, alpha=0.75, label="GPR pred",
    )
    _cho = _df_gpr_sp.hvplot.scatter(
        x="sort_idx", y="cho_pred",
        color="#94a3b8", size=50, alpha=0.5, marker="triangle", label="Cho pred",
    )
    _ridge = _df_gpr_sp.hvplot.scatter(
        x="sort_idx", y="ridge_pred",
        color="#1b7837", size=50, alpha=0.5, marker="square", label="Ridge pred",
    )
    _err = _df_gpr_sp.hvplot.errorbars(
        x="sort_idx", y="gpr_pred",
        yerr1="gpr_std", yerr2="gpr_std",
        color="#7c3aed", alpha=0.35, line_width=2,
    )

    _flag_line = hv.HLine(0.5).opts(
        color="orange", line_dash="dashed", line_width=1.2, alpha=0.7
    )

    mo.vstack([
        _obs * _pred * _cho * _ridge * _err,
        mo.md(f"> Orange dashed = σ=0.5 logit re-assay threshold. **{_n_flagged}/{len(_df_gpr_sp)} sputum samples flagged** as high-uncertainty. Mean σ = {_mean_std:.3f}."),
        mo.ui.table(_df_gpr_sp.select(["id", "pct_microbial", "gpr_pred", "gpr_std", "ci_lo", "ci_hi", "high_uncertainty", "regime"])),
    ])
    return


@app.cell
def _(
    MODEL_NAMES,
    df_sputum,
    mean_absolute_error,
    mean_squared_error,
    mo,
    np,
    pl,
    r2_score,
    sp_pred,
    y_sp_pct,
):
    mo.md("## S6 — OOD metrics: all models on sputum (n=13)")

    REGIME_ORDER_OOD = ["< 2% (BAL)", "2–4% (Cho fail)", "4–98% (valid)", "> 98% (sat.)"]
    regimes_sp = df_sputum["regime"].to_list()

    def _met_ood(yt, yp, label, regime="sputum_all"):
        mask = ~np.isnan(yp)
        if mask.sum() < 2:
            return {"model": label, "regime": regime, "n": 0,
                    "RMSE": np.nan, "MAE": np.nan, "MedAE": np.nan, "R2": np.nan}
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

    # Global metrics
    rows_g = [_met_ood(y_sp_pct, sp_pred[_m], _m) for _m in MODEL_NAMES]
    df_global_ood = pl.DataFrame(rows_g)

    # Per-regime breakdown
    rows_r = []
    for _reg in REGIME_ORDER_OOD:
        _idx = np.where(np.array(regimes_sp) == _reg)[0]
        if len(_idx) < 2:
            continue
        for _m in MODEL_NAMES:
            rows_r.append(_met_ood(y_sp_pct[_idx], sp_pred[_m][_idx], _m, _reg))
    df_regime_ood = pl.DataFrame(rows_r) if rows_r else pl.DataFrame()

    # RMSE grouped bar chart
    _bar_data = pl.DataFrame([
        {"model": _m, "RMSE": float(row["RMSE"])}
        for _m, row in zip(MODEL_NAMES, rows_g)
        if row["RMSE"] is not None and not np.isnan(float(row["RMSE"]))
    ])

    _bar = _bar_data.hvplot.bar(
        x="model", y="RMSE",
        title="OOD RMSE on sputum — all models (lower is better)",
        xlabel="", ylabel="RMSE (% microbial)",
        color=["#94a3b8", "#0070cc", "#e08214", "#1b7837", "#d6282a", "#7c3aed"],
        alpha=0.85, rot=20, width=680, height=320,
    )

    _stack = [
        mo.md("### Global OOD performance on lung_sputum"),
        mo.ui.table(df_global_ood),
        _bar,
    ]
    if len(df_regime_ood) > 0:
        _stack += [
            mo.md("### Per-regime OOD performance"),
            mo.ui.table(df_regime_ood),
        ]
    mo.vstack(_stack)
    return


@app.cell
def _(MODEL_NAMES, df_sputum, mo, np, pl, sp_pred):
    mo.md("## S7 — Per-sample predictions")

    _rows = []
    for _i in range(len(df_sputum)):
        _row = {
            "id":           df_sputum["id"][_i] if "id" in df_sputum.columns else str(_i),
            "regime":       df_sputum["regime"][_i],
            "pct_obs":      round(float(df_sputum["pct_microbial"][_i]), 4),
            "qbit":         float(df_sputum["qbit_1"][_i]) if df_sputum["qbit_1"][_i] is not None else None,
            "delta":        round(float(df_sputum["delta"][_i]), 3),
            "ct_16S":       round(float(df_sputum["ct_16S"][_i]), 3),
            "ct_ACTB":      round(float(df_sputum["ct_ACTB"][_i]), 3),
        }
        for _m in MODEL_NAMES:
            _yp = sp_pred[_m][_i]
            _row[_m] = round(float(_yp), 3) if not np.isnan(_yp) else None
        _row["M5_σ"]   = round(float(sp_pred["M5_GPR_std"][_i]), 4)
        _row["M5_flag"] = "⚠ high-σ" if sp_pred["M5_GPR_std"][_i] > 0.5 else "ok"
        _rows.append(_row)

    df_per_sample = pl.DataFrame(_rows)

    mo.vstack([
        mo.ui.table(df_per_sample),
        mo.md("> **M5_σ**: GPR posterior std in logit units. `⚠ high-σ` = σ > 0.5 → consider re-assay."),
    ])
    return


@app.cell
def _(
    MODEL_COL,
    MODEL_NAMES,
    functools,
    hv,
    mo,
    np,
    operator,
    pl,
    sp_pred,
    y_sp_pct,
):
    mo.md("## S8 — Residuals: error structure across sputum samples")

    _parts = []
    for _m in MODEL_NAMES:
        _yp  = sp_pred[_m]
        _res = y_sp_pct - _yp
        _df  = pl.DataFrame({"residual": _res[~np.isnan(_res)].tolist()})
        _parts.append(
            _df.hvplot.hist(
                y="residual", bins=8,
                color=MODEL_COL[_m], alpha=0.65,
                label=_m, width=320, height=230,
                title=f"{_m} residuals",
                xlabel="Obs − Pred (%)", ylabel="count",
            )
        )

    _zero_lines = [hv.VLine(0).opts(color="red", line_dash="dashed",
                                     line_width=1.2, alpha=0.6) for _ in MODEL_NAMES]
    _panels_r = [_p * _z for _p, _z in zip(_parts, _zero_lines)]
    _r1 = functools.reduce(operator.add, _panels_r[:3])
    _r2 = functools.reduce(operator.add, _panels_r[3:])
    (_r1 + _r2).cols(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## S9 — OOD Assessment and Retrain Decision

    | Criterion | Finding | Implication |
    |-----------|---------|-------------|
    | **Feature space** | Sputum sits outside training convex hull — lowest ct_ACTB of any type | Pure OOD; sample_type one-hot encodes as all-zeros |
    | **GPR σ** | Mean σ > 0.5 logit on sputum? | High σ = model correctly signals uncertainty — GPR behaves as expected |
    | **RMSE vs Cho** | Best model RMSE < Cho RMSE on sputum? | If yes: partial generalisation; if no: models degrade to Cho-level at OOD |
    | **Regime distribution** | Median pct ~3.6%, range 1–17% — straddles Cho fail zone | No training type covers this range cleanly |
    | **Qubit coverage** | 100% Qubit available for sputum | GPR+Qubit ablation immediately feasible after retrain |

    **Recommended next steps**:
    - If GPR RMSE on sputum is substantially worse than LOOCV BAL RMSE (2.28%) → **retrain mandatory**
    - Add `lung_sputum` as a new one-hot type; retrain M2–M5 on 141 + 13 = 154 samples
    - Use **LOSO-CV** (leave-one-sample-type-out): hold all 13 sputum out as test fold, train on 141
    - Compare LOSO-CV RMSE vs the OOD RMSE computed here — the gap estimates overfitting to sputum type
    - Run Qubit ablation on sputum (same design as BAL ablation in modelling.py)
    - Consider **Beta Regression (M1)** re-evaluation: sputum pct range (1–17%) avoids the near-zero boundary instability that hurt M1 in BAL
    """)
    return


if __name__ == "__main__":
    app.run()
