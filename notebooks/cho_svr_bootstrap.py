import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import hvplot.polars
    import matplotlib.pyplot as plt
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.neighbors import KNeighborsRegressor
    from scipy import stats
    import warnings
    warnings.filterwarnings("ignore")
    return (
        GaussianProcessRegressor,
        GradientBoostingRegressor,
        KNeighborsRegressor,
        OneHotEncoder,
        RBF,
        RandomForestRegressor,
        SVR,
        StandardScaler,
        WhiteKernel,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pl,
        plt,
        r2_score,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Cho Model E  vs  SVR-RBF — Step-by-Step Bootstrap Comparison

    **Central question:**
    Does a Support Vector Regressor with an RBF kernel outperform the
    fixed Cho logistic model for predicting % microbial reads from
    ΔCt = ACTB − 16S?  Especially in the **< 4 % failure zone** where
    the Cho asymptote structurally compresses all predictions toward
    the same floor value?

    **Notebook structure**
    1. Cho equation and asymptote analysis
    2. Build dataset (Cho paper + BAL COPD samples)
    3. Fit one SVR manually — understand C, ε, γ
    4. Bootstrap loop: 100 × out-of-bag evaluation
    5. Summary table and paired statistical test
    """)
    return


@app.cell
def _(np):
    def cho_model(delta_ct):
        """Cho et al. 2021 Model E.  Input: ΔCt scalar or array."""
        d = np.asarray(delta_ct, dtype=float)
        return np.clip(2.7201549 / (99.50267 * np.exp(-0.7218 * d) + 0.02733),
                       0.0, 100.0)

    # --- sanity check ---
    test_d = np.array([-10, -5, 0, 5, 10, 15, 20])
    print("Cho model — ΔCt → predicted % microbial")
    print(f"{'ΔCt':>6}   {'pred%':>8}")
    for _d, _p in zip(test_d, cho_model(test_d)):
        print(f"  {_d:>4}     {_p:>7.4f}")

    print(f"\nFloor (ΔCt = −100) : {cho_model(np.array([-100.]))[0]:.6f} %")
    print(f"Ceiling (ΔCt = +100): {cho_model(np.array([+100.]))[0]:.4f} %")
    print()
    print("Practical problem: at ΔCt = −10 the model is already at ~0.027 %.")
    print("Every BAL sample with ΔCt < −5 gets the same compressed prediction.")
    return (cho_model,)


@app.cell
def _(pl):
    data = pl.read_csv("../data/data.csv") 
    print(f"Combined dataset: {data.shape[0]} samples")
    print()
    n_low = (data["pct_microbial"] < 4.0).sum()
    print(f"Samples < 4 % microbial (Cho failure zone): {n_low} / {data.shape[0]}")
    print(f"ΔCt range:  {data['delta'].min():.2f}  to  {data['delta'].max():.2f}")
    return (data,)


@app.cell
def _(data):
    X = data["delta"].to_numpy()
    y = data["pct_microbial"].to_numpy()
    low_mask = y < 4.0
    print(f"X (ΔCt):          shape={X.shape}   range=[{X.min():.2f}, {X.max():.2f}]")
    print(f"y (% microbial):  shape={y.shape}   range=[{y.min():.2f}, {y.max():.2f}]")
    print(f"Near-zero (<4%):  {low_mask.sum()} samples")
    return X, low_mask, y


@app.cell
def _(data):
    data.hvplot.scatter(x='delta', 
                        y='pct_microbial',
                        by= 'sample_type', 
                        height = 500, 
                        width = 700, 
                        size = 100, 
                        alpha = 0.6,
                       title = " Delta vs % microbial in reads")
    return


@app.cell
def _(data):
    data.hvplot.scatter(x='ct_16S', 
                        y='ct_ACTB',
                        by= 'sample_type', 
                        height = 500, 
                        width = 700, 
                        size = 100, 
                        alpha = 0.6,
                       title = " Delta vs % microbial in reads")
    return


@app.cell
def _(data):
    data.sample()
    return


@app.cell
def _(data, np):
    """Absolute Ct comparison — same ΔCt, very different biological context."""
    for _t in sorted(data["sample_type"].unique().to_list()):
        _d = data.filter(data["sample_type"] == _t)
        print(f"{_t}  (n={len(_d)})")
        for _col in ["ct_16S", "ct_ACTB", "delta", "pct_microbial"]:
            _v = _d[_col].to_numpy()
            print(f"  {_col:<16}  median={np.median(_v):6.2f}  "
                  f"range=[{_v.min():.2f}, {_v.max():.2f}]")
        print()

    # Spotlight: ΔCt 4–7 across sample types
    print("─" * 74)
    print("Samples with ΔCt 4–7  (same ΔCt window, different sample types):")
    print(f"{'id':<12} {'type':<18} {'ct_16S':>7} {'ct_ACTB':>8} "
          f"{'delta':>7} {'pct_microbial':>14}")
    _w = data.filter((data["delta"] >= 4) & (data["delta"] <= 7))
    for _row in _w.sort("sample_type").iter_rows(named=True):
        print(f"{_row['id']:<12} {_row['sample_type']:<18} "
              f"{_row['ct_16S']:>7.2f} {_row['ct_ACTB']:>8.2f} "
              f"{_row['delta']:>7.2f} {_row['pct_microbial']:>14.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cho model — baseline

    Fixed parameters, no refit.  Applying the published equation to our
    combined dataset gives an honest cross-sample-type baseline because
    the BAL samples are completely out-of-distribution for Cho et al.
    """)
    return


@app.cell
def _(
    X,
    cho_model,
    low_mask,
    mean_absolute_error,
    mean_squared_error,
    np,
    r2_score,
    y,
):
    def print_metrics(yt, yp, label):
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae  = float(mean_absolute_error(yt, yp))
        r2   = float(r2_score(yt, yp))
        bias = float(np.mean(yp - yt))
        print(f"  {label}")
        print(f"    RMSE {rmse:.4f}%   MAE {mae:.4f}%   R² {r2:.4f}   Bias {bias:+.4f}%")

    cho_pred_full = cho_model(X)
    cho_pred_low  = cho_model(X[low_mask])

    print("=== Cho Model E — baseline (no bootstrap) ===")
    print_metrics(y,           cho_pred_full, "Full combined dataset")
    print_metrics(y[low_mask], cho_pred_low,  "Near-zero zone  (<4%)")
    print()
    print("Positive bias in the near-zero zone = systematic over-prediction.")
    print("This is the asymptote floor pulling predictions away from zero.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## SVR-RBF — understanding each hyperparameter

    SVR imposes **no functional form**.  It learns purely from data.
    There is no baked-in asymptote.

    ---

    ### C — regularisation strength
    Trade-off between fitting training points vs. keeping the model smooth.
    - **Small C** (0.1): smooth curve, tolerates large errors → high bias
    - **Large C** (100): tight fit, penalises all errors → may overfit
    - Start with **C = 10** for noisy biological data

    ### ε (epsilon) — insensitive tube
    Errors *inside* ±ε are ignored completely.  Only points *outside*
    the tube (the support vectors) determine the model.
    - **Large ε**: fewer support vectors, simpler, noise-robust
    - For % microbial reads on a 0–100 scale: **ε = 2** is a good start

    ### γ (gamma) — RBF kernel width
    K(xᵢ, xⱼ) = exp(−γ · |xᵢ − xⱼ|²)
    - **Small γ**: wide influence, smooth global fit
    - **Large γ**: narrow influence, spiky, risk of overfitting
    - `gamma='scale'` sets γ = 1 / (n_features × Var(X))  ← use this

    ---

    ### Critical: StandardScaler is mandatory
    SVR is distance-based.  ΔCt values (~−15 to +25) dominate distance
    calculations if not normalised.  Always fit the scaler on **in-bag
    (training) data only** — fitting on the full data before splitting
    is data leakage and will make your results look better than they are.
    """)
    return


@app.cell
def _(SVR, StandardScaler, X, cho_model, mean_squared_error, np, r2_score, y):

    # 1. Scale
    sc_demo  = StandardScaler()
    X_sc     = sc_demo.fit_transform(X.reshape(-1, 1))
    print(f"Before scaling:  mean={X.mean():.2f}  std={X.std():.2f}")
    print(f"After  scaling:  mean={X_sc.mean():.2f}  std={X_sc.std():.2f}")
    print()

    # 2. Fit SVR
    C_demo, eps_demo = 10.0, 2.0
    svr_demo = SVR(kernel="rbf", C=C_demo, epsilon=eps_demo, gamma="scale")
    svr_demo.fit(X_sc, y)

    n_sv = len(svr_demo.support_vectors_)
    print(f"Hyperparameters: C={C_demo}  ε={eps_demo}  γ=scale")
    print(f"Support vectors: {n_sv}/{len(X)}  ({100*n_sv/len(X):.0f}%)")
    print("  (points outside the ε-tube that define the fitted curve)")
    print()

    # 3. In-sample metrics — biased, for inspection only
    y_hat = np.clip(svr_demo.predict(X_sc), 0., 100.)
    rmse  = float(np.sqrt(mean_squared_error(y, y_hat)))
    r2    = float(r2_score(y, y_hat))
    print(f"In-sample (optimistic): RMSE={rmse:.4f}%  R²={r2:.4f}")
    print("→ bootstrap in cell 10 gives the honest out-of-sample version")
    print()

    # 4. Compare predictions across the near-zero ΔCt range
    delta_grid  = np.linspace(-15, 5, 200)
    X_grid_sc   = sc_demo.transform(delta_grid.reshape(-1, 1))
    svr_curve   = np.clip(svr_demo.predict(X_grid_sc), 0., 100.)
    cho_curve   = cho_model(delta_grid)

    print("Prediction comparison — near-zero regime:")
    print(f"{'ΔCt':>6}   {'Cho%':>8}   {'SVR%':>8}   {'SVR−Cho':>9}")
    print("─" * 40)
    for d, c, s in zip(delta_grid[::25], cho_curve[::25], svr_curve[::25]):
        print(f"  {d:>5.1f}   {c:>8.4f}   {s:>8.4f}   {s-c:>+9.4f}")

    print()
    print(f"Cho prediction at ΔCt=−15:  {cho_curve[0]:.4f}%  (near the floor)")
    print(f"SVR prediction at ΔCt=−15:  {svr_curve[0]:.4f}%")
    print("SVR trained on near-zero BAL data can predict below the Cho floor.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Bootstrap resampling — why and how

    With ~116 combined samples, a single 80/20 split leaves ≈23 test
    points.  The <4% zone might contribute only 5 of those — far too few
    for a reliable metric.

    **One bootstrap iteration**
    1. Draw n samples **with replacement** from the full dataset.
       On average ~63.2% unique samples are drawn (in-bag).
    2. The ~36.8% never drawn form the **out-of-bag (OOB)** set —
       completely unseen by the fitted model.
    3. Fit both models on the in-bag set.
    4. Evaluate on the OOB set and record the metrics.

    After 100 iterations you have a **distribution** of RMSE values:
    - Mean: expected performance
    - SD: how stable the model is across different training sets
    - Paired differences (Cho−SVR per iteration): which wins more often?

    **Key data-leakage rule:**
    The StandardScaler is fit **inside the loop on in-bag data only**,
    then applied to OOB.  Fitting it on the full dataset before the
    loop would leak information from OOB samples into model training.
    """)
    return


@app.cell
def _(
    GaussianProcessRegressor,
    GradientBoostingRegressor,
    KNeighborsRegressor,
    RBF,
    RandomForestRegressor,
    SVR,
    StandardScaler,
    WhiteKernel,
    X,
    cho_model,
    mean_absolute_error,
    mean_squared_error,
    np,
    r2_score,
    y,
):

    # ── hyperparameters ──────────────────────────────────
    N_BOOT  = 100
    C, EPSILON, GAMMA = 10.0, 2.0, "scale"
    SEED    = 42
    # ──────────────────────────────────────────────────────

    MODELS = {
        "SVR": lambda: SVR(kernel="rbf", C=C, epsilon=EPSILON, gamma=GAMMA),
        "RF":  lambda: RandomForestRegressor(n_estimators=200, random_state=SEED),
        "GBR": lambda: GradientBoostingRegressor(n_estimators=100, random_state=SEED),
        "GPR": lambda: GaussianProcessRegressor(
                           kernel=RBF() + WhiteKernel(), normalize_y=True,
                           random_state=SEED),
        "KNN": lambda: KNeighborsRegressor(n_neighbors=5),
    }

    # boot dict — one sub-dict per model including "cho"
    _keys = ["rmse", "mae", "r2", "bias", "rmse_low", "mae_low"]
    boot = {name: {k: [] for k in _keys} for name in ["cho"] + list(MODELS)}
    boot["n_oob"] = []; boot["n_oob_low"] = []

    def _met(yt, yp):
        """Return (rmse, mae, r2, bias) as floats."""
        return (
            float(np.sqrt(mean_squared_error(yt, yp))),
            float(mean_absolute_error(yt, yp)),
            float(r2_score(yt, yp)),
            float(np.mean(yp - yt)),
        )

    rng_b = np.random.default_rng(SEED)
    n = len(X)

    for _i in range(N_BOOT):
        ib  = rng_b.integers(0, n, size=n)
        oob = np.setdiff1d(np.arange(n), ib)
        if len(oob) < 5:
            continue

        Xib, yib   = X[ib],  y[ib]
        Xoob, yoob = X[oob], y[oob]
        oob_low    = yoob < 4.0

        # Cho — fixed model, no fitting
        cho_oob = cho_model(Xoob)
        rm, ma, r2c, bi = _met(yoob, cho_oob)
        boot["cho"]["rmse"].append(rm);  boot["cho"]["mae"].append(ma)
        boot["cho"]["r2"].append(r2c);   boot["cho"]["bias"].append(bi)
        if oob_low.sum() >= 3:
            rm_l, ma_l, _, _ = _met(yoob[oob_low], cho_oob[oob_low])
            boot["cho"]["rmse_low"].append(rm_l)
            boot["cho"]["mae_low"].append(ma_l)

        # Scale once on in-bag, apply to OOB — used by all ML models
        sc      = StandardScaler()
        Xib_sc  = sc.fit_transform(Xib.reshape(-1, 1))
        Xoob_sc = sc.transform(Xoob.reshape(-1, 1))

        for _name, factory in MODELS.items():
            mdl  = factory()
            mdl.fit(Xib_sc, yib)
            pred = np.clip(mdl.predict(Xoob_sc), 0., 100.)
            rm, ma, r2m, bi = _met(yoob, pred)
            boot[_name]["rmse"].append(rm);  boot[_name]["mae"].append(ma)
            boot[_name]["r2"].append(r2m);   boot[_name]["bias"].append(bi)
            if oob_low.sum() >= 3:
                rm_l, ma_l, _, _ = _met(yoob[oob_low], pred[oob_low])
                boot[_name]["rmse_low"].append(rm_l)
                boot[_name]["mae_low"].append(ma_l)

        boot["n_oob"].append(len(oob))
        boot["n_oob_low"].append(int(oob_low.sum()))

    valid = len(boot["cho"]["rmse"])
    print(f"Bootstrap complete: {valid}/{N_BOOT} valid iterations")
    print(f"Mean OOB size      : {np.mean(boot['n_oob']):.1f} samples/iter")
    print(f"Mean OOB near-zero : {np.mean(boot['n_oob_low']):.1f} samples/iter")
    return MODELS, boot


@app.cell
def _(MODELS, boot, np):
    def _row(name, metric):
        a = np.array(boot[name][metric])
        return a.mean(), a.std(), np.percentile(a,5), np.median(a), np.percentile(a,95)

    model_names = ["cho"] + list(MODELS)
    hdr = f"{'Model':<8} {'Mean':>8} {'SD':>7} {'p5':>8} {'Median':>8} {'p95':>8}"
    sep = "─" * 55

    for zone_label, metric_keys in [
        ("Full dataset",          ["rmse", "mae", "r2", "bias"]),
        ("Near-zero zone (<4%)",  ["rmse_low", "mae_low"]),
    ]:
        print(f"\n{zone_label}")
        for metric in metric_keys:
            print(f"\n  {metric.upper().replace('_LOW','')}")
            print(f"  {hdr}"); print(f"  {sep}")
            for _name in model_names:
                if metric not in boot[_name] or len(boot[_name][metric]) == 0:
                    continue
                mn, sd, p5, med, p95 = _row(_name, metric)
                print(f"  {_name:<8} {mn:>8.4f} {sd:>7.4f} {p5:>8.4f} {med:>8.4f} {p95:>8.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Statistical comparison

    All models are evaluated on the **same OOB set** each iteration,
    so we compute per-iteration **ΔRMSE = Cho_RMSE − model_RMSE**.
    Positive = each ML model has lower error = ML model is better.

    We use the **Wilcoxon signed-rank test** (paired, non-parametric).
    H₀: median ΔRMSE = 0 (models are equivalent).

    The bootstrapped 95% CI shows the practical magnitude of the
    improvement — important for the paper, not just the p-value.
    """)
    return


@app.cell
def _(MODELS, boot, np, stats):
    for zone, cho_key, ml_key in [
        ("Full dataset",  "rmse",     "rmse"),
        ("Near-zero <4%", "rmse_low", "rmse_low"),
    ]:
        print(f"\n══ {zone} ══")
        cho_arr = np.array(boot["cho"][cho_key])
        for _name in MODELS:
            ml_arr = np.array(boot[_name][ml_key])
            n_ = min(len(cho_arr), len(ml_arr))
            if n_ < 10:
                print(f"  {_name}: insufficient data"); continue
            diff = cho_arr[:n_] - ml_arr[:n_]
            _, p = stats.wilcoxon(diff)
            ci   = np.percentile(diff, [2.5, 97.5])
            winner_pct = 100 * np.mean(diff > 0)
            sig = "✓" if p < 0.05 else "–"
            print(f"  {_name:<5}  ΔRMSE={diff.mean():+.3f}%  "
                  f"CI=[{ci[0]:+.2f},{ci[1]:+.2f}]  "
                  f"wins={winner_pct:.0f}%  p={p:.2e}  {sig}")
    return


@app.cell
def _(MODELS, boot, np):
    print("Bias (positive = over-prediction, full OOB set)")
    for _name in ["cho"] + list(MODELS):
        b = np.array(boot[_name]["bias"])
        print(f"  {_name:<5}  {b.mean():+.4f}% ± {b.std():.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Hyperparameter grid search — SVR × KNN

    - SVR: C ∈ {1, 10, 100} × ε ∈ {1, 2, 5}, γ=scale → 9 configs
    - KNN: k ∈ {3, 5, 7, 10, 15} → 5 configs
    - Cho included as fixed reference row every iteration
    - 200 bootstrap iterations, optimise mean OOB RMSE in <4% zone
    """)
    return


@app.cell
def _(
    KNeighborsRegressor,
    SVR,
    StandardScaler,
    X,
    cho_model,
    mean_absolute_error,
    mean_squared_error,
    np,
    y,
):
    SVR_GRID = [{"C": c, "epsilon": e, "gamma": "scale"}
                for c in [1, 10, 100] for e in [1, 2, 5]]
    KNN_GRID = [{"n_neighbors": k} for k in [3, 5, 7, 10, 15]]

    N_BOOT_TUNE = 200
    SEED_TUNE   = 99

    all_configs = {}
    for _p in SVR_GRID:
        _lbl = f"SVR_C{_p['C']}_e{_p['epsilon']}"
        all_configs[_lbl] = {"type": "svr", "params": _p}
    for _p in KNN_GRID:
        _lbl = f"KNN_k{_p['n_neighbors']}"
        all_configs[_lbl] = {"type": "knn", "params": _p}

    boot_tune = {lbl: {"rmse_low": [], "mae_low": []}
                 for lbl in ["cho"] + list(all_configs)}
    boot_tune["n_oob_low"] = []

    _rng_t = np.random.default_rng(SEED_TUNE)
    _n = len(X)

    for _i in range(N_BOOT_TUNE):
        _ib  = _rng_t.integers(0, _n, size=_n)
        _oob = np.setdiff1d(np.arange(_n), _ib)
        if len(_oob) < 5:
            continue
        _Xoob, _yoob = X[_oob], y[_oob]
        _oob_low = _yoob < 4.0
        if _oob_low.sum() < 3:
            continue

        _sc      = StandardScaler()
        _Xib_sc  = _sc.fit_transform(X[_ib].reshape(-1, 1))
        _Xoob_sc = _sc.transform(_Xoob.reshape(-1, 1))
        _yib     = y[_ib]

        # Cho reference — no fitting
        _cho_pred = cho_model(_Xoob[_oob_low])
        boot_tune["cho"]["rmse_low"].append(
            float(np.sqrt(mean_squared_error(_yoob[_oob_low], _cho_pred))))
        boot_tune["cho"]["mae_low"].append(
            float(mean_absolute_error(_yoob[_oob_low], _cho_pred)))

        for _lbl, _cfg in all_configs.items():
            _mdl = (SVR(kernel="rbf", **_cfg["params"]) if _cfg["type"] == "svr"
                    else KNeighborsRegressor(**_cfg["params"]))
            _mdl.fit(_Xib_sc, _yib)
            _pred = np.clip(_mdl.predict(_Xoob_sc), 0., 100.)
            boot_tune[_lbl]["rmse_low"].append(
                float(np.sqrt(mean_squared_error(_yoob[_oob_low], _pred[_oob_low]))))
            boot_tune[_lbl]["mae_low"].append(
                float(mean_absolute_error(_yoob[_oob_low], _pred[_oob_low])))

        boot_tune["n_oob_low"].append(int(_oob_low.sum()))

    _valid = len(boot_tune["cho"]["rmse_low"])
    print(f"Tuning bootstrap: {_valid}/{N_BOOT_TUNE} valid iterations")
    print(f"Mean OOB near-zero: {np.mean(boot_tune['n_oob_low']):.1f} samples/iter")
    return all_configs, boot_tune


@app.cell
def _(all_configs, boot_tune, np, stats):
    _results = []
    for _lbl in ["cho"] + list(all_configs):
        _a = np.array(boot_tune[_lbl]["rmse_low"])
        _results.append((_lbl, _a.mean(), _a.std(),
                         np.percentile(_a, 5), np.median(_a), np.percentile(_a, 95)))
    _results.sort(key=lambda _r: _r[1])

    _cho_mean = np.mean(boot_tune["cho"]["rmse_low"])
    _hdr = f"{'Config':<18} {'Mean':>8} {'SD':>7} {'p5':>8} {'Median':>8} {'p95':>8}"
    _sep = "─" * 65
    print("Near-zero RMSE (<4%) — ranked by mean  (lower = better)\n")
    print(_hdr); print(_sep)
    for _r in _results:
        _tag = " ← Cho baseline" if _r[0] == "cho" else f"  Δ={_r[1]-_cho_mean:+.3f}"
        print(f"{_r[0]:<18} {_r[1]:>8.4f} {_r[2]:>7.4f} {_r[3]:>8.4f} "
              f"{_r[4]:>8.4f} {_r[5]:>8.4f}{_tag}")

    # Wilcoxon: top-3 non-Cho configs vs Cho
    _cho_arr = np.array(boot_tune["cho"]["rmse_low"])
    _top3 = [_r for _r in _results if _r[0] != "cho"][:3]
    print("\nWilcoxon (top-3 vs Cho, near-zero RMSE):")
    for _r in _top3:
        _ml_arr = np.array(boot_tune[_r[0]]["rmse_low"])
        _nw = min(len(_cho_arr), len(_ml_arr))
        _diff = _cho_arr[:_nw] - _ml_arr[:_nw]
        _, _p = stats.wilcoxon(_diff)
        _ci = np.percentile(_diff, [2.5, 97.5])
        _sig = "✓" if _p < 0.05 else "–"
        print(f"  {_r[0]:<18} ΔRMSE={_diff.mean():+.3f}%  "
              f"CI=[{_ci[0]:+.2f},{_ci[1]:+.2f}]  p={_p:.2e}  {_sig}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Final model — SVR(C=1, ε=1, γ=scale)

    Grid search (200 bootstrap iterations) identified **SVR(C=1, ε=1)** as the
    optimal configuration: mean OOB RMSE 7.78% in the <4% zone vs Cho's 11.32%
    (ΔRMSE = +3.54%, p = 1.9×10⁻²³).

    The cells below refit this model on the full 141-sample dataset for
    inference and generate the three main paper figures.
    """)
    return


@app.cell
def _(SVR, StandardScaler, X, boot_tune, np, y):
    sc_final = StandardScaler()
    svr_final = SVR(kernel="rbf", C=1, epsilon=1, gamma="scale")
    svr_final.fit(sc_final.fit_transform(X.reshape(-1, 1)), y)

    _r = np.array(boot_tune["SVR_C1_e1"]["rmse_low"])
    _c = np.array(boot_tune["cho"]["rmse_low"])
    print("═══ Final model: SVR(C=1, ε=1, kernel=rbf, γ=scale) ═══\n")
    print("OOB near-zero RMSE (<4% microbial) — 200 bootstrap iterations:")
    print(f"  SVR(C=1, ε=1) : {_r.mean():.3f}% ± {_r.std():.3f}%"
          f"   median={np.median(_r):.3f}%")
    print(f"  Cho baseline  : {_c.mean():.3f}% ± {_c.std():.3f}%"
          f"   median={np.median(_c):.3f}%")
    print(f"  ΔRMSE (Cho−SVR): +{(_c - _r).mean():.3f}%")
    print(f"\n  Support vectors (full-dataset fit): "
          f"{len(svr_final.support_vectors_)}/{len(X)}")
    return sc_final, svr_final


@app.cell
def _(X, cho_model, data, np, plt, sc_final, svr_final, y):
    _fig1, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35},
    )

    _dg      = np.linspace(-4, 27, 400)
    _dg_zoom = np.linspace(-4, 10, 300)
    _prop    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _types   = sorted(data["sample_type"].unique().to_list())

    # ── Left panel: full range, Cho only ───────────────────────────────
    _ax1.axhspan(0, 4, color="salmon", alpha=0.15, zorder=0)
    _ax1.axhline(4, color="salmon", lw=1.2, ls="--", alpha=0.7)
    _ax1.text(26.5, 5, "< 4 % failure zone", color="#c0392b",
              ha="right", va="bottom", fontsize=9)
    _ax1.plot(_dg, cho_model(_dg), color="black", lw=2.5,
              label="Cho Model E", zorder=3)
    for _i, _t in enumerate(_types):
        _m = (data["sample_type"] == _t).to_numpy()
        _ax1.scatter(X[_m], y[_m], color=_prop[_i],
                     alpha=0.55, s=45, label=str(_t), zorder=4)
    _ax1.set_xlabel("ΔCt (ACTB − 16S)", fontsize=12)
    _ax1.set_ylabel("% microbial reads", fontsize=12)
    _ax1.set_title("Full ΔCt range — Cho global fit", fontsize=12)
    _ax1.set_xlim(-4.5, 27.5)
    _ax1.set_ylim(-2, 105)
    _ax1.legend(fontsize=9, loc="upper left")

    # ── Right panel: <4% zone zoom, both models, only near-zero data ───
    _low = y < 4.0
    _ax2.axhspan(0, 4, color="salmon", alpha=0.20, zorder=0)
    _ax2.axhline(4, color="salmon", lw=1.2, ls="--", alpha=0.8)
    _ax2.plot(_dg_zoom, cho_model(_dg_zoom), color="black", lw=2.5,
              label="Cho Model E", zorder=3)
    _ax2.plot(_dg_zoom,
              np.clip(svr_final.predict(
                  sc_final.transform(_dg_zoom.reshape(-1, 1))), 0, 100),
              color="#1f77b4", lw=2.5, label="SVR (C=1, ε=1)", zorder=3)
    for _i, _t in enumerate(_types):
        _m = (data["sample_type"] == _t).to_numpy() & _low
        if _m.sum() == 0:
            continue
        _ax2.scatter(X[_m], y[_m], color=_prop[_i],
                     alpha=0.70, s=55, zorder=4)
    _ax2.set_xlabel("ΔCt (ACTB − 16S)", fontsize=12)
    _ax2.set_ylabel("% microbial reads", fontsize=12)
    _ax2.set_title("Near-zero zone (<4%) — SVR vs Cho", fontsize=12)
    _ax2.set_xlim(-4.5, 10.5)
    _ax2.set_ylim(-0.3, 15)
    _ax2.legend(fontsize=10)

    _fig1.suptitle("Figure 1 — Model comparison", fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X, cho_model, np, plt, sc_final, svr_final, y):
    _low = y < 4.0
    _yt  = y[_low]
    _Xl  = X[_low]
    _cho_p = cho_model(_Xl)
    _svr_p = np.clip(
        svr_final.predict(sc_final.transform(_Xl.reshape(-1, 1))), 0, 100)

    _lim = max(_yt.max(), _cho_p.max(), _svr_p.max()) * 1.1

    _fig2, _ax = plt.subplots(figsize=(7, 6))
    _ax.plot([0, _lim], [0, _lim], "k--", lw=1.2, label="1:1 line", zorder=1)
    _ax.scatter(_yt, _cho_p, color="gray",    alpha=0.65, s=60, marker="s",
                label="Cho Model E", zorder=3)
    _ax.scatter(_yt, _svr_p, color="#1f77b4", alpha=0.65, s=60,
                label="SVR (C=1, ε=1)", zorder=4)

    _ax.set_xlabel("Observed % microbial reads", fontsize=12)
    _ax.set_ylabel("Predicted % microbial reads", fontsize=12)
    _ax.set_title("Figure 2 — Predicted vs observed, near-zero zone (<4%)\n"
                  "(in-sample fit on full dataset — illustrative)", fontsize=12)
    _ax.set_xlim(-0.1, _lim)
    _ax.set_ylim(-0.1, _lim)
    _ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(boot_tune, np, plt):
    _keys   = ["cho", "SVR_C1_e1", "SVR_C1_e2", "KNN_k5", "KNN_k7"]
    _labels = ["Cho\n(baseline)", "SVR\nC=1 ε=1", "SVR\nC=1 ε=2",
               "KNN\nk=5", "KNN\nk=7"]
    _colors = ["#d62728", "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]
    _vdata  = [np.array(boot_tune[k]["rmse_low"]) for k in _keys]

    _fig3, _ax = plt.subplots(figsize=(9, 5))
    _parts = _ax.violinplot(_vdata, positions=range(len(_keys)),
                            showmedians=True, showextrema=False)
    for _pc, _col in zip(_parts["bodies"], _colors):
        _pc.set_facecolor(_col)
        _pc.set_alpha(0.75)
    _parts["cmedians"].set_color("black")
    _parts["cmedians"].set_linewidth(2)

    _ax.axhline(np.mean(boot_tune["cho"]["rmse_low"]),
                color="#d62728", ls="--", lw=1.2, alpha=0.6,
                label=f"Cho mean ({np.mean(boot_tune['cho']['rmse_low']):.2f}%)")
    _ax.set_xticks(range(len(_labels)))
    _ax.set_xticklabels(_labels, fontsize=11)
    _ax.set_ylabel("OOB RMSE — near-zero zone (<4%)", fontsize=11)
    _ax.set_title("Figure 3 — Bootstrap RMSE distributions (200 iterations)", fontsize=13)
    _ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Multi-feature model — site + absolute Ct

    ΔCt alone discards absolute scale: at ΔCt ≈ 6, lung_BAL gives ~0.1% microbial
    while oropharyngeal gives ~6%. The ct_16S and ct_ACTB values carry this information.

    Feature sets tested (all SVR C=1, ε=1, γ=scale):
    - SVR_1d        : [delta]                      — current baseline
    - SVR_2d_dACTB  : [delta, ct_ACTB]             — adds host normalisation
    - SVR_2d_raw    : [ct_16S, ct_ACTB]            — raw Ct, mechanistically clean
    - SVR_3d_site   : [delta, ct_ACTB, site_ohe]   — adds explicit community context

    200 bootstrap iterations (SEED=77), same OOB protocol.
    """)
    return


@app.cell
def _(OneHotEncoder, X, data, np):
    _ct16S  = data["ct_16S"].to_numpy()
    _ctACTB = data["ct_ACTB"].to_numpy()

    X_2d_dACTB = np.column_stack([X, _ctACTB])        # [delta, ct_ACTB]
    X_2d_raw   = np.column_stack([_ct16S, _ctACTB])   # [ct_16S, ct_ACTB]

    # OHE fit on full dataset — site names are fixed metadata, not response-derived
    _ohe = OneHotEncoder(sparse_output=False, drop="first")
    _site_enc = _ohe.fit_transform(
        data["sample_type"].to_numpy().reshape(-1, 1))
    X_3d_site = np.column_stack([X, _ctACTB, _site_enc])

    _types = _ohe.categories_[0].tolist()
    print(f"Feature shapes:  1d={X.shape}  2d_dACTB={X_2d_dACTB.shape}"
          f"  2d_raw={X_2d_raw.shape}  3d_site={X_3d_site.shape}")
    print(f"Site OHE categories (drop first={_types[0]}): {_types[1:]}")
    return (X_2d_dACTB, X_2d_raw, X_3d_site)


@app.cell
def _(
    SVR,
    StandardScaler,
    X,
    X_2d_dACTB,
    X_2d_raw,
    X_3d_site,
    cho_model,
    mean_absolute_error,
    mean_squared_error,
    np,
    y,
):
    N_BOOT_MF = 200
    SEED_MF   = 77

    _feature_sets = {
        "SVR_1d":       X.reshape(-1, 1),
        "SVR_2d_dACTB": X_2d_dACTB,
        "SVR_2d_raw":   X_2d_raw,
        "SVR_3d_site":  X_3d_site,
    }

    boot_mf = {"cho": {"rmse_low": [], "mae_low": []}}
    boot_mf.update({_k: {"rmse_low": [], "mae_low": []} for _k in _feature_sets})
    boot_mf["n_oob_low"] = []

    _rng  = np.random.default_rng(SEED_MF)
    _n    = len(y)

    for _i in range(N_BOOT_MF):
        _ib  = _rng.integers(0, _n, size=_n)
        _oob = np.setdiff1d(np.arange(_n), _ib)
        if len(_oob) < 5:
            continue
        _yoob    = y[_oob]
        _oob_low = _yoob < 4.0
        if _oob_low.sum() < 3:
            continue

        # Cho — fixed model, no fitting needed
        _cho_pred = cho_model(X[_oob][_oob_low])
        boot_mf["cho"]["rmse_low"].append(
            float(np.sqrt(mean_squared_error(_yoob[_oob_low], _cho_pred))))
        boot_mf["cho"]["mae_low"].append(
            float(mean_absolute_error(_yoob[_oob_low], _cho_pred)))

        for _key, _Xall in _feature_sets.items():
            _sc      = StandardScaler()
            _Xib_sc  = _sc.fit_transform(_Xall[_ib])
            _Xoob_sc = _sc.transform(_Xall[_oob])
            _mdl     = SVR(kernel="rbf", C=1, epsilon=1, gamma="scale")
            _mdl.fit(_Xib_sc, y[_ib])
            _pred = np.clip(_mdl.predict(_Xoob_sc), 0., 100.)
            boot_mf[_key]["rmse_low"].append(
                float(np.sqrt(mean_squared_error(_yoob[_oob_low], _pred[_oob_low]))))
            boot_mf[_key]["mae_low"].append(
                float(mean_absolute_error(_yoob[_oob_low], _pred[_oob_low])))

        boot_mf["n_oob_low"].append(int(_oob_low.sum()))

    _valid = len(boot_mf["cho"]["rmse_low"])
    print(f"Multi-feature bootstrap: {_valid}/{N_BOOT_MF} valid iterations")
    print(f"Mean OOB near-zero: {np.mean(boot_mf['n_oob_low']):.1f} samples/iter")
    return (boot_mf,)


@app.cell
def _(boot_mf, np, stats):
    _order = ["cho", "SVR_1d", "SVR_2d_dACTB", "SVR_2d_raw", "SVR_3d_site"]
    _results = []
    for _k in _order:
        _a = np.array(boot_mf[_k]["rmse_low"])
        _results.append((_k, _a.mean(), _a.std(),
                         np.percentile(_a, 5), np.median(_a), np.percentile(_a, 95)))

    _cho_mean = np.mean(boot_mf["cho"]["rmse_low"])
    _1d_mean  = np.mean(boot_mf["SVR_1d"]["rmse_low"])
    _hdr = f"{'Config':<18} {'Mean':>8} {'SD':>7} {'p5':>8} {'Median':>8} {'p95':>8}"
    _sep = "─" * 70
    print("Near-zero RMSE (<4%) — multi-feature comparison\n")
    print(_hdr); print(_sep)
    for _r in _results:
        if _r[0] == "cho":
            _tag = " ← Cho baseline"
        elif _r[0] == "SVR_1d":
            _tag = f"  Δ_cho={_r[1]-_cho_mean:+.3f}"
        else:
            _tag = f"  Δ_cho={_r[1]-_cho_mean:+.3f}  Δ_1d={_r[1]-_1d_mean:+.3f}"
        print(f"{_r[0]:<18} {_r[1]:>8.4f} {_r[2]:>7.4f} {_r[3]:>8.4f} "
              f"{_r[4]:>8.4f} {_r[5]:>8.4f}{_tag}")

    # Wilcoxon vs Cho and vs SVR_1d for all multi-feature variants
    _cho_arr = np.array(boot_mf["cho"]["rmse_low"])
    _1d_arr  = np.array(boot_mf["SVR_1d"]["rmse_low"])
    print("\nWilcoxon tests — near-zero RMSE:")
    for _k in ["SVR_1d", "SVR_2d_dACTB", "SVR_2d_raw", "SVR_3d_site"]:
        _ml  = np.array(boot_mf[_k]["rmse_low"])
        _nw  = min(len(_cho_arr), len(_ml))
        _d1  = _cho_arr[:_nw] - _ml[:_nw]
        _, _p1 = stats.wilcoxon(_d1)
        if _k != "SVR_1d":
            _d2  = _1d_arr[:_nw] - _ml[:_nw]
            _, _p2 = stats.wilcoxon(_d2)
            _extra = f"  vs SVR_1d: ΔRMSE={_d2.mean():+.3f}% p={_p2:.2e} {'✓' if _p2<0.05 else '–'}"
        else:
            _extra = ""
        _sig1 = "✓" if _p1 < 0.05 else "–"
        print(f"  {_k:<18} vs Cho: ΔRMSE={_d1.mean():+.3f}% p={_p1:.2e} {_sig1}{_extra}")
    return


if __name__ == "__main__":
    app.run()
