import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


# ============================================================
# HOW TO USE
# ============================================================
# marimo edit cho_svr_bootstrap.py
#
# Marimo looks like Jupyter with one structural rule:
#   Variables flow from cell A to cell B by being
#   (a) returned from cell A, and
#   (b) listed as parameters of cell B.
# Run cells top-to-bottom exactly like a Jupyter notebook.
# ============================================================


# ── CELL 1 ── imports ────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy import stats
    import warnings
    warnings.filterwarnings("ignore")
    return mo, np, pl, SVR, StandardScaler, mean_squared_error, mean_absolute_error, r2_score, stats


# ── CELL 2 ── title ──────────────────────────────────────────
@app.cell
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


# ── CELL 3 ── The Cho Model E equation ──────────────────────
#
# Cho et al. 2021 Model E (mSystems, doi:10.1128/msystems.00552-21)
#
#   ŷ = 2.7201549 / ( 99.50267 · exp(−0.7218 · ΔCt) + 0.02733 )
#
# The constant +0.02733 in the denominator creates the hard lower
# asymptote.  Let us work out the direction:
#
#   Large positive ΔCt  (ACTB >> 16S, lots of human DNA, few bacteria)
#     → exp(−0.7218 × large+) → 0
#     → denominator → 0.02733
#     → ŷ → 2.7201549 / 0.02733 ≈ 99.5 %   [upper ceiling]
#
#   Large negative ΔCt  (16S >> ACTB, lots of bacterial DNA)
#     → exp(−0.7218 × large−) = exp(large+) → ∞
#     → denominator → ∞
#     → ŷ → 0 %                              [lower floor]
#
# So the floor IS near zero — but at ΔCt = −10 it is already at
# ~0.027 % and all samples with ΔCt < −5 are compressed into an
# indistinguishably narrow band.  That is the resolution problem.
#
# For BAL/COPD samples:
#   ACTB Ct is low (abundant human lung/immune cells)
#   16S Ct is high (sparse bacteria)
#   → ΔCt = ACTB − 16S is likely NEGATIVE or small positive
#   → predictions land near the floor and cannot be distinguished
# ─────────────────────────────────────────────────────────────
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
    for d, p in zip(test_d, cho_model(test_d)):
        print(f"  {d:>4}     {p:>7.4f}")

    print(f"\nFloor (ΔCt = −100) : {cho_model(np.array([-100.]))[0]:.6f} %")
    print(f"Ceiling (ΔCt = +100): {cho_model(np.array([+100.]))[0]:.4f} %")
    print()
    print("Practical problem: at ΔCt = −10 the model is already at ~0.027 %.")
    print("Every BAL sample with ΔCt < −5 gets the same compressed prediction.")
    return cho_model


# ── CELL 4 ── dataset ────────────────────────────────────────
#
# qPCR Ct values are from Cho et al. Table S1.
# Ground-truth % microbial is SIMULATED as Cho_prediction + noise
# (SD = 4.35 %, matching the reported training-set residual SD).
#
# *** Replace pct_microbial with real sequencing values ***
# Cho sequencing data: SRA accession PRJNA718445
# ─────────────────────────────────────────────────────────────
@app.cell
def _(np, pl, cho_model):
    rng = np.random.default_rng(seed=42)

    # ── Cho paper samples ──────────────────────────────────
    s16 = np.array([
        17.150,13.208,13.772,15.664,13.859,14.071,17.716,19.196,
        15.276,15.070,16.950,14.181,15.744,           # rectal (13)
        15.288,14.883,13.352,16.097,14.188,16.705,22.317,  # vaginal (7)
        17.795,15.953,16.624,16.193,15.171,14.976,
        17.762,13.673,15.140,14.019,13.479,18.192,    # stool (12)
        22.074,22.219,21.962,19.334,24.129,26.864,19.412,18.957,
        20.471,25.027,25.835,16.629,24.245,24.735,    # oropharyngeal (14)
    ])
    actb = np.array([
        24.881,27.498,29.418,24.227,26.632,29.838,27.141,26.005,
        27.362,26.048,24.876,28.721,30.329,
        29.487,34.364,23.588,25.045,24.816,24.585,23.727,
        37.971,34.286,35.094,33.827,37.238,36.013,38.448,34.234,
        32.067,34.945,33.873,38.773,
        24.316,34.049,30.278,30.600,22.908,29.193,27.238,30.068,
        26.683,29.333,30.654,21.802,26.367,21.284,
    ])
    s_types = (["Rectal swab"] * 13 + ["Vaginal"] * 7 +
               ["Stool"] * 12    + ["Oropharyngeal"] * 14)
    s_ids = (
        ["AH-PS-001","AH-PS-003","AH-PS-004","AH-PS-006","AH-PS-011",
         "AH-PS-016","AH-PS-018","AH-PS-020","AH-PS-021","AH-PS-031",
         "AH-PS-039","AH-PS-044","AH-PS-051"] +
        ["HSV-11","HSV-16","HSV-24","HSV-29","HSV-31","HSV-32","HSV-8"] +
        ["S01","S04","S05","S07","S08","S11","S13","S14",
         "S15","S17","S18","S19"] +
        ["S49","S50","S51","S52","S53","S54","S55","S56",
         "S57","S58","S70","S71","S83","S92"]
    )

    cho_delta = actb - s16
    # SIMULATED ground truth — replace with real sequencing data
    cho_truth = np.clip(
        cho_model(cho_delta) + rng.normal(0, 4.35, len(cho_delta)),
        0.01, 99.99
    )

    cho_df = pl.DataFrame({
        "sample_id":     s_ids,
        "sample_type":   s_types,
        "delta_ct":      cho_delta,
        "pct_microbial": cho_truth,
        "dataset":       ["cho"] * len(s_ids),
    })

    # ── BAL COPD samples ────────────────────────────────────
    # Typical BAL: ACTB Ct ~22-30, 16S Ct ~28-38  →  ΔCt negative
    # REPLACE bal_delta and bal_truth with your real data
    n_bal     = 70
    bal_delta = rng.uniform(-15, 3, size=n_bal)
    bal_truth = np.clip(
        0.5 + 3.0 * np.exp(0.3 * bal_delta) + rng.normal(0, 0.5, n_bal),
        0.01, 15.0
    )

    bal_df = pl.DataFrame({
        "sample_id":     [f"BAL-{i:03d}" for i in range(n_bal)],
        "sample_type":   ["BAL_COPD"] * n_bal,
        "delta_ct":      bal_delta,
        "pct_microbial": bal_truth,
        "dataset":       ["bal"] * n_bal,
    })

    data = pl.concat([cho_df, bal_df])
    print(f"Combined dataset: {data.shape[0]} samples")
    print()
    print(data.group_by(["dataset","sample_type"]).len().sort(["dataset","sample_type"]))
    print()
    n_low = (data["pct_microbial"] < 4.0).sum()
    print(f"Samples < 4 % microbial (Cho failure zone): {n_low} / {data.shape[0]}")
    print(f"ΔCt range:  {data['delta_ct'].min():.2f}  to  {data['delta_ct'].max():.2f}")
    return data


# ── CELL 5 ── extract numpy arrays ──────────────────────────
@app.cell
def _(data, np):
    X        = data["delta_ct"].to_numpy()
    y        = data["pct_microbial"].to_numpy()
    low_mask = y < 4.0
    print(f"X (ΔCt):          shape={X.shape}   range=[{X.min():.2f}, {X.max():.2f}]")
    print(f"y (% microbial):  shape={y.shape}   range=[{y.min():.2f}, {y.max():.2f}]")
    print(f"Near-zero (<4%):  {low_mask.sum()} samples")
    return X, y, low_mask


# ── CELL 6 ── Cho baseline evaluation ───────────────────────
@app.cell
def _(mo):
    mo.md("""
    ## Cho model — baseline

    Fixed parameters, no refit.  Applying the published equation to our
    combined dataset gives an honest cross-sample-type baseline because
    the BAL samples are completely out-of-distribution for Cho et al.
    """)
    return


@app.cell
def _(X, y, low_mask, cho_model, np,
      mean_squared_error, mean_absolute_error, r2_score):

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
    return cho_pred_full, cho_pred_low, print_metrics


# ── CELL 7 ── SVR hyperparameters explained ─────────────────
@app.cell
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


# ── CELL 8 ── single SVR fit (pedagogical) ──────────────────
#
# We fit one SVR on the full dataset to inspect behaviour
# BEFORE automating it in the bootstrap loop.
# This is in-sample and therefore optimistic — do not report
# these numbers; use bootstrap OOB metrics instead.
# ─────────────────────────────────────────────────────────────
@app.cell
def _(X, y, cho_model, SVR, StandardScaler, np,
      mean_squared_error, mean_absolute_error, r2_score):

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
    return sc_demo, svr_demo, delta_grid, svr_curve, cho_curve


# ── CELL 9 ── bootstrap rationale ───────────────────────────
@app.cell
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


# ── CELL 10 ── bootstrap loop (100 iterations) ───────────────
@app.cell
def _(X, y, low_mask, cho_model,
      SVR, StandardScaler, np,
      mean_squared_error, mean_absolute_error, r2_score):

    # ── hyperparameters — change these to explore ─────────
    N_BOOT  = 100
    C       = 10.0
    EPSILON = 2.0
    GAMMA   = "scale"
    SEED    = 42
    # ──────────────────────────────────────────────────────

    rng_b = np.random.default_rng(SEED)
    n     = len(X)

    # One list per model × metric × zone
    boot = {
        "cho_rmse": [], "cho_mae": [], "cho_r2": [], "cho_bias": [],
        "svr_rmse": [], "svr_mae": [], "svr_r2": [], "svr_bias": [],
        "cho_rmse_low": [], "cho_mae_low": [],
        "svr_rmse_low": [], "svr_mae_low": [],
        "n_oob": [], "n_oob_low": [],
    }

    def _met(yt, yp):
        """Return (rmse, mae, r2, bias) as floats."""
        return (
            float(np.sqrt(mean_squared_error(yt, yp))),
            float(mean_absolute_error(yt, yp)),
            float(r2_score(yt, yp)),
            float(np.mean(yp - yt)),
        )

    for _i in range(N_BOOT):

        # ── in-bag / OOB split ──────────────────────────
        ib  = rng_b.integers(0, n, size=n)      # draw with replacement
        oob = np.setdiff1d(np.arange(n), ib)    # never-drawn samples
        if len(oob) < 5:
            continue

        Xib, yib    = X[ib],   y[ib]
        Xoob, yoob  = X[oob],  y[oob]
        oob_low     = yoob < 4.0                 # near-zero mask within OOB

        # ── Cho on OOB — no fitting, just evaluate ──────
        cho_oob             = cho_model(Xoob)
        rm, ma, r2, bi      = _met(yoob, cho_oob)
        boot["cho_rmse"].append(rm);   boot["cho_mae"].append(ma)
        boot["cho_r2"].append(r2);     boot["cho_bias"].append(bi)
        if oob_low.sum() >= 3:
            rm_l, ma_l, _, _ = _met(yoob[oob_low], cho_oob[oob_low])
            boot["cho_rmse_low"].append(rm_l)
            boot["cho_mae_low"].append(ma_l)

        # ── SVR on OOB — scale on in-bag only ───────────
        sc      = StandardScaler()
        Xib_sc  = sc.fit_transform(Xib.reshape(-1, 1))   # fit on training
        Xoob_sc = sc.transform(Xoob.reshape(-1, 1))       # apply to OOB

        svr = SVR(kernel="rbf", C=C, epsilon=EPSILON, gamma=GAMMA)
        svr.fit(Xib_sc, yib)
        svr_oob             = np.clip(svr.predict(Xoob_sc), 0., 100.)
        rm, ma, r2, bi      = _met(yoob, svr_oob)
        boot["svr_rmse"].append(rm);   boot["svr_mae"].append(ma)
        boot["svr_r2"].append(r2);     boot["svr_bias"].append(bi)
        if oob_low.sum() >= 3:
            rm_l, ma_l, _, _ = _met(yoob[oob_low], svr_oob[oob_low])
            boot["svr_rmse_low"].append(rm_l)
            boot["svr_mae_low"].append(ma_l)

        boot["n_oob"].append(len(oob))
        boot["n_oob_low"].append(int(oob_low.sum()))

    valid = len(boot["cho_rmse"])
    print(f"Bootstrap complete: {valid}/{N_BOOT} valid iterations")
    print(f"Mean OOB size      : {np.mean(boot['n_oob']):.1f} samples/iter")
    print(f"Mean OOB near-zero : {np.mean(boot['n_oob_low']):.1f} samples/iter")
    return boot, N_BOOT, C, EPSILON, GAMMA


# ── CELL 11 ── summary table ─────────────────────────────────
@app.cell
def _(boot, np):
    def _row(key, label):
        a = np.array(boot[key])
        return (label, a.mean(), a.std(),
                np.percentile(a, 5), np.median(a), np.percentile(a, 95))

    hdr = f"{'Metric':<26}  {'Mean':>8} {'SD':>7} {'p5':>8} {'Median':>8} {'p95':>8}"
    sep = "─" * 70

    print("Full dataset (all OOB samples)")
    print(hdr); print(sep)
    for lbl, key in [
        ("Cho  RMSE", "cho_rmse"), ("SVR  RMSE", "svr_rmse"),
        ("Cho  MAE",  "cho_mae"),  ("SVR  MAE",  "svr_mae"),
        ("Cho  R²",   "cho_r2"),   ("SVR  R²",   "svr_r2"),
        ("Cho  Bias", "cho_bias"), ("SVR  Bias", "svr_bias"),
    ]:
        r = _row(key, lbl)
        print(f"{r[0]:<26}  {r[1]:>8.4f} {r[2]:>7.4f} {r[3]:>8.4f} {r[4]:>8.4f} {r[5]:>8.4f}")

    print()
    print("Near-zero zone  (<4% microbial — Cho failure zone)")
    print(hdr); print(sep)
    for lbl, key in [
        ("Cho  RMSE", "cho_rmse_low"), ("SVR  RMSE", "svr_rmse_low"),
        ("Cho  MAE",  "cho_mae_low"),  ("SVR  MAE",  "svr_mae_low"),
    ]:
        r = _row(key, lbl)
        print(f"{r[0]:<26}  {r[1]:>8.4f} {r[2]:>7.4f} {r[3]:>8.4f} {r[4]:>8.4f} {r[5]:>8.4f}")
    return


# ── CELL 12 ── paired statistical test ──────────────────────
@app.cell
def _(mo):
    mo.md("""
    ## Statistical comparison

    Both models are evaluated on the **same OOB set** each iteration,
    so we compute per-iteration **ΔRMSE = Cho_RMSE − SVR_RMSE**.
    Positive = SVR has lower error = SVR is better.

    We use the **Wilcoxon signed-rank test** (paired, non-parametric).
    H₀: median ΔRMSE = 0 (models are equivalent).

    The bootstrapped 95% CI shows the practical magnitude of the
    improvement — important for the paper, not just the p-value.
    """)
    return


@app.cell
def _(boot, np, stats):
    for zone, ck, sk in [
        ("Full dataset",    "cho_rmse",     "svr_rmse"),
        ("Near-zero <4%",   "cho_rmse_low", "svr_rmse_low"),
    ]:
        n_   = min(len(boot[ck]), len(boot[sk]))
        diff = np.array(boot[ck][:n_]) - np.array(boot[sk][:n_])
        _, p = stats.wilcoxon(diff)
        ci   = np.percentile(diff, [2.5, 97.5])

        print(f"── {zone}")
        print(f"  Mean ΔRMSE (Cho−SVR)  : {diff.mean():+.4f}%")
        print(f"  SD                    : {diff.std():.4f}%")
        print(f"  95% bootstrap CI      : [{ci[0]:+.4f},  {ci[1]:+.4f}]")
        print(f"  SVR wins              : {100*np.mean(diff>0):.0f}% of iterations")
        print(f"  Wilcoxon p            : {p:.3e}")
        if p < 0.05:
            winner = "SVR" if diff.mean() > 0 else "Cho"
            print(f"  → {winner} significantly better (p < 0.05)")
        else:
            print(f"  → No significant difference (p ≥ 0.05)")
        print()
    return


# ── CELL 13 ── bias analysis ─────────────────────────────────
@app.cell
def _(boot, np):
    cho_bias = np.array(boot["cho_bias"])
    svr_bias = np.array(boot["svr_bias"])
    print("Bias (positive = systematic over-prediction, full OOB set)")
    print(f"  Cho:  {cho_bias.mean():+.4f}% ± {cho_bias.std():.4f}")
    print(f"  SVR:  {svr_bias.mean():+.4f}% ± {svr_bias.std():.4f}")
    print()
    print("A positive Cho bias confirms the asymptote floor drags predictions")
    print("upward for near-zero samples that land in the OOB set.")
    print()
    print("Reminder: these numbers are based on SIMULATED ground truth.")
    print("Replace cho_truth and bal_truth in cell 4 with real sequencing data.")
    return


# ── CELL 14 ── next steps ────────────────────────────────────
@app.cell
def _(mo):
    mo.md("""
    ## What to do next

    **1 — Replace simulated data with real values (cell 4)**
    - Cho ground truth: download from SRA **PRJNA718445**, run KneadData,
      compute pre/post-filter read counts → % microbial per sample.
    - BAL ground truth: use your own shotgun sequencing output.

    **2 — Add Gaussian Process Regression to the bootstrap loop**
    ```python
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    kernel = RBF() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    ```
    GPR gives calibrated prediction intervals out of the box —
    useful clinically and differentiates your paper from Cho.

    **3 — Hyperparameter grid search**
    Wrap cell 10 with a grid over `C ∈ {1, 10, 100}` and
    `epsilon ∈ {1, 2, 5}`.  Pick the combination with the lowest
    **mean OOB RMSE in the <4% zone** (not the full range).

    **4 — Add 18S as a second feature**
    Change `X = data["delta_ct"].to_numpy()` to a two-column matrix:
    ```python
    X = data.select(["delta_ct", "s18_ct"]).to_numpy()
    ```
    SVR and GPR handle multi-feature input natively.

    **5 — Figures for the methods paper**
    - Predicted vs observed scatter for the <4% zone, both models
      overlaid, Cho asymptote floor as a dashed horizontal line
    - Bootstrap RMSE distributions (violin), full range and <4% side by side
    - Cho model curve with failure zones shaded, your BAL samples overlaid
    """)
    return


if __name__ == "__main__":
    app.run()
