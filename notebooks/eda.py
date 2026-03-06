import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import hvplot.polars   # noqa: F401
    import hvplot
    import holoviews as hv
    hvplot.extension("bokeh")
    return hv, mo, np, pl


@app.cell
def _(pl):
    _raw = pl.read_csv(
        "../data/data.csv",
        null_values=["NA", "", "None"],
        infer_schema_length=200,
    )

    df = _raw.with_columns([
        pl.col("ct_16S").cast(pl.Float64),
        pl.col("ct_ACTB").cast(pl.Float64),
        pl.col("delta").cast(pl.Float64),
        pl.col("pct_microbial").cast(pl.Float64),
        pl.col("reads_in").cast(pl.Int64),
        pl.col("reads_out").cast(pl.Int64),
        pl.col("qbit_1").cast(pl.Float64),
        pl.when(pl.col("pct_microbial") < 2).then(pl.lit("< 2% (BAL regime)"))
          .when(pl.col("pct_microbial") < 4).then(pl.lit("2–4% (Cho failure)"))
          .when(pl.col("pct_microbial") < 98).then(pl.lit("4–98% (Cho valid)"))
          .otherwise(pl.lit("> 98% (saturated)"))
          .alias("regime"),
    ])

    PALETTE = {
        "stool":          "#2166ac",
        "oropharyngeal":  "#e08214",
        "rectal_swab":    "#1b7837",
        "vaginal_sample": "#d6282a",
        "lung_bal":       "#7c3aed",
        "lung_sputum":    "#3a3a3a"
    }
    import functools, operator

    def scatter_by_type(data, x, y, palette, **kwargs):
        """hvplot scatter coloured by sample_type using PALETTE, without by=."""
        plots = [
            data.filter(pl.col("sample_type") == st).hvplot.scatter(
                x=x, y=y,
                color=palette.get(st, "grey"),
                label=st,
                **kwargs,
            )
            for st in data["sample_type"].unique().sort().to_list()
        ]
        return functools.reduce(operator.mul, plots)

    df.sample(3)
    return PALETTE, df, functools, operator, scatter_by_type


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CTomics — Exploratory Data Analysis
    ## Dataset overview before modelling

    | Figure | Question |
    |--------|----------|
    | **E1** | Sample composition: counts, missingness, sequencing depth |
    | **E2** | Target variable distribution: pct_microbial by sample type |
    | **E3** | Feature space: (ct_ACTB, ct_16S) plane |
    | **E4** | ΔCt vs pct_microbial with Cho model overlay |
    | **E5** | Patient / longitudinal structure and CV implications |
    | **E6** | Qubit coverage and relationship to Ct values |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## E1 — Sample composition and data completeness
    """)
    return


@app.cell
def _(df, pl):
    _counts = (
        df.group_by("sample_type")
          .agg(pl.len().alias("n"))
          .sort("n", descending=True)
    )

    _plot_e1a = _counts.hvplot.bar(
        x="sample_type", y="n",
        title="E1A — Samples per type",
        xlabel="", ylabel="Count",
        color="steelblue", rot=20,
        width=380, height=300,
    )

    _regime_counts = (
        df.group_by(["sample_type", "regime"])
          .agg(pl.len().alias("n"))
          .sort("sample_type")
    )

    _plot_e1b = _regime_counts.hvplot.bar(
        x="sample_type", y="n", by="regime",
        title="E1B — Regime breakdown per sample type",
        xlabel="", ylabel="Count",
        stacked=True, rot=20,
        width=420, height=300,
        legend="top_right",
    )

    _cols_check = ["ct_16S", "ct_ACTB", "delta", "pct_microbial",
                   "qbit_1", "run", "reads_in", "reads_out"]
    _miss_rows = []
    for _c in _cols_check:
        for _st in df["sample_type"].unique().to_list():
            _sub = df.filter(pl.col("sample_type") == _st)
            _miss_rows.append({
                "column":      _c,
                "sample_type": _st,
                "pct_missing": round(100 * _sub[_c].null_count() / len(_sub), 1),
            })

    _df_miss = pl.DataFrame(_miss_rows)

    _plot_e1c = _df_miss.hvplot.heatmap(
        x="sample_type", y="column", C="pct_missing",
        title="E1C — % missing per column × sample type",
        xlabel="", ylabel="",
        rot=20, width=460, height=300,
        cmap="Reds", colorbar=True,
    )

    (_plot_e1a + _plot_e1b + _plot_e1c).cols(3)
    return


@app.cell
def _(df, pl):
    df.group_by("sample_type").agg([
        pl.len().alias("n"),
        pl.col("pct_microbial").mean().round(2).alias("mean_pct"),
        pl.col("pct_microbial").median().round(2).alias("median_pct"),
        pl.col("pct_microbial").min().round(4).alias("min_pct"),
        pl.col("pct_microbial").max().round(2).alias("max_pct"),
        pl.col("ct_16S").mean().round(2).alias("mean_ct16S"),
        pl.col("ct_ACTB").mean().round(2).alias("mean_ctACTB"),
        pl.col("qbit_1").null_count().alias("n_missing_qubit"),
    ]).sort("mean_pct")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## E2 — Target variable: pct_microbial
    """)
    return


@app.cell
def _(df, hv, np, pl):
    _plot_e2a = df.hvplot.hist(
        y="pct_microbial", by="sample_type",
        bins=40,
        title="E2A — Distribution of pct_microbial (linear)",
        xlabel="% microbial reads", ylabel="Count",
        alpha=0.6, width=460, height=320,
        legend="top_right",
    )

    _log_bins = np.logspace(np.log10(0.01), 2, 50).tolist()
    _plot_e2b = df.filter(pl.col("pct_microbial") >= 0.01).hvplot.hist(
        y="pct_microbial", by="sample_type",
        bins=_log_bins,
        logx=True,
        title="E2B — Distribution of pct_microbial (log x)",
        xlabel="% microbial reads (log)", ylabel="Count",
        alpha=0.6, width=460, height=320,
        legend=False,
        xticks=[(0.01, "0.01"), (0.1, "0.1"), (1, "1"),
                (10, "10"), (100, "100")],
    )

    _ecdf_rows = []
    for _st in df["sample_type"].unique().sort().to_list():
        _vals = (
            df.filter(pl.col("sample_type") == _st)
              ["pct_microbial"].drop_nulls().sort().to_numpy()
        )
        _n = len(_vals)
        for _i, _v in enumerate(_vals):
            _ecdf_rows.append({
                "pct_microbial": float(_v),
                "ecdf":          (_i + 1) / _n,
                "sample_type":   _st,
            })

    _df_ecdf = pl.DataFrame(_ecdf_rows).filter(pl.col("pct_microbial") >= 0.01)

    _plot_e2c = (
        _df_ecdf.hvplot.line(
            x="pct_microbial", y="ecdf", by="sample_type",
            logx=True,
            title="E2C — ECDF of pct_microbial",
            xlabel="% microbial reads (log)", ylabel="Cumulative fraction",
            width=460, height=320,
            legend="bottom",
            xticks=[(0.01, "0.01"), (0.1, "0.1"), (1, "1"),
                    (10, "10"), (100, "100")],
        )
        * hv.VLine(4).opts(color="red", line_dash="dashed", line_width=1.5)
        * hv.VLine(2).opts(color="orange", line_dash="dashed", line_width=1.5)
    )

    (_plot_e2a + _plot_e2b + _plot_e2c).cols(3)
    return


@app.cell
def _(mo):
    mo.md("""
    > Red = 4% Cho threshold.  Orange = 2% practical BAL ceiling.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## E3 — Feature space: the (ct_ACTB, ct_16S) plane
    """)
    return


@app.cell
def _(PALETTE, df, hv, np, pl, scatter_by_type):
    _ct16s_grid = np.linspace(8, 45, 200)
    _iso_rows = []
    for _dv in [0, 5, 10, 15, 20]:
        for _v in _ct16s_grid:
            _iso_rows.append({"ct_16S": float(_v),
                              "ct_ACTB": float(_v + _dv),
                              "label": f"ΔCt={_dv}"})
    _df_iso = pl.DataFrame(_iso_rows).filter(
        (pl.col("ct_ACTB") >= 10) & (pl.col("ct_ACTB") <= 45)
    )
    _iso_lines = _df_iso.hvplot.line(
        x="ct_16S", y="ct_ACTB", by="label",
        color="lightgrey", line_width=0.8,
        xlim=(8, 45), ylim=(10, 45),
    )

    _detect = hv.VLine(33).opts(color="red", line_dash="dashed",
                                line_width=1.5, alpha=0.6)

    _scatter_type = scatter_by_type(
        df, x="ct_16S", y="ct_ACTB", palette=PALETTE,
        title="E3A — (ct_16S, ct_ACTB) coloured by sample type",
        xlabel="ct_16S", ylabel="ct_ACTB",
        alpha=0.75, size=60, width=480, height=420,
    )

    (_iso_lines * _scatter_type * _detect)
    return


@app.cell
def _(df, hv, np, pl):
    _ct16s_grid2 = np.linspace(8, 45, 200)
    _iso_rows2 = []
    for _dv2 in [0, 5, 10, 15, 20]:
        for _v2 in _ct16s_grid2:
            _iso_rows2.append({"ct_16S": float(_v2),
                               "ct_ACTB": float(_v2 + _dv2),
                               "label": f"ΔCt={_dv2}"})
    _df_iso2 = pl.DataFrame(_iso_rows2).filter(
        (pl.col("ct_ACTB") >= 10) & (pl.col("ct_ACTB") <= 45)
    )
    _iso_lines2 = _df_iso2.hvplot.line(
        x="ct_16S", y="ct_ACTB", by="label",
        color="lightgrey", line_width=0.8,
    )

    _df_e3b = df.filter(pl.col("pct_microbial").is_not_null())

    _scatter_pct = _df_e3b.hvplot.scatter(
        x="ct_16S", y="ct_ACTB",
        c="pct_microbial",
        cmap="viridis", logz=True,
        title="E3B — (ct_16S, ct_ACTB) coloured by pct_microbial (log)",
        xlabel="ct_16S", ylabel="ct_ACTB",
        alpha=0.85, size=60,
        width=480, height=420,
        colorbar=True, clabel="% microbial (log)",
    )

    _detect2 = hv.VLine(33).opts(color="red", line_dash="dashed",
                                  line_width=1.5, alpha=0.6)

    (_iso_lines2 * _scatter_pct * _detect2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Grey diagonals = iso-ΔCt lines — identical ΔCt-only model prediction.
    > E3B shows the pct_microbial gradient cuts **across** these lines at low values:
    > same ΔCt, very different true composition. Red dashed = ct_16S ≥ 33 (detection limit zone).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## E4 — ΔCt vs pct_microbial and the Cho model
    """)
    return


@app.cell
def _(PALETTE, df, functools, hv, np, operator, pl):
    def _cho(d):
        return np.clip(
            2.7201549 / (99.50267 * np.exp(-0.7218 * d) + 0.02733), 0, 100
        )

    _delta_c = np.linspace(-15, 35, 600)
    _df_cho_lin = pl.DataFrame({
        "delta": _delta_c,
        "pct_microbial": _cho(_delta_c),
    })

    _df_valid = df.filter(pl.col("pct_microbial").is_not_null())

    def _scatter_overlay(data, x, y, palette, **kwargs):
        plots = [
            data.filter(pl.col("sample_type") == st).hvplot.scatter(
                x=x, y=y, color=palette.get(st, "grey"), label=st, **kwargs
            )
            for st in data["sample_type"].unique().sort().to_list()
        ]
        return functools.reduce(operator.mul, plots)

    _plot_e4a = (
        _scatter_overlay(
            _df_valid, x="delta", y="pct_microbial", palette=PALETTE,
            title="E4A — ΔCt vs pct_microbial (linear)",
            xlabel="ΔCt (ct_ACTB − ct_16S)",
            ylabel="% microbial reads",
            ylim = (0.001,100),
            alpha=0.7, size=60, width=500, height=380,
        )
        * _df_cho_lin.hvplot.line(
            x="delta", y="pct_microbial",
            color="black", line_dash="dashed", line_width=2,
        )
        * hv.HLine(4).opts(color="red", line_dash="dashed",
                           line_width=1.2, alpha=0.5)
    )

    _df_valid_log = _df_valid.filter(pl.col("pct_microbial") >= 0.01)
    _df_cho_log = _df_cho_lin.filter(pl.col("pct_microbial") >= 0.01)

    _plot_e4b = (
        _scatter_overlay(
            _df_valid_log, x="delta", y="pct_microbial", palette=PALETTE,
            logy=True,
            title="E4B — ΔCt vs pct_microbial (log y — BAL regime)",
            xlabel="ΔCt (ct_ACTB − ct_16S)",
            ylabel="% microbial reads",
            alpha=0.7, size=60, width=500, height=380,
            yticks=[(0.01, "0.01"), (0.1, "0.1"), (1, "1"),
                    (10, "10"), (100, "100")],
        ).opts(show_legend=False)
        * _df_cho_log.hvplot.line(
            x="delta", y="pct_microbial",
            color="black", line_dash="dashed", line_width=2,
        )
        * hv.HLine(4).opts(color="red", line_dash="dashed",
                            line_width=1.2, alpha=0.5)
    )

    (_plot_e4a + _plot_e4b).cols(2)
    return


@app.cell
def _(PALETTE, df, functools, np, operator, pl):
    def _cho2(d):
        return np.clip(
            2.7201549 / (99.50267 * np.exp(-0.7218 * d) + 0.02733), 0, 100
        )

    _df_resid = (
        df.filter(
            pl.col("pct_microbial").is_not_null() &
            pl.col("delta").is_not_null()
        )
        .with_columns([
            pl.col("delta").map_elements(
                lambda d: float(_cho2(d)), return_dtype=pl.Float64
            ).alias("cho_pred"),
        ])
        .with_columns([
            (pl.col("pct_microbial") - pl.col("cho_pred")).alias("residual"),
            (pl.col("pct_microbial") - pl.col("cho_pred")).abs().alias("abs_residual"),
        ])
    )

    def _scatter_overlay(data, x, y, palette, **kwargs):
        plots = [
            data.filter(pl.col("sample_type") == st).hvplot.scatter(
                x=x, y=y, color=palette.get(st, "grey"), label=st, **kwargs
            )
            for st in data["sample_type"].unique().sort().to_list()
        ]
        return functools.reduce(operator.mul, plots)

    _plot_e4c = _scatter_overlay(
        _df_resid, x="cho_pred", y="residual", palette=PALETTE,
        title="E4C — Cho residuals vs predicted",
        xlabel="Cho predicted % microbial",
        ylabel="Residual (obs − pred)",
        alpha=0.7, size=60, width=480, height=340,
    )

    _df_resid_log = _df_resid.filter(pl.col("pct_microbial") >= 0.01)

    _plot_e4d = _scatter_overlay(
        _df_resid_log, x="pct_microbial", y="abs_residual", palette=PALETTE,
        logx=True,
        title="E4D — |residual| vs true pct_microbial",
        xlabel="True % microbial (log)",
        ylabel="|obs − pred|",
        alpha=0.7, size=60, width=480, height=340,
        xticks=[(0.01, "0.01"), (0.1, "0.1"), (1, "1"),
                (10, "10"), (100, "100")],
    ).opts(show_legend=False)

    (_plot_e4c + _plot_e4d).cols(2)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## E5 — Patient structure and CV implications
    """)
    return


@app.cell
def _(df, pl):
    _df_pat = (
        df.filter(pl.col("patient").is_not_null())
          .group_by(["patient", "sample_type"])
          .agg(pl.len().alias("n"))
          .sort(["patient", "sample_type"])
    )

    _df_pat.hvplot.bar(
        x="patient", y="n", by="sample_type",
        stacked=True,
        title="E5A — Samples per patient (stacked by sample type)",
        xlabel="Patient ID", ylabel="Number of samples",
        rot=90, width=820, height=320,
        legend="top_right",
    )
    return


@app.cell
def _(df, pl):
    _df_within = (
        df.filter(pl.col("sample_type") == "lung_bal")
          .group_by("patient")
          .agg([
              pl.col("pct_microbial").min().alias("min_pct"),
              pl.col("pct_microbial").max().alias("max_pct"),
              pl.col("pct_microbial").mean().alias("mean_pct"),
              pl.col("pct_microbial").std().alias("sd_pct"),
              pl.len().alias("n"),
          ])
          .with_columns(
              (pl.col("max_pct") / pl.col("min_pct")).alias("fold_range")
          )
          .sort("mean_pct")
    )

    _plot_e5b_dot = _df_within.hvplot.scatter(
        x="patient", y="mean_pct",
        title=("E5B — BAL: within-patient pct_microbial range\n"
               "dot = mean,  bar = min–max"),
        xlabel="Patient", ylabel="% microbial reads",
        size=80, color="#7c3aed", alpha=0.85,
        width=500, height=320,
    )

    _plot_e5b_err = _df_within.hvplot.errorbars(
        x="patient", y="mean_pct",
        yerr1="sd_pct", yerr2="sd_pct",
        color="#7c3aed", alpha=0.5, line_width=2,
    )

    (_plot_e5b_dot * _plot_e5b_err)
    return


@app.cell
def _(df, pl):
    _df_cv = (
        df.filter(
            pl.col("pct_microbial").is_not_null() &
            pl.col("patient").is_not_null()
        )
        .group_by("patient")
        .agg([
            pl.col("pct_microbial").std().alias("within_sd"),
            pl.col("pct_microbial").mean().alias("patient_mean"),
            pl.col("sample_type").first().alias("sample_type"),
            pl.len().alias("n"),
        ])
        .filter(pl.col("n") > 1)
        .sort("patient_mean")
    )

    _df_cv.hvplot.scatter(
        x="patient_mean", y="within_sd",
        by="sample_type",
        logx=True,
        title=("E5C — Within-patient SD vs patient mean pct_microbial\n"
               "Low within-patient SD → LOOCV adequate; high → LOPOCV needed"),
        xlabel="Patient mean % microbial (log)",
        ylabel="Within-patient SD (%)",
        alpha=0.8, size=80,
        width=540, height=340, legend="top_left",
        xticks=[(0.01, "0.01"), (0.1, "0.1"), (1, "1"),
                (10, "10"), (100, "100")],
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## E6 — Qubit: coverage and relationship to Ct values
    """)
    return


@app.cell
def _(df, pl):
    _qbit_avail = (
        df.with_columns(
            pl.col("qbit_1").is_not_null().cast(pl.Utf8).alias("has_qubit")
        )
        .group_by(["sample_type", "has_qubit"])
        .agg(pl.len().alias("n"))
        .sort(["sample_type", "has_qubit"])
    )

    _plot_e6a = _qbit_avail.hvplot.bar(
        x="sample_type", y="n", by="has_qubit",
        title="E6A — Qubit availability by sample type",
        xlabel="", ylabel="Count",
        stacked=True, rot=20,
        width=360, height=300, legend="top_right",
    )

    _df_q = df.filter(pl.col("qbit_1").is_not_null())

    _plot_e6b = _df_q.hvplot.scatter(
        x="qbit_1", y="ct_ACTB", by="sample_type",
        logx=True,
        title="E6B — Qubit vs ct_ACTB\n(expected: negative correlation)",
        xlabel="Qubit (ng/µL, log)", ylabel="ct_ACTB",
        alpha=0.75, size=70,
        width=400, height=300, legend="top_right",
    )

    _plot_e6c = _df_q.hvplot.scatter(
        x="qbit_1", y="ct_16S", by="sample_type",
        logx=True,
        title="E6C — Qubit vs ct_16S\n(decoupled from human DNA)",
        xlabel="Qubit (ng/µL, log)", ylabel="ct_16S",
        alpha=0.75, size=70,
        width=400, height=300, legend=False,
    )

    (_plot_e6a + _plot_e6b + _plot_e6c).cols(3)
    return


@app.cell
def _(df, pl):
    _df_q2 = df.filter(
        pl.col("qbit_1").is_not_null() &
        pl.col("pct_microbial").is_not_null() &
        (pl.col("pct_microbial") >= 0.01)
    )

    _df_q2.hvplot.scatter(
        x="qbit_1", y="pct_microbial",
        by="sample_type",
        logx=True, logy=True,
        title=("E6D — Qubit vs pct_microbial\n"
               "No correlation expected if ΔCt cancels load; "
               "residual correlation → include Qubit as feature"),
        xlabel="Qubit (ng/µL, log)", ylabel="% microbial (log)",
        alpha=0.75, size=70,
        width=540, height=360, legend="bottom_right",
        yticks=[(0.01, "0.01"), (0.1, "0.1"), (1, "1"),
                (10, "10"), (100, "100")],
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Modelling implications — summary

    | Observation | Figure | Modelling implication |
    |---|---|---|
    | lung_bal entirely below 2%, fully separated | E1B, E2C | Stratified evaluation; BAL is OOD relative to Cho training |
    | BAL clusters in high ct_16S region (>33) absent from other types | E3A | Use ct_ACTB + ct_16S separately; include sample_type |
    | Iso-ΔCt lines cut across pct_microbial gradient at low values | E3B | ΔCt alone insufficient; 2D feature space required |
    | Cho residuals large and structured at pct < 4% | E4C, E4D | Cho is the baseline; beat it specifically in BAL regime |
    | Within-patient BAL SD small relative to between-patient range | E5B, E5C | Standard LOOCV defensible; LOPOCV as sensitivity check |
    | Qubit available only for BAL cohort | E6A | Ablation study: model with vs without Qubit |
    | Qubit × ct_ACTB correlation within BAL | E6B | If strong: Qubit captures extraction efficiency variation |
    """)
    return


if __name__ == "__main__":
    app.run()
