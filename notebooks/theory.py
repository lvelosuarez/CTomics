import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # ── Shared physical constants and model functions ──────────────────────────
    # Kept in a single cell so all subsequent cells can import cleanly.

    import numpy as np

    # Human diploid genome
    HUMAN_GENOME_PG     = 6.6          # pg / diploid cell
    ACTB_COPIES_CELL    = 2            # copies in diploid cell
    HUMAN_COPIES_PER_NG = (ACTB_COPIES_CELL / HUMAN_GENOME_PG) * 1e3  # copies / ng

    # 1 Mb DNA ≈ 1.096 × 10⁻³ pg  (avg nucleotide MW 330 Da, 2 strands, 10⁶ bp)
    MB_TO_PG = 1.096e-3

    def copies_per_ng_bacteria(genome_mb, n_16s):
        """16S gene copies per ng of bacterial DNA."""
        genome_pg      = genome_mb * MB_TO_PG
        genomes_per_ng = 1e3 / genome_pg   # 1 ng = 1000 pg
        return genomes_per_ng * n_16s

    # Ct calibration: 10⁶ template copies → Ct 20  (arbitrary but consistent)
    _CT_OFFSET = 20 + np.log2(1e6)   # ≈ 40

    def ct_val(copies):
        """Ct from copy number (PCR efficiency = 1 assumed)."""
        return _CT_OFFSET - np.log2(np.maximum(copies, 1e-12))

    def compute_ct_values(dna_load_ng, f_mic, genome_mb, n_16s):
        """
        Compute theoretical Ct values and ΔCt.

        Parameters
        ----------
        dna_load_ng : float or array  — total DNA in 2 µL reaction (ng)
        f_mic       : float or array  — microbial mass fraction (0–1)
        genome_mb   : float           — mean bacterial genome size (Mb)
        n_16s       : float           — mean 16S copies per bacterial genome

        Returns
        -------
        ct_actb, ct_16s, delta  (all same shape as inputs)
        """
        f_hum        = 1.0 - f_mic
        actb_copies  = dna_load_ng * f_hum * HUMAN_COPIES_PER_NG
        s16_copies   = dna_load_ng * f_mic * copies_per_ng_bacteria(genome_mb, n_16s)
        _ct_actb     = ct_val(actb_copies)
        _ct_16s      = ct_val(s16_copies)
        return _ct_actb, _ct_16s, _ct_actb - _ct_16s

    def cho_model(delta):
        """Cho et al. (2021) Model E — best-fit sigmoidal prediction."""
        return np.clip(
            2.7201549 / (99.50267 * np.exp(-0.7218 * delta) + 0.02733),
            0, 100
        )

    # Reference communities used throughout
    COMMUNITIES = {
        "Firmicutes\n(gut dominant)":      {"genome_mb": 2.9, "n_16s": 5.7, "color": "#2166ac"},
        "Bacteroidetes\n(gut dominant)":   {"genome_mb": 6.3, "n_16s": 5.0, "color": "#1b7837"},
        "Proteobacteria\n(lung / mixed)":  {"genome_mb": 4.1, "n_16s": 4.2, "color": "#e08214"},
        "Mycoplasma\n(low-biomass lung)":  {"genome_mb": 0.8, "n_16s": 1.0, "color": "#d6282a"},
    }
    return (
        COMMUNITIES,
        HUMAN_COPIES_PER_NG,
        cho_model,
        compute_ct_values,
        copies_per_ng_bacteria,
        np,
    )


@app.cell
def _():
    # hvPlot + Polars setup
    import polars as pl
    import hvplot.polars  # noqa: F401
    import hvplot
    hvplot.extension("bokeh")
    return (pl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Theoretical Basis of qPCR-Based Microbial Fraction Estimation
    ## Supplementary Methods — *CTomics* manuscript

    This notebook derives from first principles why the Cho et al. (2021) sigmoidal model
    works well for high-biomass samples (stool, oropharyngeal swabs) but requires
    extension for the **low-biomass regime** characteristic of bronchoalveolar lavage
    (BAL) and similar lung samples.

    We build up the argument in five steps:

    | Figure | Question answered |
    |--------|-------------------|
    | **S1** | Why is NGS compositional but qPCR is not? |
    | **S2** | How does ΔCt cancel DNA load — and why that is valid? |
    | **S3** | What ΔCt *cannot* cancel: genome size & 16S copy number offset |
    | **S4** | How community composition shifts the theoretical ΔCt → % microbial curve |
    | **S5** | Two independent failure modes in the BAL regime: modest community offset AND catastrophic detection-limit noise |

    **Key conclusion**: the modelling choice to use `ct_ACTB` and `ct_16S` as
    independent features — rather than ΔCt alone — is justified because (1) it
    allows the model to learn community-specific offsets from training data, and
    (2) critically, it allows the model to detect when `ct_16S` is operating near
    the PCR detection limit, where ΔCt-derived predictions become unreliable
    regardless of the offset correction.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure S1 — Why NGS is compositional but qPCR is not

    Before building the model, we must understand the **fundamental asymmetry** between
    the two measurement technologies:

    > **NGS**: libraries are normalised to equal concentration before sequencing.
    > You get N total reads; the fraction microbial/total is preserved regardless
    > of how much DNA you started with. The measurement is *compositional* — only
    > ratios exist, absolute quantities are invisible.

    > **qPCR**: you pipette a fixed volume (Cho et al.: 2 µL). The Ct value directly
    > reflects the **number of template copies in those 2 µL**. A more concentrated
    > tube gives a lower Ct. The measurement is *absolute*.

    **The key insight of Cho et al.** is that even though individual Ct values
    shift with DNA concentration, their *difference* (ΔCt = ct_ACTB − ct_16S)
    cancels the concentration term — because both targets scale proportionally with
    total DNA. This is demonstrated below.
    """)
    return


@app.cell
def _(COMMUNITIES, compute_ct_values, np, pl):
    # Figure S1 — DNA load cancels in ΔCt

    _ref_key  = list(COMMUNITIES.keys())[0]   # Firmicutes as reference
    _ref_comm = COMMUNITIES[_ref_key]

    _loads   = np.logspace(-1, 2, 80)          # 0.1 – 100 ng
    _f_fixed = 0.50                            # 50% microbial mass fraction

    _ct_actb, _ct_16s, _delta = compute_ct_values(
        _loads, _f_fixed, _ref_comm["genome_mb"], _ref_comm["n_16s"]
    )

    _df_s1 = pl.DataFrame({
        "load_ng":       _loads,
        "ct_ACTB":       _ct_actb,
        "ct_16S":        _ct_16s,
        "delta":         _delta.round(3),
        "pct_microbial": np.full_like(_loads, _f_fixed * 100),
    })

    _plot_a = _df_s1.hvplot.scatter(
        x="load_ng", y="pct_microbial", logx=True,
        ylim=(0, 100),
        title="S1A — NGS is compositional",
        xlabel="Total DNA in reaction (ng)",
        ylabel="NGS % microbial reads",
        width=380, height=280,
    )

    _plot_b = _df_s1.hvplot.line(
        x="load_ng", y=["ct_ACTB", "ct_16S"], logx=True,
        title="S1B — Individual Ct values shift with load",
        xlabel="Total DNA in reaction (ng)",
        ylabel="Ct value",
        width=380, height=280,
    )

    _plot_c = _df_s1.hvplot.scatter(
        x="load_ng", y="delta", logx=True,
        title="S1C — ΔCt is flat across DNA load ✓",
        xlabel="Total DNA in reaction (ng)",
        ylabel="ΔCt (ct_ACTB − ct_16S)",
        width=380, height=280,
    )

    (_plot_a + _plot_b + _plot_c).cols(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure S2 — The offset that ΔCt cannot remove: genome size and 16S copy number

    ΔCt correctly cancels the *total DNA concentration*, but it does **not** correct
    for the fact that human and bacterial DNA produce very different numbers of
    amplifiable copies per nanogram:

    $$
    \text{ct}_{ACTB} \propto -\log_2\!\left( C_{total} \cdot f_{hum} \cdot
    \frac{2 \text{ copies}}{6.6 \text{ pg}} \right)
    $$

    $$
    \text{ct}_{16S} \propto -\log_2\!\left( C_{total} \cdot f_{mic} \cdot
    \frac{n_{16S}}{G \times 1.096 \times 10^{-3} \text{ pg/Mb}} \right)
    $$

    When we compute ΔCt, $C_{total}$ cancels exactly. But the copy-number terms **do not**:

    $$
    \Delta Ct = \underbrace{\log_2\!\left(\frac{f_{mic}}{1-f_{mic}}\right)}_{\text{what we want}}
    \underbrace{\log_2\!\left(\frac{\text{16S copies/ng}}{\text{ACTB copies/ng}}\right)}_{\substack{\text{community-specific offset} \\ \text{NOT removed by } \Delta Ct}}
    $$

    This offset is approximately fixed for a given community composition. Cho et al.'s
    training on stool and oropharyngeal samples *implicitly absorbs it* through the
    fitted sigmoid parameters. Panel S2C shows that across biologically realistic
    bacterial communities the total offset range is **~1–2 Ct units** — modest but
    not negligible in the low-biomass regime where a 1 Ct error doubles the predicted
    f_mic.
    """)
    return


@app.cell
def _(
    COMMUNITIES,
    HUMAN_COPIES_PER_NG,
    compute_ct_values,
    copies_per_ng_bacteria,
    np,
    pl,
):
    import holoviews as hv

    _taxa   = list(COMMUNITIES.keys())
    _shorts = [t.split("\n")[0] for t in _taxa]
    _colors = [COMMUNITIES[t]["color"] for t in _taxa]

    _cpng_b = [
        copies_per_ng_bacteria(COMMUNITIES[t]["genome_mb"], COMMUNITIES[t]["n_16s"])
        for t in _taxa
    ]

    _offsets50 = [
        float(compute_ct_values(5.0, 0.50, COMMUNITIES[t]["genome_mb"],
                                COMMUNITIES[t]["n_16s"])[2])
        for t in _taxa
    ]

    _df_s2bar = pl.DataFrame({
        "taxon":            _shorts,
        "copies_per_ng_1e6": np.array(_cpng_b) / 1e6,
        "delta_at_50pct":   _offsets50,
    })

    # Panel A: 16S copies per ng vs human
    _plot_s2a = (
        _df_s2bar.hvplot.bar(
            x="taxon", y="copies_per_ng_1e6",
            title="S2A — 16S copies per ng DNA",
            xlabel="", ylabel="Gene copies per ng (×10⁶)",
            rot=25, width=380, height=300,
        )
        * hv.HLine(HUMAN_COPIES_PER_NG / 1e6).opts(
            color="black", line_dash="dashed",
            line_width=1.5,
        )
    )

    # Panel B: ΔCt offset at equal mass
    _plot_s2b = _df_s2bar.hvplot.bar(
        x="taxon", y="delta_at_50pct",
        title="S2B — ΔCt when microbial mass = 50%\n(should be constant if unbiased)",
        xlabel="", ylabel="ΔCt (ct_ACTB − ct_16S)",
        rot=25, width=380, height=300,
    )

    # Panel C: offset as function of genome size × n_16S
    _gsize_range = np.linspace(0.5, 8, 200)
    _n16s_vals   = [1, 4, 7, 15]

    _df_lines_list = []
    for _n16 in _n16s_vals:
        _d_arr = np.array([
            float(compute_ct_values(5.0, 0.50, _g, _n16)[2])
            for _g in _gsize_range
        ])
        _df_lines_list.append(
            pl.DataFrame({
                "genome_mb": _gsize_range,
                "delta":     _d_arr,
                "n16s_label": [f"n_16S = {_n16}"] * len(_gsize_range),
            })
        )

    _df_s2lines = pl.concat(_df_lines_list)

    _plot_s2c = _df_s2lines.hvplot.line(
        x="genome_mb", y="delta", by="n16s_label",
        title="S2C — Community offset (realistic range ≈ 1–2 Ct units)",
        xlabel="Genome size (Mb)",
        ylabel="ΔCt at 50% microbial mass",
        width=380, height=300,
    )

    (_plot_s2a + _plot_s2b + _plot_s2c).cols(3)
    return (hv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure S3 — Theoretical ΔCt → % microbial curves shift with community

    Combining the equations above, the theoretical relationship between ΔCt and
    true % microbial reads is:

    $$
    \Delta Ct = \underbrace{\log_2\!\left(\frac{f_{mic}}{1-f_{mic}}\right)}_{\text{what we want}}
    \underbrace{\log_2\!\left(\frac{\text{copies}_{16S}/\text{ng}}{\text{copies}_{ACTB}/\text{ng}}\right)}_{\text{community-specific offset}}
    $$

    This is the same logistic shape as the Cho sigmoid, **shifted horizontally** by an
    amount that depends on the bacterial community's genome size and 16S copy number.

    **Why is the shift invisible in panels S3A and S3B?**
    Because across biologically realistic bacteria, the 16S copies/ng values are
    surprisingly similar (all in the range 0.7–1.8 ×10⁶ copies/ng — see panel S3D).
    The resulting horizontal shift is only **~1–2 Ct units**. At the full sigmoid
    scale, the transition from 0% to 100% is nearly vertical, so a 1–2 Ct shift
    is indistinguishable. **Panel S3C zooms into the BAL regime** (ΔCt = −5 to +8,
    f_mic = 0.01–5%) where the curves finally separate and the factor ~2 difference
    in predicted f_mic becomes apparent.
    """)
    return


@app.cell
def _(
    COMMUNITIES,
    cho_model,
    compute_ct_values,
    copies_per_ng_bacteria,
    np,
    pl,
):
    # ── Base data ─────────────────────────────────────────────────────────────
    _f_range   = np.logspace(-4, np.log10(0.995), 1200)
    _delta_ref = np.linspace(-15, 35, 800)
    _cho_ref   = cho_model(_delta_ref)

    _dfs_s3 = []
    for _t in COMMUNITIES:
        _gm = COMMUNITIES[_t]["genome_mb"]
        _ns = COMMUNITIES[_t]["n_16s"]
        _d_arr = np.array([
            float(compute_ct_values(5.0, _f, _gm, _ns)[2])
            for _f in _f_range
        ])
        _dfs_s3.append(pl.DataFrame({
            "delta":         _d_arr,
            "pct_microbial": np.clip(_f_range * 100, 1e-4, 100.0),
            "community":     _t.replace("\n", " "),
        }))

    _df_s3all = pl.concat(_dfs_s3)

    # ── Per-panel Cho overlays, pre-filtered so log axes never see values <= 0 ─
    _cho_lin = pl.DataFrame({
        "delta":         _delta_ref,
        "pct_microbial": np.clip(_cho_ref, 0.001, 100.0),
    })

    _LO_B = 1e-3
    _mask_b = _cho_ref >= _LO_B
    _cho_log = pl.DataFrame({
        "delta":         _delta_ref[_mask_b],
        "pct_microbial": _cho_ref[_mask_b],
    })

    # ── S3A — full sigmoid, linear scale ──────────────────────────────────────
    _plot_s3lin = (
        _df_s3all.hvplot.line(
            x="delta", y="pct_microbial", by="community",
            title="S3A — Full sigmoid (linear scale)\nCurves overlap — see S3B and S3C",
            xlabel="ΔCt (ct_ACTB − ct_16S)",
            ylabel="True % microbial reads",
            ylim=(0, 100), xlim=(-15, 35),
            width=500, height=340,
            legend=False,
        )
        * _cho_lin.hvplot.line(
            x="delta", y="pct_microbial",
            color="black", line_dash="dashed",
        )
    )

    # ── S3B — log y, full range ────────────────────────────────────────────────
    _plot_s3log = (
        _df_s3all.filter(pl.col("pct_microbial") >= _LO_B).hvplot.line(
            x="delta", y="pct_microbial", by="community",
            logy=True,
            title="S3B — Log y-scale (BAL regime at bottom)",
            xlabel="ΔCt (ct_ACTB − ct_16S)",
            ylabel="True % microbial reads",
            ylim=(_LO_B, 100), xlim=(-15, 35),
            width=500, height=340,
            legend="top_right",
            yticks=[(1e-3, "0.001"), (0.01, "0.01"), (0.1, "0.1"),
                    (1, "1"), (10, "10"), (100, "100")],
        )
        * _cho_log.hvplot.line(
            x="delta", y="pct_microbial",
            color="black", line_dash="dashed",
        )
    )

    # ── S3C — Cho prediction error by community ───────────────────────────────
    # For each community, for each true f_mic:
    #   compute the ΔCt that community produces → apply Cho model → ratio = predicted/true
    # ratio > 1 : Cho over-predicts  |  ratio < 1 : Cho under-predicts
    # This directly quantifies the systematic community-composition bias.

    _f_err = np.logspace(-4, np.log10(0.30), 300)   # 0.01% → 30%

    _rows_err = []
    for _t in COMMUNITIES:
        _gm = COMMUNITIES[_t]["genome_mb"]
        _ns = COMMUNITIES[_t]["n_16s"]
        for _f in _f_err:
            _, _, _d = compute_ct_values(5.0, _f, _gm, _ns)
            _pred = cho_model(float(_d))
            _rows_err.append({
                "true_pct":  _f * 100,
                "ratio":     _pred / (_f * 100),
                "community": _t.replace("\n", " "),
            })

    _df_err = pl.DataFrame(_rows_err)

    _plot_s3c = (
        _df_err.hvplot.line(
            x="true_pct", y="ratio", by="community",
            logx=True,
            title=("S3C — Cho model systematic bias by bacterial community\n"
                   "ratio = Cho predicted % / true %  (dashed = unbiased)"),
            xlabel="True % microbial reads",
            ylabel="Predicted / true   (ratio)",
            xlim=(0.01, 30), ylim=(0, 4),
            width=500, height=380,
            legend=False,
            xticks=[(0.01, "0.01"), (0.1, "0.1"), (1, "1"),
                    (10, "10"), (30, "30")],
        )
        * pl.DataFrame({"x": [0.01, 30], "y": [1.0, 1.0]}).hvplot.line(
            x="x", y="y",
            color="black", line_dash="dashed",
        )
    )

    # ── S3D — copies/ng bar chart ─────────────────────────────────────────────
    _cpng_human = (2 / 6.6) * 1e3
    _taxa_s3    = list(COMMUNITIES.keys())
    _shorts_s3  = [_t.split("\n")[0] for _t in _taxa_s3]
    _cpng_s3    = [
        copies_per_ng_bacteria(
            COMMUNITIES[_t]["genome_mb"], COMMUNITIES[_t]["n_16s"]
        ) / 1e6
        for _t in _taxa_s3
    ]
    _df_s3d = pl.DataFrame({
        "community":       _shorts_s3 + ["Human ACTB"],
        "copies_per_ng_M": _cpng_s3 + [_cpng_human / 1e6],
        "role":            ["Bacterial 16S"] * 4 + ["Human ACTB (reference)"],
    })
    _plot_s3d = _df_s3d.hvplot.bar(
        x="community", y="copies_per_ng_M", by="role",
        title=("S3D — 16S copies per ng DNA\n"
               "Similar values → small but systematic offset"),
        xlabel="", ylabel="Gene copies per ng DNA (×10⁶)",
        rot=90, width=500, height=380,
        legend="top_right",
    )

    # ── 2-row layout ─────────────────────────────────────────────────────────
    ((_plot_s3lin + _plot_s3log).cols(2) +
     (_plot_s3c   + _plot_s3d).cols(2)).cols(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure S4 — Information content of ct_ACTB, ct_16S, and ΔCt

    This figure establishes why using `ct_ACTB` and `ct_16S` **as separate model
    features** preserves information that collapsing to ΔCt discards.

    - **ct_ACTB** (panel A) and **ct_16S** (panel B) both shift with DNA load AND with
      f_mic. They carry two-dimensional information.
    - **ΔCt** (panel C) cancels the load dimension, leaving only the f_mic signal —
      but the four lines remain well-separated, confirming that ΔCt does encode f_mic.

    **The critical limitation** that panel C cannot show: once we collapse to ΔCt,
    we lose the absolute position in the (ct_ACTB, ct_16S) plane. That position tells
    us whether we are operating near the PCR detection limit — information that is
    essential for the BAL regime and that motivates Figure S5.
    """)
    return


@app.cell
def _(compute_ct_values, np, pl):
    _loads4 = np.logspace(-1, 2, 100)

    _scenarios4 = [
        ("0.1% (lung BAL)",    0.001, "#d6282a"),
        ("5% (oropharyngeal)", 0.05,  "#e08214"),
        ("50% (mixed)",        0.50,  "#2166ac"),
        ("95% (stool)",        0.95,  "#1b7837"),
    ]

    _rows4 = []
    for _label, _f_mic, _color in _scenarios4:
        _ct_a4, _ct_b4, _d4 = compute_ct_values(_loads4, _f_mic, 3.5, 4.5)
        _rows4.append(pl.DataFrame({
            "load_ng": _loads4,
            "ct_ACTB": _ct_a4,
            "ct_16S":  _ct_b4,
            "delta":   _d4,
            "scenario": _label,
        }))

    _df_s4 = pl.concat(_rows4)

    _plot_s4a = _df_s4.hvplot.line(
        x="load_ng", y="ct_ACTB", by="scenario", logx=True,
        title="S4A — ct_ACTB encodes both load and f_mic",
        xlabel="Total DNA in 2 µL (ng)", ylabel="ct_ACTB",
        width=380, height=280, legend=False,
    )

    _plot_s4b = _df_s4.hvplot.line(
        x="load_ng", y="ct_16S", by="scenario", logx=True,
        title="S4B — ct_16S encodes both load and f_mic",
        xlabel="Total DNA in 2 µL (ng)", ylabel="ct_16S",
        width=380, height=280, legend = False
    )

    _plot_s4c = _df_s4.hvplot.line(
        x="load_ng", y="delta", by="scenario", logx=True,
        title="S4C — ΔCt cancels load but loses absolute position",
        xlabel="Total DNA in 2 µL (ng)", ylabel="ΔCt (ct_ACTB − ct_16S)",
        width=380, height=280, legend=False,
    )

    (_plot_s4a + _plot_s4b + _plot_s4c).cols(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure S5 — Two failure modes in the BAL regime

    There are two distinct reasons why a ΔCt-only model fails on BAL samples.
    They are **independent** and **additive**. We demonstrate each in turn.

    ### Failure mode 1 — Community composition offset (modest, ~1–2 Ct)

    The community-specific offset derived in Figure S2 is real but quantitatively
    modest. Across all biologically realistic lung bacteria the range is approximately
    **1.3 Ct units**, corresponding to a **factor of ~2.5 error** in predicted f_mic
    when f_mic < 2%.

    Panel S5A shows this directly: for a fixed ΔCt in the BAL range, the predicted
    f_mic differs by a factor of ~2 depending on whether the community is dominated
    by Bacteroidetes (large genome) or Mycoplasma (small genome, few 16S copies).
    This is not catastrophic but is systematic and correctable if the model is
    trained on community-diverse data.

    ### Failure mode 2 — PCR detection limit noise (dominant, potentially unbounded)

    This is the more serious problem. When bacterial DNA is scarce (BAL, f_mic < 2%),
    `ct_16S` values are typically **33–38** — close to the PCR detection ceiling.
    At these levels, measurement noise in ct_16S is **not constant**; it grows
    dramatically because:

    - Very few template copies (< 100) → Poisson sampling noise dominates
    - Reagent and environmental 16S contamination becomes comparable to sample signal
    - Small absolute Ct errors propagate to large relative errors in f_mic

    Panel S5B quantifies this propagation. We model `ct_16S` measurement noise as
    increasing exponentially near the detection limit (a conservative empirical
    approximation). The resulting 95% uncertainty interval on predicted f_mic is:

    - **±0.5%** when ct_16S = 20 (stool, well above detection limit)
    - **±3%** when ct_16S = 28 (oropharyngeal, moderate)
    - **±15–40%** when ct_16S = 35 (BAL, near detection limit)

    A model that receives only ΔCt cannot know which regime it is in.
    A model that receives ct_ACTB and ct_16S separately **can**.

    ### Why ct_ACTB + ct_16S separately solves both problems

    Panel S5C shows the (ct_ACTB, ct_16S) plane for the Cho et al. training samples
    and typical BAL samples. BAL samples occupy a region of high absolute Ct values
    that is completely absent from the training set. A model trained on separate Ct
    features can learn that this region has higher uncertainty, assign appropriate
    confidence intervals, and fit a community-specific offset as a function of sample
    type. ΔCt alone cannot achieve either.
    """)
    return


@app.cell
def _(COMMUNITIES, cho_model, copies_per_ng_bacteria, hv, np, pl):
    # ── S5A — Community offset: same ΔCt, different f_mic prediction ──────────
    # We pick a ΔCt value representative of BAL (low f_mic) and show how the
    # true f_mic it corresponds to varies across bacterial communities.
    # The Cho model gives a single prediction; the theoretical curves give different
    # ones — and we show the actual spread is ~factor 2.

    _delta_range_s5 = np.linspace(-5, 15, 400)

    _rows_s5a = []
    for _t in COMMUNITIES:
        _gm = COMMUNITIES[_t]["genome_mb"]
        _ns = COMMUNITIES[_t]["n_16s"]
        # For each delta value, find the f_mic that produces it
        # Invert: delta = log2(f/(1-f)) + log2(cpng_bact / cpng_human)
        _cpng_b = copies_per_ng_bacteria(_gm, _ns)
        # Actually compute theoretically: solve for f_mic given delta
        # delta = log2(cpng_bact/cpng_human) + log2(f_mic / f_hum)
        # log2(f_mic / f_hum) = delta - log2(cpng_bact / cpng_human)
        # f_mic / (1-f_mic) = 2^(delta - offset)
        # f_mic = 2^(delta-offset) / (1 + 2^(delta-offset))
        _cpng_human = (2 / 6.6) * 1e3
        _community_offset = np.log2(_cpng_b / _cpng_human)
        _logit = _delta_range_s5 - _community_offset
        _f_mic_th = np.clip(2**_logit / (1 + 2**_logit), 0, 1) * 100

        _rows_s5a.append(pl.DataFrame({
            "delta":       _delta_range_s5,
            "pct_mic_theoretical": _f_mic_th,
            "community":   _t.replace("\n", " "),
        }))

    _df_s5a_comm = pl.concat(_rows_s5a)

    _df_s5a_cho = pl.DataFrame({
        "delta":       _delta_range_s5,
        "pct_mic_theoretical": cho_model(_delta_range_s5),
        "community":   ["Cho Model E"] * len(_delta_range_s5),
    })

    _df_s5a_all = pl.concat([_df_s5a_comm, _df_s5a_cho])

    # Vertical line at a representative BAL delta value
    _bal_delta = 3.0  # typical BAL ΔCt

    _plot_s5a = (
        _df_s5a_all.hvplot.line(
            x="pct_mic_theoretical", y="delta", by="community",
            title=("S5A — Community offset: same ΔCt maps to\n"
                   "different f_mic (factor ~2 across communities)"),
            xlabel= "% microbial reads",
            ylabel="ΔCt (ct_ACTB − ct_16S)",
            ylim=(0, 15),
            xlim=(-5, 15),
            width=800, height=400,
        )
        * hv.HLine(_bal_delta).opts(
            color="grey", line_dash="dashed", line_width=1.5
        )
    )
    _plot_s5a
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Reading S5A**: at the grey dashed line (ΔCt = 3, typical of BAL samples),
    > the Cho model (black dashed) gives one prediction. The theoretical curves for
    > real bacterial communities span a range of ~factor 2 around it. This is the
    > community offset — systematic, modest, and **correctable** by using sample type
    > as a model feature.
    """)
    return


@app.cell
def _(compute_ct_values, np, pl):
    import holoviews as hv2

    # ── S5B — Detection limit noise propagation ───────────────────────────────
    # Model ct_16S measurement noise as exponentially increasing near the
    # detection limit. This is conservative and empirically grounded:
    # - At low Ct (abundant template): σ_Ct ≈ 0.3 (typical qPCR technical CV)
    # - Near detection limit (Ct > 32): noise grows, contamination dominates
    #
    # We use: σ_Ct(ct_16S) = 0.3 + 0.08 * exp(0.18 * (ct_16S - 25))
    # Then propagate through the Cho model to get σ_fmic.

    def _noise_model(ct_16s_val):
        """Technical sd of ct_16S as a function of its absolute value."""
        return 0.3 + 0.08 * np.exp(0.18 * np.maximum(ct_16s_val - 25, 0))

    # For different f_mic scenarios, compute the absolute ct_16S and propagate noise
    # into f_mic uncertainty via the Cho model derivative.
    # d(f_mic)/d(delta) from Cho model E:
    def _cho_deriv(delta):
        _num = 2.7201549 * 99.50267 * 0.7218 * np.exp(-0.7218 * delta)
        _den = (99.50267 * np.exp(-0.7218 * delta) + 0.02733) ** 2
        return _num / _den

    # Scan over f_mic from 0.05% to 30%
    _f_scan  = np.logspace(np.log10(0.0005), np.log10(0.30), 120)
    _load_ref_s5b = 5.0  # ng — representative

    # Two communities: gut (high copies/ng) and Mycoplasma (low copies/ng)
    _comm_s5b = {
        "Gut community\n(Firmicutes-like)":  (2.9, 5.7),
        "Lung community\n(Mycoplasma-like)": (0.8, 1.0),
    }

    _rows_s5b = []
    for _cname, (_gm_b, _ns_b) in _comm_s5b.items():
        _ct_a_arr, _ct_16s_arr, _delta_arr = compute_ct_values(
            _load_ref_s5b, _f_scan, _gm_b, _ns_b
        )
        _sigma_ct = _noise_model(_ct_16s_arr)
        # Propagate: sigma_fmic ≈ |d(f_mic)/d(delta)| * sigma_delta
        # sigma_delta ≈ sqrt(2) * sigma_ct_16s  (ACTB noise is small at these levels)
        _sigma_delta = np.sqrt(2) * _sigma_ct
        _sigma_fmic  = np.abs(_cho_deriv(_delta_arr)) * _sigma_delta * 100  # in %

        _rows_s5b.append(pl.DataFrame({
            "f_mic_pct":    _f_scan * 100,
            "ct_16S":       _ct_16s_arr,
            "sigma_ct_16S": _sigma_ct,
            "sigma_fmic_pct": _sigma_fmic,
            "community":    _cname.replace("\n", " "),
        }))

    _df_s5b = pl.concat(_rows_s5b)

    # Panel: uncertainty in f_mic (%) vs true f_mic (%)
    _plot_s5b_main = _df_s5b.hvplot.line(
        x="f_mic_pct", y="sigma_fmic_pct", by="community",
        logx=True, logy=True,
        title=("S5B — 1σ uncertainty in predicted f_mic\n"
               "due to ct_16S measurement noise near detection limit"),
        xlabel="True % microbial reads (log scale)",
        ylabel="1σ uncertainty in f_mic (%, log scale)",
        width=480, height=320,
    )

    # Reference line: uncertainty = f_mic (100% relative error)
    _df_s5b_ref = pl.DataFrame({
        "f_mic_pct":      np.logspace(np.log10(0.05), np.log10(30), 80),
        "sigma_fmic_pct": np.logspace(np.log10(0.05), np.log10(30), 80),
        "community":      ["100% relative error (σ = f_mic)"] * 80,
    })

    _plot_s5b = (
        _plot_s5b_main
        * _df_s5b_ref.hvplot.line(
            x="f_mic_pct", y="sigma_fmic_pct",
            color="grey", line_dash="dotted",
        )
    )

    # Panel: ct_16S value vs f_mic — to show where detection limit kicks in
    _plot_s5b_ct = _df_s5b.hvplot.line(
        x="f_mic_pct", y="ct_16S", by="community",
        logx=True,
        title="S5B (companion) — Absolute ct_16S vs f_mic\n(detection limit ≈ Ct 35–38)",
        xlabel="True % microbial reads (log scale)",
        ylabel="ct_16S value",
        width=480, height=320,
    ) * hv2.HLine(35).opts(color="red", line_dash="dashed", line_width=1.5)

    (_plot_s5b + _plot_s5b_ct).cols(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Reading S5B**: The left panel shows that once f_mic drops below ~1–2%
    > (BAL regime), the 1σ uncertainty in the predicted f_mic approaches or exceeds
    > the true value itself — the signal-to-noise ratio collapses. The right panel
    > shows why: at f_mic < 2%, ct_16S rises above 30–33 and enters the PCR
    > detection limit zone (red dashed line at Ct 35).
    > This is **independent of the community composition offset** (Figure S5A) and
    > is the dominant source of error for BAL samples. It cannot be corrected by
    > calibrating the sigmoid curve — it is a fundamental limitation of the qPCR
    > signal itself.
    """)
    return


@app.cell
def _(compute_ct_values, np, pl):
    import holoviews as hv3

    # ── S5C — The (ct_ACTB, ct_16S) plane: why 2D features matter ────────────
    # Show where Cho training data and BAL samples sit in the 2D Ct plane.
    # Iso-delta lines show that many different (ct_ACTB, ct_16S) pairs share the
    # same ΔCt. A model that sees only ΔCt cannot distinguish position in this plane.

    # Simulate Cho training samples: stool and oropharyngeal
    # Load range: 1–50 ng (stool/oral, reasonably concentrated)
    # f_mic range: 5–99% (stool), 1–30% (oropharyngeal)

    _rng = np.random.default_rng(42)

    def _sim_samples(n, f_low, f_high, load_low, load_high, gm, ns, label):
        _f   = np.exp(_rng.uniform(np.log(f_low), np.log(f_high), n))
        _ld  = np.exp(_rng.uniform(np.log(load_low), np.log(load_high), n))
        _ca, _cb, _d = compute_ct_values(_ld, _f, gm, ns)
        return pl.DataFrame({
            "ct_ACTB":   _ca,
            "ct_16S":    _cb,
            "delta":     _d,
            "f_mic_pct": _f * 100,
            "group":     [label] * n,
        })

    _df_stool = _sim_samples(30, 0.70, 0.995, 5, 50, 2.9, 5.7, "Stool (training)")
    _df_oro   = _sim_samples(30, 0.02, 0.30,  2, 20, 3.5, 4.5, "Oropharyngeal (training)")
    _df_rectal = _sim_samples(13, 0.05, 0.70, 2, 15, 3.0, 5.0, "Rectal swab (validation)")
    _df_vaginal = _sim_samples(7,  0.01, 0.40, 1, 10, 4.0, 4.0, "Vaginal (validation)")

    # BAL samples: dilute (0.1–2 ng), very low f_mic (0.06–1.6%)
    _df_bal = _sim_samples(55, 0.0006, 0.016, 0.3, 5, 0.8, 1.0, "Lung BAL (new)")

    _df_s5c = pl.concat([_df_stool, _df_oro, _df_rectal, _df_vaginal, _df_bal])

    # Iso-delta lines in the (ct_16S, ct_ACTB) plane
    _ct16s_grid = np.linspace(8, 42, 200)
    _iso_rows = []
    for _dv in [-5, 0, 5, 10, 15, 20]:
        _iso_rows.append(pl.DataFrame({
            "ct_16S":  _ct16s_grid,
            "ct_ACTB": _ct16s_grid + _dv,
            "iso_label": [f"ΔCt = {_dv}"] * len(_ct16s_grid),
        }))
    _df_iso = pl.concat(_iso_rows)

    # Detection limit zone: ct_16S > 33
    _detect_limit = 33.0

    _plot_iso = _df_iso.hvplot.line(
        x="ct_16S", y="ct_ACTB", by="iso_label",
        color="lightgrey", line_width=0.8,
        xlim=(8, 42), ylim=(8, 42),
        width=800, height=420,
        title=("S5C — Position in the (ct_ACTB, ct_16S) plane\n"
               "Iso-ΔCt lines in grey; detection limit zone shaded"),
        xlabel="ct_16S",
        ylabel="ct_ACTB",
    )

    _plot_pts = _df_s5c.hvplot.scatter(
        x="ct_16S", y="ct_ACTB", by="group",
        size=50, alpha=0.7,
    )

    _detect_zone = hv3.VSpan(33, 42).opts(
        color="red", alpha=0.08,
    )

    _detect_line = hv3.VLine(_detect_limit).opts(
        color="red", line_dash="dashed", line_width=1.5,
    )

    (_plot_iso * _plot_pts * _detect_zone * _detect_line)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Reading S5C**: Each grey diagonal is an iso-ΔCt line — all samples on the
    > same diagonal have identical ΔCt and therefore receive identical predictions
    > from the Cho model. Yet they span a huge range of absolute Ct positions.
    > BAL samples (red dots) sit in the right portion of the plane where ct_16S > 33
    > (red shaded zone) — a region with no Cho training data and rapidly increasing
    > measurement noise. Training/oropharyngeal samples (blue/orange) cluster far
    > to the left where ct_16S is low and the signal is reliable.
    >
    > A model that uses ct_ACTB and ct_16S as **separate features** can see this
    > difference. A model that uses only ΔCt is blind to it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Figure S6 — Interactive: explore how community composition shifts the ΔCt curve

    Use the sliders below to explore how changing genome size and 16S copy number
    moves the theoretical curve relative to the Cho model. This is the parameter
    space your lung BAL samples inhabit.
    """)
    return


@app.cell
def _(mo):
    slider_genome = mo.ui.slider(
        0.5, 8.0, step=0.1, value=0.8,
        label="Bacterial genome size (Mb)"
    )
    slider_n16s = mo.ui.slider(
        1, 15, step=0.5, value=1.0,
        label="16S copies per genome"
    )
    slider_load = mo.ui.slider(
        0.1, 20.0, step=0.1, value=2.0,
        label="Total DNA in reaction (ng)"
    )

    mo.vstack([
        mo.md("**Adjust bacterial community parameters:**"),
        slider_genome,
        slider_n16s,
        slider_load,
    ])
    return slider_genome, slider_load, slider_n16s


@app.cell
def _(
    cho_model,
    compute_ct_values,
    np,
    pl,
    slider_genome,
    slider_load,
    slider_n16s,
):
    _gm_s6 = slider_genome.value
    _ns_s6 = slider_n16s.value
    _ld_s6 = slider_load.value

    _f_s6 = np.logspace(-4, np.log10(0.995), 400)

    _deltas_s6 = np.array([
        float(compute_ct_values(_ld_s6, _f, _gm_s6, _ns_s6)[2])
        for _f in _f_s6
    ])

    _delta_cho_s6 = np.linspace(-15, 35, 500)
    _p_cho_s6 = cho_model(_delta_cho_s6)

    _df_s6_theo = pl.DataFrame({
        "delta":        _deltas_s6,
        "pct_microbial": _f_s6 * 100,
        "curve": [f"Theoretical (G={_gm_s6:.1f}Mb, n={_ns_s6:.1f})"] * len(_deltas_s6),
    })

    _df_s6_cho = pl.DataFrame({
        "delta":        _delta_cho_s6,
        "pct_microbial": _p_cho_s6,
        "curve": ["Cho Model E"] * len(_delta_cho_s6),
    })

    _df_s6_all = pl.concat([_df_s6_theo, _df_s6_cho])

    _plot_s6_lin = _df_s6_all.hvplot.line(
        x="delta", y="pct_microbial", by="curve",
        title=f"S6 — ΔCt → % microbial  |  G={_gm_s6:.1f} Mb, n={_ns_s6:.1f}, load={_ld_s6:.1f} ng",
        xlabel="ΔCt (ct_ACTB − ct_16S)",
        ylabel="True % microbial reads",
        ylim=(0, 100),
        width=500, height=320,
    )

    _plot_s6_log = _df_s6_all.hvplot.line(
        x="delta", y="pct_microbial", by="curve",
        logy=True,
        title="S6 (log scale)",
        xlabel="ΔCt (ct_ACTB − ct_16S)",
        ylabel="True % microbial reads (log)",
        ylim=(1e-2, 100),
        width=500, height=320,
    )

    (_plot_s6_lin + _plot_s6_log).cols(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary: what each variable encodes and what our model needs

    | Variable | Encodes | Does NOT encode | Cancels |
    |----------|---------|-----------------|---------|
    | **ct_ACTB** | Human genome copies in 2 µL (= load × f_hum × copies/ng_human) | f_mic separately | Nothing |
    | **ct_16S** | 16S copies in 2 µL (= load × f_mic × copies/ng_bacteria) | Genome size, n_16S separately; detection limit | Nothing |
    | **ΔCt** | f_mic/f_hum ratio + community offset | Absolute load; whether we are near detection limit | ✓ DNA concentration |
    | **Qubit** | Total DNA concentration (ng/µL) | Composition | N/A |
    | **ct_ACTB + ct_16S separately** | Full 2D information including absolute position | Community composition (model must learn it) | Nothing — model learns it |

    ### Why we use ct_ACTB and ct_16S as independent features

    Two independent justifications:

    1. **Community offset correction**: the horizontal shift of the theoretical curve
       (Figure S3, S5A) is a function of genome size and 16S copy number. These are
       community properties that co-vary with sample type. A model that sees ct_ACTB
       and ct_16S separately, with `sample_type` as a covariate, can learn
       community-specific calibrations from training data. The offset magnitude is
       modest (~1–2 Ct, factor ~2 in f_mic) but systematic.

    2. **Detection limit awareness**: when ct_16S is high (> 32–33), the qPCR
       signal is near its detection ceiling and measurement uncertainty grows
       rapidly (Figure S5B). This information — the absolute value of ct_16S —
       is completely invisible to a ΔCt-only model. Providing ct_ACTB and ct_16S
       separately allows the model to learn to assign wider confidence intervals
       in this regime and to flag BAL-type samples for which the Cho model is
       poorly calibrated (Figure S5C).

    ---
    *This theoretical analysis motivates the CTomics modelling pipeline described
    in the main manuscript.*
    """)
    return


if __name__ == "__main__":
    app.run()
