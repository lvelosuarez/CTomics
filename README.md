<p align="left">
  <img src="images/logo.svg" alt="CTomics logo" width="320"/>
</p>

**CT-Based Prediction of Host–Microbiome Library Composition Prior to Shotgun Sequencing**

## The Problem

In host-associated metagenomics, microbial reads can represent anywhere from 0.01% to >99% of a sequencing library — but this is only known *after* sequencing. Two failure modes result:

- **Underpowering**: insufficient microbial depth for resistome profiling, taxonomic resolution, or functional analysis
- **Oversequencing**: wasted reads when the host fraction is unexpectedly high

This is especially acute for **low-biomass respiratory specimens** (BAL, sputum) where microbial fractions below 4% are the norm, not the exception.

## The Approach

CTomics predicts the microbial read fraction **before sequencing** from two routine qPCR measurements:

| Measurement | Target | Proxy for |
|---|---|---|
| `Ct_16S` | 16S rRNA gene | Total bacterial DNA load |
| `Ct_ACTB` | Human β-actin gene | Total human DNA load |

The key derived feature is:

```
ΔCt = Ct_ACTB − Ct_16S
```

Higher ΔCt → more bacterial DNA relative to human DNA → higher expected microbial read fraction.

## Models

CTomics provides two complementary prediction models:

### Cho Model E (baseline)

Cho et al. (2021) showed that ΔCt follows a four-parameter sigmoid (R² = 0.990 on stool and oropharyngeal samples):

$$\hat{y} = \frac{2.7201549}{99.50267 \cdot e^{-0.7218 \cdot \Delta Ct} + 0.02733}$$

**Limitation:** The sigmoid has a lower asymptote at ~0.027%, making it unable to resolve samples below ~4% microbial reads. BAL and other low-biomass respiratory specimens cluster precisely in this compressed tail. Bootstrap LOOCV (1000 iterations, n=154) confirms RMSE ≈ 12% in the <4% zone.

### GPR — site-aware model

A Gaussian Process Regressor with Matérn ν=1.5 kernel trained on 154 samples across six clinical specimen types. Uses Ct_ACTB, Ct_16S, ΔCt, and a one-hot-encoded specimen type as features (9 features total):

$$\hat{y} = \frac{100}{1 + e^{-\mu}}, \quad \mu \sim \mathcal{GP}\!\left(\text{Matérn}_{\nu=1.5},\,\sigma_f^2=3.33,\,l=10.9\right)$$

The fitted kernel `1.82² × Matérn(l=10.9, ν=1.5) + WhiteKernel(0.067)` was selected by marginal likelihood. Outputs include calibrated posterior uncertainty (σ) and asymmetric 95% CIs via `inv-logit(μ ± 1.96σ)`.

**Performance vs Cho (bootstrap LOOCV, n=154):**

| Zone | Cho RMSE | GPR RMSE | ΔRMSE | p-value |
|---|---|---|---|---|
| All samples | 5.7% | 2.4% | −3.3% | <0.05 |
| BAL regime (<2%) | ~12% | ~2.4% | ~−10% | <0.001 |
| Cho failure zone (<4%) | ~12% | ~4.5% | −7.8% | <10⁻³¹ |

The site OHE alone accounts for ~3.5 pp of the improvement (p=10⁻³¹), reflecting the fact that identical ΔCt maps to very different microbial fractions across specimen types.

## Training Data

154 samples across six specimen types (ΔCt range −3.45 to +26.4, microbial fraction 0.06–99.5%):

| Specimen type | n |
|---|---|
| Lung BAL | ~68 |
| Oropharyngeal | ~55 |
| Stool | ~14 |
| Rectal swab | ~7 |
| Vaginal sample | ~7 |
| Lung sputum | 13 |

**Sources:**
- **Cho et al. 2021** — Ct values from Table S3; percent microbial reads re-analysed from NCBI for validation samples
- **REPAIR cohort** — BAL from patients with rheumatoid arthritis-associated ILD; Ct_16S (SYBR Green) and Ct_ACTB (TaqMan); microbial fraction from Illumina shotgun sequencing after KneadData host filtering
- **Lung sputum** — independent out-of-distribution validation set (n=13)

---

## Web Tool

**Live tool:** [lvelosuarez.github.io/CTomics](https://lvelosuarez.github.io/CTomics/)

The predictor runs entirely in the browser. The GPR model parameters are embedded as Float64 arrays in `ctomics.html`; inference uses a pure-JavaScript implementation of the Matérn kernel + Cholesky forward substitution (~2 ms for 20 samples). No server, no Python, no WebAssembly runtime required.

---

## Repository Structure

```
CTomics/
├── data/
│   ├── data.csv              # Combined training dataset (154 samples)
│   ├── data_original.csv     # Cho et al. 2021 data only
│   └── data_sputum.csv       # Lung sputum OOD validation set
├── models/
│   └── gpr_ctomics.pkl       # Fitted GPR + StandardScaler (joblib, sklearn 1.7)
├── notebooks/
│   ├── theory.py             # Theoretical derivation of the ΔCt–fraction relationship
│   ├── eda.py                # Exploratory data analysis
│   ├── modelling.py          # LOOCV benchmarking (6 models) + bootstrap CI + GPR serialisation
│   └── ood.py                # Out-of-distribution validation on lung sputum
├── images/
│   └── logo.svg
├── docs/
│   └── index.html            # Project landing page (standalone)
├── ctomics.html              # Interactive predictor (standalone, ~346 KB)
└── README.md
```

### `data/data.csv` schema

| Column | Description |
|---|---|
| `id` | Sample identifier |
| `sample_type` | Specimen type (`stool`, `oropharyngeal`, `rectal_swab`, `vaginal_sample`, `lung_bal`, `lung_sputum`) |
| `ct_16S` | 16S rRNA qPCR Ct value |
| `ct_ACTB` | β-actin qPCR Ct value |
| `delta` | ΔCt = Ct_ACTB − Ct_16S |
| `pct_microbial` | % reads surviving host-read filtering (target variable) |
| `run` | Sequencing run identifier |
| `instrument` | Sequencing platform |

---

## License

This project is dual-licensed:

• GPL-3.0 for open-source use
• Commercial license available for proprietary use

Contact: lourdes.velosuarez@chu-brest.fr
CTomics is for research purposes only. Data from Cho et al. 2021 used under CC BY 4.0. REPAIR cohort data is de-identified and used under institutional approval.

---

## References

> Cho MY, Wandro S, Fadrosh D, et al. *Two-Target Quantitative PCR To Predict Library Composition for Shallow Shotgun Sequencing.* mSystems. 2021;6(4):e00552-21. [doi:10.1128/mSystems.00552-21](https://doi.org/10.1128/mSystems.00552-21)

> Velo Suárez L, et al. *CTomics: site-aware Gaussian Process prediction of metagenomic microbial fraction from dual-target qPCR.* Manuscript in preparation.
