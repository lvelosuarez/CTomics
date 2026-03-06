<p align="left">
  <img src="images/logo.svg" alt="CTomics logo" width="320"/>
</p>

**CT-Based Prediction of Host–Microbiome Library Composition Prior to Shotgun Sequencing**

## The Problem

In host-associated samples (e.g., BAL, rectal swabs, oropharyngeal swabs), shotgun metagenomic libraries are dominated by human DNA. After computational host depletion against GRCh38, microbial reads can represent as little as **0.01–5% of total sequenced reads** — but this is only known *after* sequencing.

This creates two failure modes:
- **Underpowering**: insufficient microbial depth for resistome profiling, taxonomic resolution, or functional analysis
- **Oversequencing**: wasted reads and cost when the host fraction is unexpectedly high

## The Approach

CTomics uses **two routine qPCR measurements** taken before sequencing to predict microbial read fraction:

| Measurement | Target | Proxy for |
|---|---|---|
| `Ct_16S` | 16S rRNA gene | Total bacterial DNA load |
| `Ct_ACTB` | Human β-actin gene | Total human DNA load |

From these, a single derived feature is computed:

```
ΔCt = Ct_ACTB − Ct_16S
```

Higher ΔCt → more bacterial DNA relative to human DNA → higher expected microbial read fraction.

## Why Not Just Use the Cho et al. Model?

Cho et al. (2021, *mSystems*, DOI: [10.1128/msystems.00552-21](https://doi.org/10.1128/msystems.00552-21)) showed that ΔCt follows a sigmoidal relationship with sequencing microbial fraction (R² = 0.990). Their Model E is:

$$\hat{y} = \frac{2.7201549}{99.50267 \cdot e^{-0.7218 \cdot \Delta Ct} + 0.02733}$$

This model achieves R² = 0.990 on stool and oropharyngeal samples, but underestimates microbial fractions in transition-zone and low-biomass samples. CTomics aims to improve on this baseline specifically for clinical respiratory specimens.

- The sigmoid has a **lower asymptote of ~0.027%** — it cannot resolve samples below ~4% microbial reads
- Low-biomass samples such as **BAL fluid** cluster precisely in this compressed lower tail (ΔCt < 5)
- Predictive error in this regime has the greatest clinical consequence

> Cho MY, Wandro S, Fadrosh D, et al. *Two-Target Quantitative PCR To Predict Library Composition for Shallow Shotgun Sequencing.* mSystems. 2021;6(4):e00552-21. doi:[10.1128/mSystems.00552-21](https://doi.org/10.1128/mSystems.00552-21)


## What CTomics Does

CTomics extends the Cho framework into the **low-ΔCt regime** by:

1. Integrating Cho et al. training data with matched qPCR + sequencing data from the **REPAIR cohort** (BAL samples)
2. Training and benchmarking multiple machine learning models (Beta regression, Gaussian Process, gradient boosting, etc.) specifically optimised for the low-biomass region
3. Providing calibrated predictions with uncertainty estimates to support **a priori sequencing depth decisions**

---

## Repository Structure

```
CTomics/
├── scripts/
│   └──  
├── data/
│   └── data.csv              # Combined training dataset (Cho 2021 + REPAIR)
├── notebooks/                # Exploratory analysis and model training notebooks
├── images/
│   └── logo.svg
└── README.md
```

### `data/data.csv` schema

| Column | Description |
|---|---|
| `id` | Sample identifier |
| `sample_type` | Biological specimen type (`stool`, `oropharyngeal`, `rectal_swab`, `vaginal_sample`, `lung_BAL`) |
| `ct_16S` | 16S rRNA qPCR Ct value — bacterial load proxy |
| `ct_ACTB` | β-actin qPCR Ct value — human DNA load proxy |
| `delta` | ΔCt = Ct_ACTB − Ct_16S; key predictor of microbial fraction |
| `pct_microbial` | % reads surviving host-read filtering (target variable) |
| `source` | Dataset origin: `cho2021` or `this study` |

---

## Data Sources

**Cho et al. 2021** — Ct values transcribed from Table S3 of the supplementary material. Percent microbial reads for validation samples (rectal swab, vaginal; n = 20) were re-analyzed from their ncbi repository. 

**REPAIR cohort** — Bronchoalveolar lavage samples from patients with rheumatoid arthritis-associated interstitial lung disease. Ct_16S and Ct_ACTB were measured by SYBR Green and TaqMan qPCR, respectively. Percent microbial reads computed from real Illumina shotgun sequencing after KneadData host filtering.

---


## License

This project is dual-licensed:

• GPL-3.0 for open-source use <br>
• Commercial license available for proprietary use

Contact: lourdes.velosuarez@chu-brest-fr <br>
CTomics is for research purposes. Data from Cho et al. 2021 is used under the terms of the original publication (CC BY 4.0 / open access). REPAIR cohort data is de-identified and used under institutional approval.
