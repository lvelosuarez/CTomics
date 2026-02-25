<p align="left">
  <img src="images/logo.svg" alt="CTomics logo" width="320"/>
</p>

**CT-Based Prediction of Host–Microbiome Library Composition Prior to Shotgun Sequencing**

---

## Background

Shotgun metagenomic sequencing of microbiome samples — such as bronchoalveolar lavage (BAL) fluid, rectal swabs, or stool — is routinely used to profile microbial communities, detect pathogens, and characterise antibiotic resistance genes. However, a fundamental challenge in host-microbiome metagenomics is that total DNA extracted from a sample is dominated by the host (human) genome. After host-read filtering, the fraction of reads that are actually microbial can be as low as 0.01–5% in some samples.

This creates a critical sequencing budget problem: the host: microbe ratio is only known **after** sequencing, by which point the cost has already been incurred. Analyses requiring tens of millions of microbial reads (e.g., ~50 M reads for antimicrobial resistance profiling) may fail entirely if the sequencing depth is insufficient, or money is wasted on lanes of data that are 99% host.

---

## What is CTomics?

Two routine qPCR assays carry enough information to estimate the microbial fraction before sequencing:

- **16S rRNA qPCR** (Ct_16S): proxy for total bacterial load
- **β-actin qPCR** (Ct_ACTB): proxy for total human DNA load

The delta value ΔCt = Ct_ACTB − Ct_16S captures the relative abundance of bacteria vs. human in the extraction. Cho et al. (2021) have already shown this relationship follows a sigmoidal curve with R² = 0.990 across diverse sample types. However, their Model E was trained primarily on stool and oropharyngeal samples. Low-biomass respiratory samples, such as BAL, fall in the sigmoid's floor region (ΔCt < 5), where the model has limited resolution and clinical impact is highest.

**CTomics extends this approach to BAL lung samples** by combining the Cho et al. training data with real sequencing data from the REPAIR cohort, building and evaluating machine learning models that better predict the microbial fraction in low-microbiome biomass samples.

---

## Objectives

1. **Compile a multi-source training dataset** combining:
   - Cho et al. 2021 (n = 109; stool, oropharyngeal, rectal swab, vaginal samples)
   - REPAIR cohort (n = 85; lung BAL from BPCO patients)

2. **Train and benchmark ML models** to predict `% microbial reads` from Ct_16S, Ct_ACTB, and derived features (ΔCt, polynomial terms), using cross-validation with sample-type-aware splits.

3. **Estimate sequencing requirements** from a sample's qPCR profile:
   - Expected microbial read count for a planned sequencing depth
   - Minimum sequencing depth to reach a target (e.g. 50 M microbial reads)
   - Flag samples that require host depletion prior to library preparation

4. **Enable informed pre-sequencing decisions** — deplete, re-extract, or reprioritise samples before committing sequencing budget.

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

## Reference Model

Cho et al. (2021) proposed Model E, a four-parameter sigmoidal function:

$$\hat{y} = \frac{2.7201549}{99.50267 \cdot e^{-0.7218 \cdot \Delta Ct} + 0.02733}$$

This model achieves R² = 0.990 on stool and oropharyngeal samples, but underestimates microbial fractions in transition-zone and low-biomass samples. CTomics aims to improve on this baseline specifically for clinical respiratory specimens.

> Cho MY, Wandro S, Fadrosh D, et al. *Two-Target Quantitative PCR To Predict Library Composition for Shallow Shotgun Sequencing.* mSystems. 2021;6(4):e00552-21. doi:[10.1128/mSystems.00552-21](https://doi.org/10.1128/mSystems.00552-21)

---

## Data Sources

**Cho et al. 2021** — Ct values transcribed from Table S3 of the supplementary material. Percent microbial reads for validation samples (rectal swab, vaginal; n = 20) were digitised from Figure 2C. Training samples (stool, oropharyngeal; n = 89) use Model E predictions as surrogates, since individual points are not resolvable in the published figure at the sigmoid floor/ceiling.

**REPAIR cohort** — Bronchoalveolar lavage samples from patients with rheumatoid arthritis-associated interstitial lung disease. Ct_16S and Ct_ACTB were measured by SYBR Green and TaqMan qPCR, respectively. Percent microbial reads computed from real Illumina shotgun sequencing after KneadData host filtering.

---

## Dependencies

```
python >= 3.10
polars
scikit-learn
scipy
matplotlib
joblib
pyreadr        # to load R objects
rdata          # to parse RDX3-format .Rdata files
```

---

## License

This project is for research purposes. Data from Cho et al. 2021 is used under the terms of the original publication (CC BY 4.0 / open access). REPAIR cohort data is de-identified and used under institutional approval.
