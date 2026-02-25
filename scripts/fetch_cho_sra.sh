#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# fetch_cho_sra.sh
# Download Cho et al. 2021 shotgun reads (PRJNA718445) and compute
# % microbial reads for each sample using KneadData.
#
# REQUIREMENTS (install before running):
#   conda install -c bioconda sra-tools kneaddata bowtie2 trimmomatic
#   # KneadData human reference (GRCh38):
#   kneaddata_database --download human_genome bowtie2 ./kneaddata_db/
#
# OUTPUT:
#   cho_kneaddata_results/   – per-sample KneadData logs
#   cho_microbial_reads.csv  – sample | reads_total | reads_microbial | pct_microbial
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

BIOPROJECT="PRJNA718445"
KNEADDATA_DB="/mnt/san/microbio/database/kneaddata_db"     # adjust path to your bowtie2 human index
THREADS=8
OUTDIR="cho_kneaddata_results"
FASTQ_DIR="cho_fastqs"

mkdir -p "$OUTDIR" "$FASTQ_DIR"

echo "=== Step 1: Fetch SRA run list for ${BIOPROJECT} ==="
# Requires esearch/efetch (entrez-direct) or just use the accession list from
# SRA Run Selector: https://www.ncbi.nlm.nih.gov/Traces/study/?acc=PRJNA718445
# Save the AccessionList.txt from there, then run this script.

if [ ! -f "AccessionList.txt" ]; then
  echo "Please download AccessionList.txt from:"
  echo "  https://www.ncbi.nlm.nih.gov/Traces/study/?acc=${BIOPROJECT}"
  echo "  (Send Results → Accession List → Download)"
  exit 1
fi

echo "=== Step 2: Download FASTQs ==="
while IFS= read -r SRR; do
  [[ -z "$SRR" ]] && continue
  echo "  Downloading $SRR ..."
  prefetch "$SRR" --output-directory "$FASTQ_DIR"
  fastq-dump --gzip --split-files \
    --outdir "$FASTQ_DIR" \
    "$FASTQ_DIR/${SRR}/${SRR}.sra" 2>/dev/null \
    || fasterq-dump "$SRR" --outdir "$FASTQ_DIR" --threads "$THREADS"
done < AccessionList.txt

echo "=== Step 3: Run KneadData and compute % microbial reads ==="
echo "sample,reads_total,reads_microbial,pct_microbial" > cho_microbial_reads.csv

while IFS= read -r SRR; do
  [[ -z "$SRR" ]] && continue

  R1="${FASTQ_DIR}/${SRR}_1.fastq.gz"
  R2="${FASTQ_DIR}/${SRR}_2.fastq.gz"

  # Paired-end check
  if [ -f "$R1" ] && [ -f "$R2" ]; then
    KNEADDATA_ARGS="-i1 $R1 -i2 $R2"
  elif [ -f "$R1" ]; then
    KNEADDATA_ARGS="-i $R1"
  else
    echo "  WARNING: no FASTQ found for $SRR – skipping"
    continue
  fi

  echo "  KneadData: $SRR ..."
  kneaddata \
    $KNEADDATA_ARGS \
    --reference-db "$KNEADDATA_DB" \
    --output "${OUTDIR}/${SRR}" \
    --threads "$THREADS" \
    --trimmomatic-options "SLIDINGWINDOW:4:20 MINLEN:50" \
    --bowtie2-options "--very-sensitive" \
    --remove-intermediate-output \
    --log "${OUTDIR}/${SRR}/kneaddata.log"

  # Parse read counts from KneadData log
  LOG="${OUTDIR}/${SRR}/kneaddata.log"
  TOTAL=$(grep -oP "(?<=Total reads after trimming: )\d+" "$LOG" | head -1 || echo 0)
  MICROBIAL=$(grep -oP "(?<=Final contaminant reads: )" "$LOG" || true)
  # More robust: count final output reads
  MICROBIAL=$(zcat "${OUTDIR}/${SRR}"/*kneaddata_paired*.fastq.gz 2>/dev/null | \
              awk 'NR%4==1' | wc -l || echo 0)
  TOTAL=$(zcat "${OUTDIR}/${SRR}"/*kneaddata.log 2>/dev/null | \
          grep -oP "(?<=Total reads after trimming: )\d+" | head -1 || echo "$TOTAL")

  if [ "$TOTAL" -gt 0 ]; then
    PCT=$(python3 -c "print(round(${MICROBIAL}/${TOTAL}*100, 4))")
  else
    PCT=0
  fi

  echo "  $SRR: total=${TOTAL}, microbial=${MICROBIAL}, pct=${PCT}%"
  echo "${SRR},${TOTAL},${MICROBIAL},${PCT}" >> cho_microbial_reads.csv

done < AccessionList.txt

echo ""
echo "=== Done! ==="
echo "Results saved to cho_microbial_reads.csv"
echo ""
echo "Next step: join cho_microbial_reads.csv with Table S3 Ct values, then run:"
echo "  python train_models.py --cho_sra cho_microbial_reads.csv --cho_ct TableS3.xlsx"
