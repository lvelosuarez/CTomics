#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# fetch_cho_sra.sh
# Download Cho et al. 2021 shotgun reads (PRJNA718445), QC with BBDuk,
# dehost with hostile, and summarise host-removal statistics per sample.
#
# REQUIREMENTS:
#   conda install -c bioconda sra-tools hostile bowtie2 pigz fastp
#   hostile fetch --name human-t2t-hla-argos985
#
# OUTPUT:
#   cho_qc/               – per-sample fastp-trimmed FASTQs
#   cho_bbduk_logs/       – per-sample fastp logs (json + html + log)
#   cho_dehost/           – per-sample dehosted FASTQs
#   cho_hostile_logs/     – per-sample hostile logs
#   cho_hostile_stats.csv – sample | reads_in | reads_removed_proportion | reads_out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

BIOPROJECT="PRJNA718445"
FASTQ_DIR="cho_fastqs"
QC_DIR="cho_qc"
QC_LOG_DIR="cho_bbduk_logs"
DEHOST_DIR="cho_dehost"
LOG_DIR="cho_hostile_logs"
OUT_CSV="cho_hostile_stats.csv"

mkdir -p "$FASTQ_DIR" "$FASTQ_DIR/tmp" "$QC_DIR" "$QC_LOG_DIR" "$DEHOST_DIR" "$LOG_DIR"

# ── Step 1: Check accession list ─────────────────────────────────────────────
echo "=== Step 1: Fetch SRA run list for ${BIOPROJECT} ==="

if [ ! -f "AccessionList.txt" ]; then
  echo "Please download AccessionList.txt from:"
  echo "  https://www.ncbi.nlm.nih.gov/Traces/study/?acc=${BIOPROJECT}"
  echo "  (Send Results → Accession List → Download)"
  exit 1
fi

# ── Step 2: Download FASTQs ──────────────────────────────────────────────────
echo "=== Step 2: Download FASTQs ==="

while IFS= read -r SRR; do
  [[ -z "$SRR" ]] && continue

  # Skip if FASTQ already exists
  if ls "$FASTQ_DIR"/${SRR}*.fastq.gz 1>/dev/null 2>&1; then
    echo "  [SKIP] $SRR — FASTQ already present"
    continue
  fi

  echo "  Downloading $SRR ..."
  fasterq-dump "$SRR" \
    --outdir "$FASTQ_DIR" \
    --threads "$THREADS" \
    --skip-technical \
    --progress \
    --temp "$FASTQ_DIR/tmp"

  # Compress immediately after dump (gzip single-stream for Java/BBDuk compatibility)
  echo "  Compressing $SRR ..."
  gzip "$FASTQ_DIR"/${SRR}*.fastq

  echo "  [DONE] $SRR"

done < AccessionList.txt

echo "=== Download complete ==="

# ── Step 3: QC with fastp ─────────────────────────────────────────────────────
echo "=== Step 3: QC samples with fastp ==="

for R1 in "$FASTQ_DIR"/*_1.fastq.gz; do
  SAMPLE="${R1##*/}"
  SAMPLE="${SAMPLE/_1.fastq.gz/}"
  R2="${FASTQ_DIR}/${SAMPLE}_2.fastq.gz"

  if [[ ! -f "$R2" ]]; then
    echo "[WARN] Missing R2 for $SAMPLE, skipping" >&2
    continue
  fi

  # Skip if QC output already exists and is non-empty
  if [[ -s "$QC_DIR/${SAMPLE}_1.fastq.gz" ]]; then
    echo "[SKIP] $SAMPLE — QC already done"
    continue
  fi

  echo "[INFO] fastp: $SAMPLE"

  fastp \
    --in1 "$R1" --in2 "$R2" \
    --out1 "$QC_DIR/${SAMPLE}_1.fastq.gz" --out2 "$QC_DIR/${SAMPLE}_2.fastq.gz" \
    --cut_front --cut_front_mean_quality 20 \
    --cut_tail  --cut_tail_mean_quality  20 \
    --average_qual 20 \
    --length_required 100 \
    --detect_adapter_for_pe \
    --thread "$THREADS" \
    --json "$QC_LOG_DIR/${SAMPLE}.fastp.json" \
    --html "$QC_LOG_DIR/${SAMPLE}.fastp.html" \
    2>&1 | tee "$QC_LOG_DIR/${SAMPLE}.fastp.log"

  echo "[INFO] Done: $SAMPLE"
done

echo "[INFO] fastp QC complete"

# ── Step 4: Dehost with hostile ──────────────────────────────────────────────
echo "=== Step 4: Dehost samples with hostile ==="

for QC_R1 in "$QC_DIR"/*_1.fastq.gz; do
  SAMPLE="${QC_R1##*/}"
  SAMPLE="${SAMPLE/_1.fastq.gz/}"
  QC_R2="${QC_DIR}/${SAMPLE}_2.fastq.gz"

  if [[ ! -f "$QC_R2" ]]; then
    echo "[WARN] Missing R2 for $SAMPLE, skipping" >&2
    continue
  fi

  echo "[INFO] Processing $SAMPLE"

  hostile clean \
    --fastq1 "$QC_R1" \
    --fastq2 "$QC_R2" \
    --index human-t2t-hla-argos985 \
    --aligner bowtie2 \
    --airplane \
    -o "$DEHOST_DIR" \
    --threads "$DEHOST_THREADS" --force \
    2>&1 | tee "$LOG_DIR/${SAMPLE}.hostile.log"

  echo "[INFO] Done: $SAMPLE"
done

echo "[INFO] All samples processed"

# ── Step 5: Summarise host-removal stats ─────────────────────────────────────
echo "=== Step 5: Summarise hostile statistics ==="

awk '
  /"reads_in"/ {
    gsub(",", "", $2);
    reads_in[FILENAME] = $2 / 2;
  }
  /"reads_removed_proportion"/ {
    gsub(",", "", $2);
    rrp[FILENAME] = $2;
  }
  /"reads_out"/ {
    gsub(",", "", $2);
    reads_out[FILENAME] = $2 / 2;
  }
  END {
    print "sample,reads_in,reads_removed_proportion,reads_out";
    for (f in reads_in) {
      split(f, parts, "/");
      fname = parts[length(parts)];
      sub(/\.hostile\.log$/, "", fname);
      printf "%s,%.0f,%.5f,%.0f\n", fname, reads_in[f], rrp[f], reads_out[f];
    }
  }
' "$LOG_DIR"/*hostile* > "$OUT_CSV"

echo "=== Done! ==="
echo "Results saved to $OUT_CSV"
