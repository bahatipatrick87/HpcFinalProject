#!/bin/bash
# =============================================================================
# Submission Package Creator
# Università degli Studi di Messina — High Performance Computing
# Student : Bahati Patrick  (Matricola 573868)
# Course  : High Performance Computing 2025-2026
# Report  : Hybrid MPI+OpenCL+OpenMP Parallel Image Processing Pipeline
# Revision: Final — November 2026
# =============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDENT="BahatiPatrick"
MATRICOLA="573868"
ARCHIVE_NAME="HPC_${STUDENT}_${MATRICOLA}.tar.gz"
ARCHIVE_PATH="${PROJECT_ROOT}/${ARCHIVE_NAME}"

# -----------------------------------------------------------------------------
# Colours (safe-fallback if terminal doesn't support them)
# -----------------------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
BLUE='\033[0;34m';  BOLD='\033[1m';      NC='\033[0m'

header() { echo -e "\n${BOLD}${BLUE}=== $* ===${NC}"; }
ok()     { echo -e "  ${GREEN}[OK]${NC}  $*"; }
warn()   { echo -e "  ${YELLOW}[WARN]${NC} $*"; }
fail()   { echo -e "  ${RED}[FAIL]${NC} $*"; exit 1; }
info()   { echo -e "  ${BLUE}[INFO]${NC} $*"; }

echo ""
echo -e "${BOLD}=================================================="
echo -e " HPC Project — Submission Package Builder"
echo -e " Student : Bahati Patrick (573868)"
echo -e " Revision: Final — November 2026"
echo -e "==================================================${NC}"
echo ""

cd "${PROJECT_ROOT}"

# =============================================================================
# STEP 1 — Pre-flight checks
# =============================================================================
header "Step 1: Pre-flight checks"

# Report must exist
REPORT="${PROJECT_ROOT}/report/FINAL_COMPLETE_REPORT.tex"
if [[ -f "${REPORT}" ]]; then
    ok "Report LaTeX source: report/FINAL_COMPLETE_REPORT.tex"
else
    fail "Report not found at ${REPORT}"
fi

# C++ library must exist (built)
LIB_DYLIB="${PROJECT_ROOT}/lib/libpipeline_kernels.dylib"
LIB_SO="${PROJECT_ROOT}/lib/libpipeline_kernels.so"
if [[ -f "${LIB_DYLIB}" || -f "${LIB_SO}" ]]; then
    ok "C++ shared library present in lib/"
else
    warn "Shared library not found — run scripts/build.sh first"
fi

# Key source files
for f in src/pipeline_kernels.cpp src/pipeline_kernels.h \
          src/main_mpi.py src/dataset_loader.py \
          src/pipeline/batched_pipeline.py src/pipeline/workflow_builder.py \
          src/stages/fft_stage.py src/stages/filter_stage.py \
          src/stages/feature_stage.py src/stages/io_stage.py \
          config/pipeline.yaml requirements.txt README.md; do
    if [[ -f "${PROJECT_ROOT}/${f}" ]]; then
        ok "${f}"
    else
        warn "Missing: ${f}"
    fi
done

# Experiment scripts
for f in experiments/run_scaling_experiments.py \
          experiments/run_pipeline_timing.py \
          experiments/run_gpu_scaling.py \
          experiments/run_openmp_scaling.py \
          experiments/run_gpu_stage_timing.py \
          experiments/verify_fft_against_numpy.py \
          experiments/analyze_amdahl.py; do
    if [[ -f "${PROJECT_ROOT}/${f}" ]]; then
        ok "${f}"
    else
        warn "Missing: ${f}"
    fi
done

# Report images
for img in report/images/strong_scaling_optimized.png \
           report/images/weak_scaling.png \
           report/images/mapping_hierarchy.png \
           report/images/parallel_patterns.png \
           report/images/system_architecture.png; do
    if [[ -f "${PROJECT_ROOT}/${img}" ]]; then
        ok "${img}"
    else
        warn "Missing image: ${img} — run: python3 scripts/generate_report_images.py"
    fi
done

# Results CSVs (performance data tables in the report)
for csv in results/report_snapshot_final/strong_scaling.csv \
           results/report_snapshot_final/weak_scaling.csv \
           results/report_snapshot_final/openmp_scaling.csv; do
    if [[ -f "${PROJECT_ROOT}/${csv}" ]]; then
        ok "${csv}"
    else
        warn "Missing results: ${csv}"
    fi
done

# Tests
if [[ -f "${PROJECT_ROOT}/tests/test_correctness.py" ]]; then
    ok "tests/test_correctness.py + tests/test_task_graph.py (27 unit tests)"
fi

# =============================================================================
# STEP 2 — Clean build artefacts
# =============================================================================
header "Step 2: Cleaning temporary files"

find src experiments scripts tests -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
rm -rf .pytest_cache tests/.pytest_cache
rm -f "${ARCHIVE_PATH}"   # remove any stale archive

ok "Temporary files removed"

# =============================================================================
# STEP 3 — Build the submission archive
# =============================================================================
header "Step 3: Building submission archive"

tar -czf "${ARCHIVE_PATH}" \
  --exclude='.git' \
  --exclude='.DS_Store' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.o' \
  --exclude='*.log' \
  --exclude='venv' \
  --exclude='build' \
  --exclude='data/EuroSAT' \
  --exclude='.cache' \
  --exclude='.pytest_cache' \
  --exclude='*.tar.gz' \
  --exclude='.vscode' \
  --exclude='.claude' \
  --exclude='* copy.png' \
  src/CMakeLists.txt \
  src/pipeline_kernels.cpp \
  src/pipeline_kernels.h \
  src/main_mpi.py \
  src/dataset_loader.py \
  src/pipeline/ \
  src/stages/ \
  src/utils/ \
  config/pipeline.yaml \
  experiments/ \
  scripts/build.sh \
  scripts/generate_report_images.py \
  report/FINAL_COMPLETE_REPORT.tex \
  report/FINAL_COMPLETE_REPORT.pdf \
  report/images/ \
  "report/images .jpg" \
  tests/ \
  results/*.csv \
  results/README.md \
  results/report_snapshot_final/ \
  results/adhoc_measurements.md \
  lib/ \
  requirements.txt \
  README.md \
  RUN_COMMANDS.md \
  data/README.md

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    fail "Archive creation failed."
fi

ok "Archive created: ${ARCHIVE_NAME}"

# =============================================================================
# STEP 4 — Verify archive contents
# =============================================================================
header "Step 4: Archive content summary"

SIZE=$(du -h "${ARCHIVE_PATH}" | cut -f1)
FILE_COUNT=$(tar -tzf "${ARCHIVE_PATH}" | wc -l | tr -d ' ')

info "Archive  : ${ARCHIVE_NAME}"
info "Size     : ${SIZE}"
info "Files    : ${FILE_COUNT} entries"

echo ""
echo -e "  ${BOLD}Key files in archive:${NC}"
for pattern in \
    "report/FINAL_COMPLETE_REPORT.tex" \
    "src/pipeline_kernels.cpp" \
    "src/main_mpi.py" \
    "src/pipeline/batched_pipeline.py" \
    "src/stages/fft_stage.py" \
    "src/stages/filter_stage.py" \
    "config/pipeline.yaml" \
    "experiments/run_pipeline_timing.py" \
    "experiments/run_gpu_scaling.py" \
    "tests/test_correctness.py" \
    "results/report_snapshot_final/strong_scaling.csv"; do
    if tar -tzf "${ARCHIVE_PATH}" 2>/dev/null | grep -q "${pattern}"; then
        ok "${pattern}"
    else
        warn "NOT FOUND in archive: ${pattern}"
    fi
done

# =============================================================================
# STEP 5 — Compilation reminder
# =============================================================================
header "Step 5: Report compilation instructions"

echo ""
echo -e "  ${BOLD}To compile the LaTeX report to PDF:${NC}"
echo ""
echo "    Method A — Local (if pdflatex is installed):"
echo "      cd report"
echo "      pdflatex FINAL_COMPLETE_REPORT.tex"
echo "      pdflatex FINAL_COMPLETE_REPORT.tex   # second pass for TOC"
echo ""
echo "    Method B — Overleaf:"
echo "      1. Upload report/FINAL_COMPLETE_REPORT.tex"
echo "      2. Upload the entire report/images/ folder"
echo "      3. Compile with pdflatex"
echo ""
echo "  Required LaTeX packages (all standard in TeX Live / MikTeX):"
echo "    lmodern, microtype, geometry, graphicx, hyperref, listings,"
echo "    xcolor, float, titlesec, amsmath, amssymb, booktabs, caption,"
echo "    setspace, fancyhdr, enumitem, algorithm, algorithmic,"
echo "    tikz (with shapes.geometric, arrows, positioning, calc,"
echo "          backgrounds, fit libraries), mdframed, tocloft,"
echo "    parskip, array, multirow, longtable"

# =============================================================================
# STEP 6 — Submission checklist
# =============================================================================
header "Step 6: Submission checklist"

echo ""
echo -e "  ${BOLD}Files to submit to Prof. Salvatore Distefano:${NC}"
echo ""
echo -e "  ${GREEN}[1]${NC}  FINAL_COMPLETE_REPORT.pdf   (compiled from report/FINAL_COMPLETE_REPORT.tex)"
echo -e "  ${GREEN}[2]${NC}  ${ARCHIVE_NAME}"
echo ""
echo -e "${BOLD}=================================================="
echo -e " Package ready for submission!"
echo -e " Archive: ${ARCHIVE_PATH}"
echo -e "==================================================${NC}"
echo ""
