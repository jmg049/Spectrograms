#!/usr/bin/bash
# set -euo pipefail
#
# Usage:
#   ./build.sh          — build all sizes (a4, a5, a6)
#   ./build.sh a4       — build A4 only
#   ./build.sh a5       — build A5 only
#   ./build.sh a6       — build A6 only

AUX_EXTENSIONS=(
  aux bbl blg brf fls fdb_latexmk
  log lof lot out toc synctex.gz
  nav snm run.xml xdv
)

TEX_FILE="Computational Audio and Image Analysis with the Spectrograms Library.tex"
TEX_BASE="$(basename "$TEX_FILE" .tex)"
BUILD_DIR="./build"

SIZE_ARG="${1:-all}"

case "$SIZE_ARG" in
  a4|a5|a6) SIZES=("$SIZE_ARG") ;;
  all)       SIZES=(a4 a5 a6) ;;
  *)
    echo "Error: unknown size '${SIZE_ARG}'. Use a4, a5, a6, or omit for all."
    exit 1
    ;;
esac

if [ ! -f "$TEX_FILE" ]; then
  echo "Error: ${TEX_FILE} not found"
  exit 1
fi

mkdir -p "$BUILD_DIR"

echo ">> Cleaning auxiliary files..."
for ext in "${AUX_EXTENSIONS[@]}"; do
  find . -name "*.${ext}" -exec rm -v {} + 2>/dev/null || true
done

echo ">> Formatting tex..."
tex-fmt --recursive .

# Filter that keeps only useful diagnostics
filter_tex_output() {
  grep -E --line-buffered \
    '(^!|Warning|Overfull|Underfull|Undefined|Error|Fatal|BibTeX)'
}

run_pdflatex() {
  local wrapper="$1"
  local jobname="$2"
  pdflatex \
    -interaction=nonstopmode \
    -file-line-error \
    -jobname="$jobname" \
    -output-directory="$BUILD_DIR" \
    "$wrapper" 2>&1 | filter_tex_output
}

run_bibtex() {
  local jobname="$1"
  biber "./build/${jobname}" 2>&1
}

for SZ in "${SIZES[@]}"; do
  case "$SZ" in
    a4) FONTSIZE=11 ;;
    a5) FONTSIZE=10 ;;
    a6) FONTSIZE=9  ;;
  esac

  JOBNAME="${TEX_BASE} (${SZ^^})"
  WRAPPER="_papersize_wrapper_${SZ}.tex"

  printf '\\PassOptionsToClass{%spt}{extarticle}\n\\def\\PAPERFORMAT{%s}\n\\input{%s}\n' \
    "$FONTSIZE" "$SZ" "$(basename "$TEX_FILE" .tex)" > "$WRAPPER"

  echo ">> Building ${SZ^^} PDF..."
  run_pdflatex "$WRAPPER" "$JOBNAME"
  run_bibtex   "$JOBNAME"
  run_pdflatex "$WRAPPER" "$JOBNAME"
  run_pdflatex "$WRAPPER" "$JOBNAME"

  rm -f "$WRAPPER"

  mv "${BUILD_DIR}/${JOBNAME}.pdf" "./${JOBNAME}.pdf"
  echo "PDF: ./${JOBNAME}.pdf"
done

echo ">> Done."
