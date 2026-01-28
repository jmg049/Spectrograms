#!/usr/bin/bash
# set -euo pipefail

AUX_EXTENSIONS=(
  aux bbl blg brf fls fdb_latexmk
  log lof lot out toc synctex.gz
  nav snm run.xml xdv
)

TEX_FILE="Computational Audio and Image Analysis with the Spectrograms Library.tex"
BUILD_DIR="./build"

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
  pdflatex \
    -interaction=nonstopmode \
    -file-line-error \
    -output-directory="$BUILD_DIR" \
    "$TEX_FILE" 2>&1 | filter_tex_output

}

run_bibtex() {
    biber ./build/"$(basename "$TEX_FILE" .tex)" 2>&1
}

echo ">> Building PDF..."

run_pdflatex 
run_bibtex  
run_pdflatex
run_pdflatex 

echo ">> Done."
mv "${BUILD_DIR}/$(basename "$TEX_FILE" .tex).pdf" ./
echo "PDF: ./Computational Audio and Image Analysis with the Spectrograms Library.pdf"