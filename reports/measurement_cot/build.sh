#!/bin/zsh
# Regenerate the figures, copy them next to the source, and build the PDF.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../.."
cd "$root"
PYTHONPATH=. uv run python -m experiments.measurement_cot.figures
PYTHONPATH=. uv run python -m experiments.measurement_cot.numbers
mkdir -p "$here/figures"
cp output/measurement_cot/figures/*.pdf "$here/figures/"
cd "$here"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
echo "built $here/main.pdf"
