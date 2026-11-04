#!/bin/bash
# Compile une figure standalone et dépose le PDF à côté de sa source.
# Usage (depuis n'importe où) : images/build_fig.sh images/<section>/<fig>/<fig>.tex
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
src=$(realpath "$1"); name=$(basename "$src" .tex); dir=$(dirname "$src")
tmp=$(mktemp -d)
(cd "$dir" && TEXINPUTS="$root:$root/images:" lualatex -interaction=nonstopmode -output-directory="$tmp" "$src" >"$tmp/log.txt" 2>&1) \
  || { tail -30 "$tmp/log.txt"; exit 1; }
cp "$tmp/$name.pdf" "$dir/$name.pdf"
rm -rf "$tmp"
echo "OK : $dir/$name.pdf ($(pdfinfo "$dir/$name.pdf" | awk '/Pages/{print $2}') page(s))"
