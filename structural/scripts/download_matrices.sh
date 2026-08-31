#!/usr/bin/env bash
# Download SuiteSparse matrices (MatrixMarket) for a matrix list.
#
#   structural/scripts/download_matrices.sh <matrix-list.txt> <dest-dir>
#
# List lines are <Group>/<Entry>/<file>.mtx (see structural/matrix_lists/);
# one TAMU tarball per <Group>/<Entry> is fetched and extracted to
# <dest-dir>/<Group>/<Entry>/, matching the layout the sweep drivers expect
# via --data-dir <dest-dir> --matrix-list <matrix-list.txt>.
set -u
LIST="${1:?usage: download_matrices.sh <matrix-list.txt> <dest-dir>}"
DEST="${2:?usage: download_matrices.sh <matrix-list.txt> <dest-dir>}"
mkdir -p "$DEST"

ok=0; fail=0; have=0
for ge in $(awk -F/ 'NF>=3 {print $1"/"$2}' "$LIST" | sort -u); do
    g="${ge%/*}"; e="${ge#*/}"; dir="$DEST/$g/$e"
    if [ -d "$dir" ] && ls "$dir"/*.mtx >/dev/null 2>&1; then
        have=$((have+1)); continue
    fi
    mkdir -p "$DEST/$g"
    if curl -sL --max-time 3600 "https://sparse.tamu.edu/MM/$g/$e.tar.gz" -o "$DEST/$g/$e.tar.gz" \
       && tar xzf "$DEST/$g/$e.tar.gz" -C "$DEST/$g"; then
        ok=$((ok+1)); echo "got  $g/$e"
    else
        fail=$((fail+1)); echo "FAIL $g/$e"; rm -rf "$dir"
    fi
    rm -f "$DEST/$g/$e.tar.gz"
done
echo "DONE downloaded=$ok cached=$have failed=$fail"
