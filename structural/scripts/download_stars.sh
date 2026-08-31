#!/usr/bin/env bash
# Fetch the large/dense "stacking-star" SuiteSparse matrices into $1 (default cwd).
# Used to seed cross-device (H100/A800) joint runs from the TAMU mirror.
set -u
DEST="${1:-.}"
mkdir -p "$DEST"; cd "$DEST"
STARS="Koutsovasilis/F1 PARSEC/Ga41As41H72 PARSEC/Si41Ge41H72 Williams/cant Williams/consph Williams/pdb1HYS"
for gm in $STARS; do
    n="${gm#*/}"
    [ -f "$n.mtx" ] && { echo "have $n"; continue; }
    if curl -sL --max-time 1200 "https://sparse.tamu.edu/MM/$gm.tar.gz" -o "$n.tar.gz" \
       && tar xzf "$n.tar.gz" && mv "$n/$n.mtx" ./; then
        rm -rf "$n" "$n.tar.gz"
        echo "got $n ($(du -h "$n.mtx" | cut -f1))"
    else
        echo "FAIL $n"; rm -rf "$n" "$n.tar.gz"
    fi
done
echo "ALLDONE $(date +%T)"
