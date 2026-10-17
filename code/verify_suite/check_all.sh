#!/usr/bin/env bash
# One-command four-fold verification suite (papersh skill).
#   bash check_all.sh             # run all four checks
#   bash check_all.sh --compile    # additionally rebuild the paper, require 0 undefined
#
# Exit 0 = all green (safe to deliver). Non-zero = do NOT deliver.
# RULE: when a check fails, fix the DATA or the CODE, never the script.
set -u
cd "$(dirname "$0")"

# Make the spikergnn package importable. Overwrite PYTHONPATH (do not append to
# any inherited value) so an unrelated pre-existing entry (e.g. a CLI shim path)
# cannot pollute the import and break the junction-based `import spikergnn`.
# This matches the project's own training convention (PYTHONPATH=$PWD/programfiles).
# Repo layout: the `spikergnn` package lives at code/spikergnn (importable from the
# code/ root). DATA_ROOT points at the categorized data/ directory; override with
# DVM_DATA_ROOT=/abs/path/to/data when running the suite elsewhere.
export PYTHONPATH="$(cd .. && pwd)"
export DVM_DATA_ROOT="${DVM_DATA_ROOT:-$(cd ../.. && pwd)/data}"

PY="C:/Python314/python.exe"     # managed interpreter (torch + torch_geometric)
TEX="../../paper/DvM_Paper_Anonymous.tex"
TEXFULL="$(cd ../.. && pwd)/paper/DvM_Paper_Anonymous.tex"
PDFLATEX="/d/texlive/2026/bin/windows/pdflatex"

CHECKS=(
  "audit_data.py"
  "verify_numbers.py"
  "verify_hyperparams.py"
  "verify_algorithm_mapping.py"
  "verify_p0b.py"
  "bootstrap_stats.py"
  "verify_ablation_functional.py"
)

fail=0
hr() { printf '%s\n' "------------------------------------------------------------"; }

echo "============================================================"
echo " FOUR-FOLD VERIFICATION SUITE (${#CHECKS[@]} checks) @ $(date '+%F %T')"
echo "============================================================"

i=1
for c in "${CHECKS[@]}"; do
  echo
  echo "[${i}/${#CHECKS[@]}] $c"
  hr
  if [ ! -f "$c" ]; then
    echo "  (missing — skipped)"
    i=$((i + 1)); continue
  fi
  out=$("$PY" "$c" 2>&1) || fail=1
  echo "$out" | grep -Ei "RESULT|MISMATCH|DRIFT|\[!!\]|ALL .*MATCH|ALL .*CONSISTENT|OK\]" | head -12
  i=$((i + 1))
done

if [ "$#" -gt 0 ] && [ "$1" = "--compile" ]; then
  echo
  echo "[+] compiling $TEXFULL"
  hr
  "$PDFLATEX" -interaction=nonstopmode "$TEXFULL" >/tmp/vs_c1.log 2>&1
  "$PDFLATEX" -interaction=nonstopmode "$TEXFULL" >/tmp/vs_c2.log 2>&1
  n=$(grep -ac "undefined" "${TEXFULL%.tex}.log" 2>/dev/null | head -1 | tr -d '[:space:]')
  n=${n:-0}
  echo "undefined refs: $n"
  [ "$n" -eq 0 ] || fail=1
fi

echo
echo "============================================================"
if [ "$fail" -eq 0 ]; then
  echo " ALL CHECKS PASSED"
  echo
  echo " Reminder: these checks catch DRIFT in existing entries, not"
  echo " OMISSIONS. If you added a mechanism the mapping table does not"
  echo " cover, nothing will alert you — add it to verify_algorithm_mapping.py."
  exit 0
else
  echo " CHECKS FAILED — fix data/code, not the verification scripts"
  exit 1
fi
