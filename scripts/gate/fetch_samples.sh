#!/usr/bin/env bash
# Download the four Phase 3 gate datasets from Kaggle and write a random
# sample of each, all OUTSIDE the repo (default ~/lh-data; override with
# LH_DATA). Needs the Kaggle CLI and your own token (~/.kaggle/kaggle.json,
# chmod 600); IEEE-CIS also needs its competition rules accepted on the
# Kaggle website. Raw files can be deleted afterwards; the samples stay.
#
#   scripts/gate/fetch_samples.sh            # all four
#   scripts/gate/fetch_samples.sh ulb paysim # a subset
set -euo pipefail

DATA="${LH_DATA:-$HOME/lh-data}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
case "$(realpath -m "$DATA")/" in
  "$REPO"/*) echo "LH_DATA must be outside the repo ($REPO)" >&2; exit 1 ;;
esac
KAGGLE="${KAGGLE:-$(command -v kaggle || echo "$HOME/.local/bin/kaggle")}"
ROWS="${LH_SAMPLE_ROWS:-200000}"
mkdir -p "$DATA/raw" "$DATA/samples"

fetch() {  # name kind ref file
  local name=$1 kind=$2 ref=$3 file=$4 out="$DATA/raw/$1"
  mkdir -p "$out"
  if ls "$out"/*.csv >/dev/null 2>&1; then echo "[$name] raw present"; return; fi
  echo "[$name] downloading $ref $file"
  if [ "$kind" = competition ]; then
    "$KAGGLE" competitions download -c "$ref" -f "$file" -p "$out"
  else
    "$KAGGLE" datasets download -d "$ref" -f "$file" -p "$out"
  fi
  (cd "$out" && for z in *.zip; do [ -e "$z" ] && unzip -oq "$z" && rm -f "$z"; done) || true
}

ARGS=("$@")
sel() { [ ${#ARGS[@]} -eq 0 ] || [[ " ${ARGS[*]} " == *" $1 "* ]]; }

sel paysim && fetch paysim dataset ealaxi/paysim1 PS_20174392719_1491204439457_log.csv
sel ulb    && fetch ulb dataset mlg-ulb/creditcardfraud creditcard.csv
sel baf    && fetch baf dataset sgpjesus/bank-account-fraud-dataset-neurips-2022 Base.csv
sel ieee   && fetch ieee competition ieee-fraud-detection train_transaction.csv

python3 "$REPO/scripts/gate/sample.py" "$DATA" "$ROWS" "${ARGS[@]}"
