#!/bin/bash
# Exports checkpoints, builds bin/yomi-TAG with them in place of the
# committed models (restored on exit, so commit models first), and scores it
# on every eval set, the web dev and held-out sets (data/web-dev,
# data/web-heldout, rendered by urlrender.swift and read by Vision into
# runs/vision-accurate.jsonl), and the private real captures when
# data/real-bench exists. Usage: ft-eval.sh TAG [DIR=PT]... where DIR is the
# package under models/ (western, cjk, router, caucasian).
set -e
TAG=$1; shift
cd "$(dirname "$0")"
ROOT=$(cd .. && pwd)
trap 'git -C "$ROOT" checkout -q models' EXIT
for kv in "$@"; do
  fam=${kv%%=*}; pt=${kv#*=}
  out=data/exp-$TAG-$fam
  rm -rf $out
  .venv/bin/python export.py $pt $out > /dev/null
  cp $out/model.json $out/weights.bin $ROOT/models/$fam/
done
(cd $ROOT && go build -o bin/yomi-$TAG ./cmd/yomi)
BIN=$ROOT/bin/yomi-$TAG
if [ -d data/real-bench ]; then
  echo "== real captures"
  (cd data/real-bench && ../../.venv/bin/python vrecall.py -yomi $BIN | tail -1)
fi
echo "== eval sets"
./evalall.sh $TAG $BIN | awk '{s=$1; o=""; for(i=2;i<=NF;i++){ if($i=="word"||$i=="char") o=o" "$(i+1)"/"$(i+3)} printf "%-11s%s\n", s, o}'
for w in web-dev web-heldout; do
  [ -d data/$w ] || continue
  echo "== $w"
  (cd data/$w && $BIN bench -model auto -reps 1 . > runs/yomi-$TAG.jsonl &&
    ../../.venv/bin/python ../../webscore.py runs/vision-accurate.jsonl runs/yomi-$TAG.jsonl | tail -2)
done
