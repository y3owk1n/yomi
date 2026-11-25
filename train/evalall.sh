#!/bin/bash
# Reads every eval set (data/eval-*) with BIN (default bin/yomi, -model auto)
# and scores it. Usage: evalall.sh TAG [BIN]
#   -> data/runs/TAG--SET.jsonl and one summary line per set on stdout
set -e
cd "$(dirname "$0")"
TAG=$1; BIN=${2:-../bin/yomi}
mkdir -p data/runs
for s in code layout linux ui-western ui-cjk ui-rtl ui-indic ui-sea ho-indic ho-sea ho-cau ho-rtl; do
  d=data/eval-$s; out=data/runs/$TAG--$s.jsonl
  [ -d $d ] || continue
  $BIN bench -model auto -reps 1 $d > $out
  case $s in ui-cjk|ui-sea|ho-sea) flag=--chars;; *) flag=;; esac
  printf "%-12s " $s; .venv/bin/python pagescore.py $d $out $flag | tail -2 | sed "s|^[^ ]* *||" | tr "\n" " "; echo
done
