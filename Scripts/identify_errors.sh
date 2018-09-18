#!/bin/bash
set -euo pipefail

tmpdir=$(mktemp -d)

numatlas=$1
numtemplates=$2
subject=$3
dataset=$4
output=$5

shift 5

./majority_vote.py "$@" $tmpdir/majority-vote.mnc
./awol_mrf_identify_errors.py --clobber --brain_image ${dataset}/input/atlases/brains/${subject}.mnc --dataset $dataset --n_atlases $numatlas --n_templates $numtemplates --output_data ${output}_${dataset}.csv --majority_labels $tmpdir/majority-vote.mnc --manual_labels ${dataset}/input/atlases/labels/${subject}_labels.mnc "$@" ${tmpdir}/awol-vote4.8.mnc

rm -rf $tmpdir

