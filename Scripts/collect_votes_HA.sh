#!/bin/bash
set -euo pipefail

tmpdir=$(mktemp -d)

numatlas=$1
numtemplates=$2
subject=$3
output=$4
shift 4

if [[ ! -e $output ]]
then
 echo "NumAtlas,NumTemplate,Type,Label,Total/Target,Jaccard,Dice,VolumeSimilarity,FalseNegative,FalsePositive" > $output
fi

./majority_vote.py --clobber "$@" ${tmpdir}/majority-vote.mnc
./awol_mrf6.0.py --clobber --brain_image Healthy_Aging/input/atlases/brains/${subject}.mnc "$@" ${tmpdir}/awol-vote6.0.mnc

LabelOverlapMeasures 3 $tmpdir/majority-vote.mnc Healthy_Aging/input/atlases/labels/${subject}_labels.mnc >(tail -n +2) | awk -vT="$numatlas,$numtemplates,majority-vote," '{ print T $0 }' >> $output
LabelOverlapMeasures 3 $tmpdir/awol-vote6.0.mnc Healthy_Aging/input/atlases/labels/${subject}_labels.mnc >(tail -n +2) | awk -vT="$numatlas,$numtemplates,awol-vote6.0," '{ print T $0 }' >> $output

rm -rf $tmpdir

