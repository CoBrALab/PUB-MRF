#!/bin/bash
set -euo pipefail

#This is with version 1 and default majority voting mask.

tmpdir=$(mktemp -d)

dataset=$1
output=$2
subject=$3
natlas=$4
brains="${@:5:$natlas}"
labels="${@:5+$natlas:$natlas}"

output=$dataset/output/fusion/JLF/$(basename $output .nii.gz).mnc

cmd="antsJointLabelFusion2.sh -v 1 -d 3 -t $dataset/input/atlases/brains/${subject}.nii -o $tmpdir/output"

for brain in $brains
    do cmd="${cmd} -g $brain"
done

for label in $labels
	do cmd="${cmd} -l $label"
done

$cmd
nii2mnc $tmpdir/outputLabels.nii.gz $tmpdir/output.mnc
mincresample -near -byte -keep -like $dataset/input/atlases/labels/${subject}_labels.mnc $tmpdir/output.mnc $output

rm -rf $tmpdir
