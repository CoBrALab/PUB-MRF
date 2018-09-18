#!/bin/bash

dataset=$1
algorithm=$2
output=$3

#echo "FileName,NumAtlas,NumTemplate,Label,Total/Target,Jaccard,Dice,VolumeSimilarity,FalseNegative,FalsePositive" > $output

for file in $dataset/output/fusion/$algorithm/*HA0261*
do
	IFS='-' read -ra file_info <<< "$(basename "$file")"
	numatlas=${file_info[0]}
	numtemplates=${file_info[1]}
	subject=${file_info[2]}

	LabelOverlapMeasures 3 $file $dataset/input/atlases/labels/${subject}-t2_labels.mnc >(tail -n +2) | awk -vT="$(basename $file),$numatlas,$numtemplates," '{ print T $0 }' >> $output
	echo $(wc -l $output)
done
