#!/bin/bash

for subj in Healthy_Aging/input/subjects/brains/*
	#do for atl in Healthy_Aging/input/atlases/brains/*
	#	do mincresample -like $subj $atl Healthy_Aging/input/atlases/resampled/$(basename $atl .mnc).$(basename $subj)
	#done
	
	do for label in Healthy_Aging/input/atlases/labels/*
		do mincresample -near -byte -keep -like $subj $label Healthy_Aging/input/atlases/resampled/$(basename $label .mnc).$(basename $subj)
	done
done	

for file in Healthy_Aging/input/atlases/resampled/*
	do mnc2nii $file Healthy_Aging/input/atlases/resampled$(basename $file .mnc).nii
done
