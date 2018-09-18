#!/usr/bin/env python

import numpy as np
import seaborn as sns; sns.set()
import SimpleITK as sitk

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import glob
    
if __name__ == "__main__":
    parser = ArgumentParser(description="""Get the multiclass confusion heatmap for a test algorithm and a baseline
                            algorithm with respect to some target ground-truth segmentations.""")  
                            
    parser.add_argument("test", type=str, help="The directory with the output from the test algorithms.")
    parser.add_argument("baseline", type=str, nargs="+", 
                        help="The directory(ies) with the output from the baseline algorithm(s).")
    parser.add_argument("target", type=str, help="The directory with the target ground-truth segmentations.")
    parser.add_argument("output", type=str, help="The prefix where the multiclass confusion heatmaps will be saved.")
    parser.add_argument("--label_map", dest="label_map", type=str, nargs="+", default=[],
                        help="Set the names of structural labels in ascending order of their assigned int values.")
    
    opt = parser.parse_args()
    
    test_labels = []
    baseline_labels = []
    target_labels = []
    
    # Get manual labels
    for filename in glob.glob(opt.target + "/*"):
        print("manual labels detected")
        label_image = sitk.ReadImage(filename)
        label_array = np.array(sitk.GetArrayFromImage(label_image), dtype=np.int16)
        target_labels.append(label_array.ravel())
        
    label_values = np.unique(target_labels[0])
    nlabels = label_values.shape[0]
    cmatrix_test = np.zeros((nlabels, nlabels), dtype=np.int64)
    
    print(cmatrix_test)
        
    # Get test algorithm labels
    for filename in glob.glob(opt.test + "/*"):
        print("test algorithm labels detected")
        label_image = sitk.ReadImage(filename)
        label_array = np.array(sitk.GetArrayFromImage(label_image), dtype=np.int16)
        test_labels.append(label_array.ravel())
        
    target_labels = np.asarray(target_labels)
    test_labels = np.asarray(test_labels)
        
    print(target_labels)
    print(test_labels)
    
    # Get the multiclass confusion matrix for the test algorithm
    for i in range(nlabels):
        for j in range(nlabels):
            print(np.where(target_labels == i))
            print(np.where(test_labels == j))
            cmatrix_test[i][j] = np.where((target_labels == i) & (test_labels == j))[0].shape[0]
            
    print(cmatrix_test)
        
    for baseline in opt.baseline:
        cmatrix_baseline = np.zeros((nlabels, nlabels), dtype=np.int64)        
        
        # Get baseline algorithm labels
        for filename in glob.glob(baseline + "/*"):
            print("baseline algorithm labels detected")
            label_image = sitk.ReadImage(filename)
            label_array = np.array(sitk.GetArrayFromImage(label_image), dtype=np.int16)
            baseline_labels.append(label_array.ravel())
            
        baseline_labels = np.asarray(baseline_labels)
        
        # Get the multiclass confusion matrix for the baseline algorithm
        for i in range(nlabels):
            for j in range(nlabels):
                print(np.where(target_labels == i))
                print(np.where(baseline_labels == j))
                cmatrix_baseline[i][j] = np.where((target_labels == i) & (baseline_labels == j))[0].shape[0]
        
        print(cmatrix_test)
        
        label_size = np.sum(cmatrix_test, axis=1, dtype=np.float64)
        cmatrix_diff = np.transpose(cmatrix_test - cmatrix_baseline)
        cmatrix_combined = np.transpose(cmatrix_diff / label_size)
        
        print(cmatrix_diff)
        print(cmatrix_combined)
        
        mc_heatmap = sns.heatmap(cmatrix_combined, center=0, cmap=sns.diverging_palette(240, 15, as_cmap=True),
                                 xticklabels=opt.label_map, yticklabels=opt.label_map)
        mc_heatmap.set_xlabel("Manual Labels")
        mc_heatmap.set_ylabel("PUB-MRF - {}".format(baseline.replace("_", " ")))
        
        fig = mc_heatmap.get_figure()
        fig.savefig(opt.output + baseline + ".png")
        
        baseline_labels = []