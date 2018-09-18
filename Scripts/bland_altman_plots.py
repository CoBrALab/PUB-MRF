#!/usr/bin/env python

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import glob
    
if __name__ == "__main__":
    parser = ArgumentParser(description="""Get the Bland-Altman plots for a set of automated
                            segmentations, with respect to their ground-truth manual segmentation.""")  
                            
    parser.add_argument("manual", type=str, help="The directory with the manual segmentations")
    parser.add_argument("automated", type=str, nargs="+", help="The directory(ies) with the automated segmentations")
    parser.add_argument("output", type=str, help="The directory which will contain the Bland-Altman plots")
    parser.add_argument("--label_map", dest="label_map", type=str, nargs="+", default=[],
                        help="Set the names of structural labels in ascending order of their assigned int values.")
    opt = parser.parse_args()
    
    manual = []
    automated = []
    algorithms = []
    
    # Compute manual volumes
    for filename in glob.glob(opt.manual + "/*labels.mnc"):
        print("manual labels detected")
        man_labels = sitk.ReadImage(filename)
        label_array = np.array(sitk.GetArrayFromImage(man_labels), dtype=np.int64)      
        manual.append(np.bincount(label_array.ravel())) 
        #print(np.bincount(label_array.ravel()))
        
    label_values = np.nonzero(manual[0])[0][1:]
    print(label_values)
    manual = np.asarray(manual)
        
    for input_dir in opt.automated:
        print("automated dir detected")
        
        algorithm = input_dir.split("/")[-2]        
        algorithms.append(algorithm)
        automated.append([])
        
        #Compute automated volumes
        if len(glob.glob(input_dir + "/9-19*0.mnc")) > 0:
            # If the algorithm uses templates
            for filename in glob.glob(input_dir + "/9-19*0.mnc"):
                print("automated labels in {} detected".format(input_dir))
                auto_labels = sitk.ReadImage(filename)
                label_array = np.array(sitk.GetArrayFromImage(auto_labels), dtype=np.int64)
                #print(np.bincount(label_array.ravel()))
                automated[-1].append(np.bincount(label_array.ravel()))
                
        else:
            # If the algorithm only uses atlases
            for filename in glob.glob(input_dir + "/9*0_debug.mnc"):
                print("automated labels in {} detected".format(input_dir))
                auto_labels = sitk.ReadImage(filename)
                label_array = np.array(sitk.GetArrayFromImage(auto_labels), dtype=np.float64)
                
                label_array[np.where(np.around(label_array, decimals=8) == 0.87058824)] = 1
                print(np.where(np.around(label_array, decimals=8) == 0.87058824))
                label_array[np.where(np.around(label_array, decimals=8) == 1.74117649)] = 2
                label_array[np.where(np.around(label_array, decimals=8) == 21.76470566)] = 22
                label_array[np.where(np.around(label_array, decimals=8) == 34.82352829)] = 35
                label_array[np.where(np.around(label_array, decimals=8) == 100.98823547)] = 101
                label_array[np.where(np.around(label_array, decimals=8) == 101.85882568)] = 102
                label_array[np.where(np.around(label_array, decimals=8) == 103.59999847)] = 104
                
                print(np.unique(label_array))
                label_array = np.array(label_array, dtype=np.int64)
                print(np.unique(label_array))
                #print(np.bincount(label_array.ravel()))
                automated[-1].append(np.bincount(label_array.ravel()))
                
                
    
    automated = np.asarray(automated)
    #print(automated)
    #print(manual)
    #print(algorithms)
    print(opt.label_map)
        
    colormap = ['b', 'g', 'r', 'c']
            
    for i, value in enumerate(label_values):      # Get the Bland-Altman plot for each label
        man = manual[:, value]
        auto = automated[:, :, value]
        print("label value {} detected".format(value))
        #print(auto)
        #print(man)
        fig = plt.figure()
        
        for j, computed in enumerate(auto):
            print("plotting curve for algorithm")
            mean_volume = (man + computed) / 2
            diff_volume = man - computed
    
            md = np.mean(diff_volume)           # Mean of the difference
            sd = np.std(diff_volume)            # Standard deviation of the difference
            m, b = np.polyfit(mean_volume, diff_volume, 1)    # Fit with np.polyfit
        
            plt.scatter(mean_volume, diff_volume, c=colormap[j])
            line, = plt.plot(mean_volume, m*mean_volume + b, '-', c=colormap[j])
            
            if j == 0:
                line.set_label("Majority Vote")
            elif j == 1:
                line.set_label("JLF")
            elif j == 2:
                line.set_label("PUB-MRF")
            
            plt.axhline(md, linestyle='--', c=colormap[j])
            plt.axhline(md + 2*sd, linestyle=':', c=colormap[j])
            plt.axhline(md - 2*sd, linestyle=':', c=colormap[j])    
            plt.xlabel('0.5*(Manual + Computed)')
            plt.ylabel('(Manual - Computed)')
            
            print(j, "Volume underestimation: mean", md, "Volume underestimation: sd", sd, 
                  "Proportional bias:", m)
        
        lgd = plt.legend(bbox_to_anchor=(0.45, -0.2), loc="center", ncol=3, borderaxespad=0.)
        
        if len(opt.label_map) > 0:
            plt.title(opt.label_map[i])
            
        fig.set_size_inches(6, 4.5)
        plt.savefig(opt.output + "_BAplot_" + str(value) + ".png",
                    bbox_extra_artists=(lgd,), bbox_inches="tight")
        
        plt.clf()
