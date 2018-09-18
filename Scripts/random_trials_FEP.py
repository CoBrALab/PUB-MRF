#!/usr/bin/env python

from random import shuffle

#List of scans in the FEP dataset
data = ["s1_05_1", "s1_23_1", "s1_34_1", "s1_57_1", "s1_74_1", "s2_03_1", "s2_135_1", 
        "s2_37_1", "s2_54_1", "s2_73_1", "s1_10_1", "s1_26_1", "s1_42_1", "s1_61_1", 
        "s1_88_1", "s2_107_1", "s2_147_1", "s2_40_1", "s2_61_1", "s2_78_1", "s1_13_1",
        "s1_28_1", "s1_47_1", "s1_65_1", "s1_91_1", "s2_133_1", "s2_14_1", "s2_43_1",
        "s2_71_1", "s2_94_1"]

max_iter = 3
atl_config = [1, 3, 5, 7, 9]
tmpl_config = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
input_dir = "Carolina_FEP/output/intermediate/"
brain_dir = "Carolina_FEP/input/atlases/brains/"
resampled_dir = "Carolina_FEP/input/atlases/resampled/"
output_dir = "Carolina_FEP/output/fusion/"
template_label_dir = "Carolina_FEP/output/labels/"
joblist = "joblist_FEP"
joblist_jlf = "joblist_JLF_FEP"
joblist_staple = "joblist_STAPLE_FEP"
mar_info = "mar_info_FEP"
maget_info = "maget_info_FEP"
ext = ".mnc"

for i, subj in enumerate(data):
    data_copy = list(data)
    del data_copy[i] #Exclude the subject from the atlas and template library
    
    for n_iter in range(max_iter):
        for n_atl in atl_config:
            
            #Select the atlases            
            shuffle(data_copy)            
            atlases = data_copy[:n_atl]
            other_scans = data_copy[n_atl:]
            
            if n_atl > 1:
                filename = str(n_atl) + "-" + subj + "-" + str(n_iter)                
                
                #Print the command line for JLF
                brain_files = [resampled_dir + atl + "." + subj + ".nii" for atl in atlases]
                label_files = [resampled_dir + atl + "_labels." + subj + ".nii" for atl in atlases]
                
                with open(joblist_jlf, "a") as f:
                    f.write("./run_jlf.sh Carolina_FEP "+output_dir+"JLF/"+filename+".nii.gz"+" "+subj+" "+
                            str(n_atl)+" "+" ".join(brain_files)+" "+" ".join(label_files)+"\n")
                    
                #Print the command line for STAPLE
                label_staple_files = [template_label_dir + atl + "/" + subj + "/labels" + ext for atl in atlases]
                
                with open(joblist_staple, "a") as f:
                    f.write("./run_staple.sh "+output_dir+"STAPLE/"+filename+ext+" "+" ".join(label_staple_files)+"\n")
                    
                #Keep the trial info for future algorithms
                with open(mar_info, "a") as f:
                    f.write("./new_script "+subj+" "+str(n_atl)+" "+" ".join(atlases)+"\n")
            
            for n_tmpl in tmpl_config:
                if n_atl + n_tmpl >= 3 and n_atl + n_tmpl < 30: #Exclude 1x1 and 9x21 trials
                    
                    #Select the templates                    
                    shuffle(other_scans)
                    templates = other_scans[:n_tmpl]
                    
                    #Get the list of candidate segmentations                    
                    input_files = []
                    pairs = []
                    for atl in atlases:
                        for tmpl in templates:
                            input_files.append(input_dir + atl + "." + tmpl + "." + subj + "_label" + ext)
                            pairs.append((atl, tmpl))
                            
                    filename = str(n_atl)+"-"+str(n_tmpl)+"-"+subj+"-"+str(n_iter)+ext
                    
                    #Print the command lines for majority vote and PUB-MRF
                    with open(joblist, "a") as f:
                        f.write("./majority_vote.py "+" ".join(input_files)+" "+output_dir+"majority-vote/"+filename+"\n"+
                                "./pub_mrf.py --brain_image "+brain_dir+subj+ext+" "+" ".join(input_files)+" "+output_dir+"PUB-MRF/"+filename+"\n")
                                
                    with open(maget_info, "a") as f:
                        f.write("./new_script "+subj+" "+str(n_atl)+" "+str(n_tmpl)+" "+" ".join(atlases)+" "+" ".join(templates)+"\n")