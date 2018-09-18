#!/usr/bin/env python

from random import shuffle

#List of scans in the ADNI dataset
data = ["ADNI001_t1", "ADNI002_t1", "ADNI003_t1", "ADNI004_t1", "ADNI005_t1", 
        "ADNI006_t1", "ADNI007_t1", "ADNI008_t1", "ADNI009_t1", "ADNI010_t1",
        "ADNI011_t1", "ADNI012_t1", "ADNI013_t1", "ADNI014_t1", "ADNI015_t1",
        "ADNI016_t1", "ADNI017_t1", "ADNI018_t1", "ADNI019_t1", "ADNI020_t1",
        "ADNI021_t1", "ADNI022_t1", "ADNI023_t1", "ADNI024_t1", "ADNI025_t1",
        "ADNI026_t1", "ADNI027_t1", "ADNI028_t1", "ADNI029_t1", "ADNI030_t1",
        "ADNI031_t1", "ADNI032_t1", "ADNI033_t1", "ADNI034_t1", "ADNI035_t1",
        "ADNI036_t1", "ADNI037_t1", "ADNI038_t1", "ADNI039_t1", "ADNI040_t1",
        "ADNI041_t1", "ADNI042_t1", "ADNI043_t1", "ADNI044_t1", "ADNI045_t1",
        "ADNI046_t1", "ADNI047_t1", "ADNI048_t1", "ADNI049_t1", "ADNI050_t1",
        "ADNI051_t1", "ADNI052_t1", "ADNI053_t1", "ADNI054_t1", "ADNI055_t1",
        "ADNI056_t1", "ADNI057_t1", "ADNI058_t1", "ADNI059_t1", "ADNI060_t1",]

max_iter = 3
atl_config = [1, 3, 5, 7, 9]
tmpl_config = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
input_dir = "ADNI_Pruessner/output/intermediate/"
brain_dir = "ADNI_Pruessner/input/atlases/brains/"
resampled_dir = "ADNI_Pruessner/input/atlases/resampled/"
output_dir = "ADNI_Pruessner/output/fusion/"
template_label_dir = "ADNI_Pruessner/output/labels/"
joblist = "joblist_ADNI"
joblist_jlf = "joblist_JLF_ADNI"
joblist_staple = "joblist_STAPLE_ADNI"
mar_info = "mar_info_ADNI"
maget_info = "maget_info_ADNI"
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
                    f.write("./run_jlf.sh ADNI_Pruessner "+output_dir+"JLF/"+filename+".nii.gz"+" "+subj+" "+
                            str(n_atl)+" "+" ".join(brain_files)+" "+" ".join(label_files)+"\n")
                    
                #Print the command line for STAPLE
                label_staple_files = [template_label_dir + atl + "/" + subj + "/labels" + ext for atl in atlases]
                
                with open(joblist_staple, "a") as f:
                    f.write("./run_staple.sh "+output_dir+"STAPLE/"+filename+ext+" "+" ".join(label_staple_files)+"\n")
                    
                #Keep the trial info for future algorithms
                with open(mar_info, "a") as f:
                    f.write("./new_script "+subj+" "+str(n_atl)+" "+" ".join(atlases)+"\n")
            
            for n_tmpl in tmpl_config:
                if n_atl + n_tmpl >= 3: #Exclude 1x1 and 2x1 trials
                    
                    #Select the templates                    
                    shuffle(other_scans)
                    templates = other_scans[:n_tmpl]
                    
                    #Exclude intermediate files with registration error                    
                    if subj == "ADNI002_t1" and "ADNI031_t1" in atlases and "ADNI014_t1" in templates:
                        templates.remove("ADNI014_t1")
                        templates.append(other_scans[n_tmpl])
    
                    if subj == "ADNI010_t1" and "ADNI026_t1" in atlases and "ADNI058_t1" in templates:
                        templates.remove("ADNI058_t1")
                        templates.append(other_scans[n_tmpl])
                    
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