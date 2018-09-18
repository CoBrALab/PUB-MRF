#!/usr/bin/env python

from random import shuffle

atlases = ["HA0001-t2", "HA0002-t2", "HA0004-t2", "HA0019-t2", "HA0033-t2",
           "HA0036-t2", "HA0070-t2", "HA0095-t2", "HA0127-t2", "HA0261-t2"]
templates = ["HA0005-t2", "HA0006-t2", "HA0007-t2", "HA0008-t2", "HA0011-t2", 
             "HA0013-t2", "HA0016-t2", "HA0018-t2", "HA0024-t2", "HA0025-t2", 
             "HA0026-t2", "HA0028-t2", "HA0029-t2", "HA0030-t2", "HA0031-t2",
             "HA0032-t2", "HA0035-t2", "HA0038-t2", "HA0047-t2", "HA0051-t2", 
             "HA0056-t2", "HA0059-t2", "HA0060-t2", "HA0061-t2", "HA0064-t2", 
             "HA0065-t2", "HA0066-t2", "HA0067-t2", "HA0068-t2", "HA0074-t2", 
             "HA0082-t2", "HA0083-t2", "HA0085-t2", "HA0087-t2", "HA0088-t2", 
             "HA0089-t2", "HA0090-t2", "HA0091-t2", "HA0105-t2", "HA0110-t2", 
             "HA0111-t2", "HA0114-t2", "HA0122-t2", "HA0126-t2", "HA0128-t2",
             "HA0130-t2", "HA0132-t2", "HA0133-t2", "HA0137-t2", "HA0138-t2", 
             "HA0143-t2", "HA0150-t2", "HA0158-t2", "HA0160-t2", "HA0163-t2", 
             "HA0173-t2", "HA0174-t2", "HA0176-t2", "HA0186-t2", "HA0192-t2", 
             "HA0206-t2", "HA0209-t2", "HA0213-t2", "HA0216-t2", "HA0217-t2", 
             "HA0222-t2", "HA0225-t2", "HA0226-t2", "HA0238-t2", "HA0244-t2", 
             "HA0246-t2", "HA0247-t2", "HA0248-t2", "HA0251-t2", "HA0253-t2", 
             "HA0256-t2", "HA0258-t2", "HA0259-t2", "HA0260-t2", "HA0264-t2", 
             "HA0265-t2", "HA0268-t2", "HA0272-t2"]
subjects = ["HA0001-t2", "HA0002-t2", "HA0004-t2", "HA0019-t2", "HA0033-t2",
            "HA0036-t2", "HA0070-t2", "HA0095-t2", "HA0127-t2", "HA0261-t2"] 

max_iter = 10
atl_config = [1, 3, 5, 7, 9]
tmpl_config = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
input_dir = "Healthy_Aging/output/intermediate/"
output = "output_HA.csv"
extension = ".mnc"
bash_script = "collect_votes_HA.sh"

input_dir = "Healthy_Aging/output/intermediate/"
brain_dir = "Healthy_Aging/input/atlases/brains/"
resampled_dir = "Healthy_Aging/input/atlases/resampled/"
output_dir = "Healthy_Aging/output/fusion/"
template_label_dir = "Healthy_Aging/output/labels/"
joblist = "joblist_HA"
joblist_jlf = "joblist_JLF_HA"
mar_info = "mar_info_HA"
maget_info = "maget_info_HA"
ext = ".mnc"


for i, subj in enumerate(subjects):
    atlas_copy = list(atlases)
    del atlas_copy[i] #Exclude the subject from the atlas and template library
    
    for n_iter in range(max_iter):
        for n_atl in atl_config:
            
            #Select the atlases            
            shuffle(atlas_copy)            
            atlas_lib = atlas_copy[:n_atl]
            
            if n_atl > 1:
                filename = str(n_atl) + "-" + subj + "-" + str(n_iter)                
                
                #Print the command line for JLF
                brain_files = [resampled_dir + atl + "." + subj + ".nii" for atl in atlas_lib]
                label_files = [resampled_dir + atl + "_labels." + subj + ".nii" for atl in atlas_lib]
                
                with open(joblist_jlf, "a") as f:
                    f.write("./run_jlf.sh Healthy_Aging "+output_dir+"JLF/"+filename+".nii.gz"+" "+subj+" "+
                            str(n_atl)+" "+" ".join(brain_files)+" "+" ".join(label_files)+"\n")
                    
                #Keep the trial info for future algorithms
                with open(mar_info, "a") as f:
                    f.write("./new_script "+subj+" "+str(n_atl)+" "+" ".join(atlases)+"\n")
            
            for n_tmpl in tmpl_config:
                if n_atl + n_tmpl >= 3: #Exclude 1x1 trials
                    
                    #Select the templates                    
                    shuffle(templates)
                    template_lib = templates[:n_tmpl]
                    
                    #Get the list of candidate segmentations                    
                    input_files = []
                    pairs = []
                    for atl in atlas_lib:
                        for tmpl in template_lib:
                            input_files.append(input_dir + atl + "." + tmpl + "." + subj + "_label" + ext)
                            pairs.append((atl, tmpl))
                            
                    filename = str(n_atl)+"-"+str(n_tmpl)+"-"+subj+"-"+str(n_iter)+ext
                    
                    #Print the command lines for majority vote and PUB-MRF
                    with open(joblist, "a") as f:
                        f.write("./majority_vote.py "+" ".join(input_files)+" "+output_dir+"majority_vote/"+filename+"\n"+
                                "./pub_mrf.py --brain_image "+brain_dir+subj+ext+" "+" ".join(input_files)+" "+output_dir+"PUB-MRF/"+filename+"\n")
                                
                    with open(maget_info, "a") as f:
                        f.write("./new_script "+subj+" "+str(n_atl)+" "+str(n_tmpl)+" "+" ".join(atlases)+" "+" ".join(templates)+"\n")