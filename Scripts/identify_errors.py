#!/usr/bin/env python

from random import shuffle

total_nb_of_label_images = 60
atl_config = [1, 3, 5, 7, 9]
tmpl_config = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
input_dir = "ADNI_Pruessner/output/intermediate/"
dataset = "ADNI_Pruessner"
output = "analyze_outcomes.csv"
extension = ".mnc"

label_images = list(range(total_nb_of_label_images + 1))
del label_images[0]
    
for n_atl in atl_config:
    for n_tmpl in tmpl_config:
        shuffle(label_images)
        k = label_images[-1]
        atl = label_images[0:n_atl]
        tmpl = label_images[n_atl:n_atl+n_tmpl]
                 
        input_files = []
        for i in atl:
            for j in tmpl:
                input_files.append(input_dir + "ADNI" + str(i).zfill(3) + "_t1.ADNI" 
                + str(j).zfill(3) + "_t1.ADNI" + str(k).zfill(3) + "_t1_label" + extension)
                        
        print("./identify_errors.sh "+str(n_atl)+" "+str(n_tmpl)+" "+"ADNI"+str(k).zfill(3)+"_t1 "+dataset+" "+output+" "+" ".join(input_files))
        

brain_scans = ["s1_05", "s1_23", "s1_34", "s1_57", "s1_74", "s2_03", "s2_135", "s2_37", "s2_54", "s2_73", "s1_10", "s1_26", "s1_42", "s1_61", "s1_88", "s2_107", "s2_147", "s2_40", "s2_61", 
"s2_78", "s1_13", "s1_28", "s1_47", "s1_65", "s1_91", "s2_133", "s2_14", "s2_43", "s2_71", "s2_94"]
input_dir = "Carolina_FEP/output/intermediate/"
dataset = "Carolina_FEP"

for n_atl in atl_config:
    for n_tmpl in tmpl_config:
        if n_atl != 9 or n_tmpl != 21:
            shuffle(brain_scans)
            k = brain_scans[-1]
            atl = brain_scans[0:n_atl]
            tmpl = brain_scans[n_atl:n_atl+n_tmpl]
				
            input_files = []
            for i in atl:
                for j in tmpl:
                    input_files.append(input_dir + i + "_1." + j + "_1." + k + "_1_label" + extension)
                    
            print("./identify_errors.sh "+str(n_atl)+" "+str(n_tmpl)+" "+k+"_1 "+dataset+" "+output+" "+" ".join(input_files))
            
            
atlases = ["DMHU_HA_0002_T2", "DMHU_HA_0004_T2", "DMHU_HA_0019_T2", "DMHU_HA_0033_T2", "DMHU_HA_0095_T2"]
templates = ["HA0001-t2", "HA0002-t2", "HA0004-t2", "HA0005-t2", "HA0006-t2", 
             "HA0007-t2", "HA0008-t2", "HA0011-t2", "HA0013-t2", "HA0016-t2", 
             "HA0018-t2", "HA0019-t2", "HA0024-t2", "HA0025-t2", "HA0026-t2", 
             "HA0028-t2", "HA0029-t2", "HA0030-t2", "HA0031-t2", "HA0032-t2", 
             "HA0033-t2", "HA0035-t2", "HA0036-t2", "HA0038-t2", "HA0047-t2", 
             "HA0051-t2", "HA0056-t2", "HA0059-t2", "HA0060-t2", "HA0061-t2", 
             "HA0064-t2", "HA0065-t2", "HA0066-t2", "HA0067-t2", "HA0068-t2", 
             "HA0070-t2", "HA0074-t2", "HA0082-t2", "HA0083-t2", "HA0085-t2", 
             "HA0087-t2", "HA0088-t2", "HA0089-t2", "HA0090-t2", "HA0091-t2", 
             "HA0095-t2", "HA0105-t2", "HA0110-t2", "HA0111-t2", "HA0114-t2", 
             "HA0122-t2", "HA0126-t2", "HA0127-t2", "HA0128-t2", "HA0130-t2", 
             "HA0132-t2", "HA0133-t2", "HA0137-t2", "HA0138-t2", "HA0143-t2", 
             "HA0150-t2", "HA0158-t2", "HA0160-t2", "HA0163-t2", "HA0173-t2",
             "HA0174-t2", "HA0176-t2", "HA0186-t2", "HA0192-t2", "HA0206-t2", 
             "HA0209-t2", "HA0213-t2", "HA0216-t2", "HA0217-t2", "HA0222-t2", 
             "HA0225-t2", "HA0226-t2", "HA0238-t2", "HA0244-t2", "HA0246-t2", 
             "HA0247-t2", "HA0248-t2", "HA0251-t2", "HA0253-t2", "HA0256-t2", 
             "HA0258-t2", "HA0259-t2", "HA0260-t2", "HA0261-t2", "HA0264-t2", 
             "HA0265-t2", "HA0268-t2", "HA0272-t2"]
subjects = ["DMHU_HA_0002_T2", "DMHU_HA_0004_T2", "DMHU_HA_0019_T2", "DMHU_HA_0033_T2", "DMHU_HA_0095_T2"] 

atl_config = [1, 2, 3, 4]
input_dir = "Healthy_Aging/output/intermediate/"
dataset = "Healthy_Aging"  
    
for n_atl in atl_config:
    for n_tmpl in tmpl_config:
        shuffle(atlases)
        shuffle(templates)
        k = atlases[-1]
        atl = atlases[:n_atl]
        tmpl = templates[:n_tmpl]
				
        input_files = []
        for i in atl:
            for j in tmpl:
                input_files.append(input_dir + i + "." + j + "." + k + "_label" + extension)
				
        print("./identify_errors.sh "+str(n_atl)+" "+str(n_tmpl)+" "+k+" "+dataset+" "+output+" "+" ".join(input_files))
