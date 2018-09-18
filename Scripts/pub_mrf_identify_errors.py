#!/usr/bin/env python

import numpy as np
import SimpleITK as sitk

from argparse import ArgumentParser, ArgumentTypeError
from warnings import warn
import os.path
import sys

import csv

class AWoL_MRF:
    """The AWoL-MRF algorithm organizes the low-confidence voxels in patches. 
    In each patch, the labels for these voxels are updated in a sequence given
    by Prim's algorithm using the Markov Random Field potentials.
    
    Key features of this version:
    - Works with a single structural label
    - Works with 2 or more separate structural labels
    - Assumes the background label is 0 (or smaller than any structural label)
    - Assumes strictly positive integer values for the structural labels
    - Compatible with a Python notebook, see AWoL-MRF_on_notebook.ipynb
    - Uses smart bounding boxes and label counting to reduce peak memory usage
    - Defines the candidate labels for each voxel based on the MAGeT votes
    - Uses dymanic thresholds which depend on the number of candidate labels
    - Patch stats are weighted based on the MAGeT confidence level
    - Uses MRF to update the prior label probabilities from the MAGeT votes
    - Has an option to save the singleton, prior and doubleton potential maps
    - Doubleton potential is distance-weighted with the 26-voxel neighborhood
    - Output does not depend on the MRF updating sequence"""
    
    def __init__(self, labelimg_list, brainimg, majority_labels, manual_labels, n_atlases, n_templates, output_data,
                 dataset, bbox=None, alpha=2.0, beta=2.7, patch_length=5, threshold=0.2, potential_maps=False):
                
        def positive_int(x): #avoid nonsense negative parameter values   
            x = int(x)
            if x < 0:
                raise AssertionError("%r is not a positive int"%(x,))
            return x
            
        def restricted_float(x): #avoid nonsense values for the threshold
            x = float(x)
            if x < 0.0 or x > 1.0:
                raise AssertionError("%r not in range [0.0, 0.5]"%(x,))
            return x
            
        #catch invalid parameters
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.patch_length = positive_int(patch_length)
        self.threshold = restricted_float(threshold)
        self.potential_maps = bool(potential_maps)
        
        self.majority_labels = sitk.GetArrayFromImage(sitk.ReadImage(majority_labels))
        self.manual_labels = sitk.GetArrayFromImage(sitk.ReadImage(manual_labels))
        self.n_atlases = positive_int(n_atlases)
        self.n_templates = positive_int(n_templates)
        self.output_data = str(output_data)
        self.dataset = str(dataset)

        if bbox is not None:        
            bbox[:3] -= self.patch_length #pad the bounding box with the patch length
            bbox[3:] += self.patch_length
                       
        for n, img in enumerate(labelimg_list):
            if bbox is None:
                label_array = sitk.GetArrayFromImage(img) #get the label array from each image
            else: #get each array within the bounding box
                label_array = sitk.GetArrayFromImage(img)[bbox[2]:bbox[5], bbox[1]:bbox[4], bbox[0]:bbox[3]]
                
            if n == 0:
                self.label_values = np.unique(label_array) #obtain the list of labels
                self.label_shape = label_array.shape
                votes = np.zeros((self.label_values.shape[0], label_array.ravel().shape[0]), dtype=np.float32)
            
            elif np.asarray(np.unique(label_array) != self.label_values).any(): #check label equivalence
                warn("Labels in image {} not the same as in image 1.".format(n))
            
            for i, value in enumerate(self.label_values):
                votes[i][np.where(label_array.ravel() == value)] += 1 #count the votes for each label
        
        if bbox is None:
            self.intensity = sitk.GetArrayFromImage(brainimg).ravel() #array of intensities
        else:
            self.intensity = sitk.GetArrayFromImage(brainimg)[bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]].ravel()
            
        self.bbox = bbox #keep the bounding box for the final fusion labels
        self.brainimg = brainimg #keep this to copy the metadata to the output image
        
        self.probability = np.array(votes / len(labelimg_list), dtype=np.float32) #get the initial probabilities
        self.new_probability = np.copy(self.probability)
            
    def run(self):
        """Find the seeds. For each seed, get the minimum spanning tree
        sequence, and compute the MRF potentials in that order. Then compute
        the final fusion labels."""
             
        self.find_lcv()
        
        self.lcv_index = 0        
        while self.lcv_index < self.lcv.shape[0]:
            self.get_patch_stats()
            if len(self.patch_stats) > 0:            
                self.mrf_potentials()
            self.lcv_index += 1
    
        if self.potential_maps:
            self.get_potential_maps()
        
        return self.get_output_image()
        
    def find_lcv(self):
        """Find the low-confidence voxels. Also initialize the 6-voxel
        neighborhood and the 26-voxel neighborhood for each low-confidence
        voxel. Also computes the confidence level of each voxel."""    
             
        #find the low-confidence voxels using dynamic thresholds
        n_labels = np.sum(self.probability > 0, axis=0)
        self.lcv = np.where((n_labels > 1) & (np.amax(self.probability, axis=0) < 1.0/n_labels + self.threshold))[0]
        
        if self.lcv.shape[0] == 0: #in this case we just want to return the majority vote
            self.no_lcv = True
            warn("No low-confidence voxel was found.")
            
        else:
            self.no_lcv = False
                
            self.neighbors = {}
            self.patches = {}
            self.candidate_labels = {}
            
            for lcv in self.lcv: #find the neighborhood of each lcv
                x, y, z = np.unravel_index(lcv, self.label_shape) #get coordinates of voxels
                
                neighbors = []                
                for i in range(x-1, x+2):
                    for j in range(y-1, y+2):
                        for k in range(z-1, z+2):
                            neighbors.append(np.ravel_multi_index((i,j,k), self.label_shape))
                self.neighbors[lcv] = neighbors #26-voxel neighborhood
                
                patch = []                
                for i in range(x-self.patch_length, x+self.patch_length+1):
                    for j in range(y-self.patch_length, y+self.patch_length+1):
                        for k in range(z-self.patch_length, z+self.patch_length+1):
                            patch.append(np.ravel_multi_index((i,j,k), self.label_shape))
                self.patches[lcv] = patch #patch neighborhood
                
                #get the candidate labels for each low-confidence voxel
                self.candidate_labels[lcv] = self.label_values[self.probability[:, lcv].nonzero()]
        
        #prepare the arrays that will keep the potential maps for all the low-confidence voxels
        if self.potential_maps:
            self.singleton = np.zeros((self.label_values.shape[0], self.lcv.shape[0]), dtype=np.float32) - 10.0
            self.prior = np.zeros((self.label_values.shape[0], self.lcv.shape[0]), dtype=np.float32) - 10.0
            self.doubleton = np.zeros((self.label_values.shape[0], self.lcv.shape[0]), dtype=np.float32) - 10.0
            self.energy = np.zeros((self.label_values.shape[0], self.lcv.shape[0]), dtype=np.float32) - 10.0
            
        if not os.path.exists(self.output_data):
            with open(self.output_data, 'a') as csvfile:
                filewriter = csv.writer(csvfile)
                filewriter.writerow(["Dataset", "NumAtlas", "NumTemplates", "z-coord", "y-coord", "x-coord", 
                                     "ManLabels", "MajLabels", "AwolLabels", "InitialProb", "MajOutcome", 
                                     "AwolOutcome", "Intensity", "Label", "Mean", "Std", "Singleton", "Doubleton"])
                
    def get_patch_stats(self):
        """Compute the patch stats for the current low-confidence voxels
        using mean and standard deviation. These stats are used to compute the
        MRF singleton potentials."""
        
        self.patch_stats = []
        patch = np.asarray(self.patches[self.lcv[self.lcv_index]])
        probability = self.probability[:, patch]
        sorted_prob = np.sort(probability, axis=0)[::-1] #sort in descending order
            
        #get the patch stats for each candidate label
        for value in self.candidate_labels[self.lcv[self.lcv_index]]:
            intensity = self.intensity[patch]
            value_index = np.where(self.label_values == value)
            weight = np.equal(probability[value_index], sorted_prob[0]) * (sorted_prob[0] - sorted_prob[1])
            
            if np.sum(weight) > 0.0:
                mean = np.sum(intensity * weight) / np.sum(weight) #weighted mean
                std = np.sqrt(np.sum(np.power(intensity-mean,2)*weight)/np.sum(weight)) #weighted std
                
                if std != 0.0: #a standard deviation of 0 doesn't make sense for the singleton computation
                    self.patch_stats.append([value, mean, std])
                    
        self.patch_stats = np.asarray(self.patch_stats)
        del self.patches[self.lcv[self.lcv_index]]
        
    def mrf_potentials(self):
        """Compute the Markov Random Field potential for the patch in the
        order given by the minimum spanning tree sequence. The weight of the
        doubleton potentials is determined by the beta patameter."""
        
        lcv = self.lcv[self.lcv_index]
        n = self.neighbors[lcv]
        lcv_coord = np.unravel_index(lcv, self.label_shape)
        
        #compute the alpha hyperparameter        
        #new_alpha = self.alpha * np.amax(self.patch_stats[:, 2]) / np.amin(self.patch_stats[:, 2])                
        
        weight = []
        for neighbor in n: #compute the weights of the neighbors
            neighbor_coord = np.unravel_index(neighbor, self.label_shape)
            #weight.append(new_alpha*np.exp(-self.beta*np.linalg.norm(np.subtract(neighbor_coord, lcv_coord))))
            weight.append(self.alpha*np.exp(-self.beta*np.linalg.norm(np.subtract(neighbor_coord, lcv_coord))))
        weight = np.asarray(weight)
        
        label_info = []        
        
        mrf_energy = np.zeros(self.label_values.shape) + np.inf            
        for i in range(len(self.patch_stats)): #compute doubleton and singleton potentials
            (value, mean, std) = self.patch_stats[i]
            mrf_single = (np.log(np.sqrt(2*np.pi)*std)) + (np.power(self.intensity[lcv]-mean,2))/(2*np.power(std,2))
            mrf_double = np.dot(1.0 - self.probability[np.where(self.label_values==value),n][0], weight)
            mrf_energy[np.where(self.label_values==value)] = mrf_single + mrf_double
            
            label_info.extend([value, mean, std, mrf_single, mrf_double])
            
            if self.potential_maps: #update the potential map arrays
                self.singleton[np.where(self.label_values == value), self.lcv_index] = mrf_single
                self.doubleton[np.where(self.label_values == value), self.lcv_index] = mrf_double
            
        if self.potential_maps:
            self.energy[:, self.lcv_index] = np.where(mrf_energy == np.inf, -10.0, mrf_energy)
            
        if np.sum(np.exp(-mrf_energy)) > 0:
            self.new_probability[:, lcv] = np.exp(-mrf_energy) / np.sum(np.exp(-mrf_energy)) #update the probabilities
            
        #keep the information for this voxel
        c = [lcv_coord[0]+self.bbox[2], lcv_coord[1]+self.bbox[1], lcv_coord[2]+self.bbox[0]]
        manual = int(self.manual_labels[c[0], c[1], c[2]])
        majority = int(self.majority_labels[c[0], c[1], c[2]])
        awol = int(self.label_values[np.argmin(mrf_energy)])
        init_prob = self.probability[np.where(self.label_values == awol), lcv][0][0]
        
        if manual == 0:
            if awol == 0:
                outcome = "TN"
            else:
                outcome = "FP"
            if majority == 0:
                maj_outcome = "TN"
            else:
                maj_outcome = "FP"
        else:
            if awol == 0:
                outcome = "FN"
            else:
                outcome = "TP"
            if majority == 0:
                maj_outcome = "FN"
            else:
                maj_outcome = "TP"
                
        intensity = self.intensity[lcv]
        
        
        with open(self.output_data, 'a') as csvfile:
            filewriter = csv.writer(csvfile)
            columns = [self.dataset, self.n_atlases, self.n_templates, c[0], c[1], c[2], manual, 
                       majority, awol, init_prob, maj_outcome, outcome, intensity]
            for elt in label_info:
                columns.append(elt)
            filewriter.writerow(columns)
        
        del self.candidate_labels[lcv], self.neighbors[lcv], self.patch_stats
        
    def get_output_image(self):
        """Return the final AWoL-MRF output image with the final fusion label
        for each voxel."""
        
        labels = np.zeros(self.label_shape, dtype=np.uint8)
        mode_arg = np.argmax(self.new_probability, axis=0).reshape(self.label_shape)
              
        for i, value in enumerate(self.label_values): #assign the labels with maximum probability
            labels[np.where(mode_arg == i)] = value
        
        #pad the bounding box with background labels
        if self.bbox is not None:        
            labels = np.pad(labels, ((self.bbox[2], self.brainimg.GetDepth() - self.bbox[5]),
                            (self.bbox[1], self.brainimg.GetHeight() - self.bbox[4]),
                            (self.bbox[0], self.brainimg.GetWidth() - self.bbox[3])),
                            "constant", constant_values=0)
        
        #get the output SimpleITK image with fusion labels        
        output_image = sitk.GetImageFromArray(labels)
        output_image.CopyInformation(self.brainimg) #copy the metadata
        
        return output_image
        
    def get_potential_maps(self):
        """Get the singleton, prior and doubleton potential maps for each label."""
        
        self.potentials = {}
        
        for value in self.label_values:
            singleton_map = np.zeros(self.probability.shape[1], dtype=np.float32) - 10.0 #get the singleton map
            singleton_map[self.lcv] = self.singleton[np.where(self.label_values == value)][0]
            
            singleton_map = singleton_map.reshape(self.label_shape)
            singleton_map = np.pad(singleton_map, ((self.bbox[2], self.brainimg.GetDepth() - self.bbox[5]),
                            (self.bbox[1], self.brainimg.GetHeight() - self.bbox[4]),
                            (self.bbox[0], self.brainimg.GetWidth() - self.bbox[3])),
                            "constant", constant_values=-10.0)
                            
            singleton_image = sitk.GetImageFromArray(singleton_map)
            singleton_image.CopyInformation(self.brainimg) #copy the metadata
            self.potentials["singleton_" + str(int(value))] = singleton_image
            
            del singleton_map, singleton_image
            
            doubleton_map = np.zeros(self.probability.shape[1], dtype=np.float32) - 10.0 #get the doubleton map
            doubleton_map[self.lcv] = self.doubleton[np.where(self.label_values == value)][0]
            
            doubleton_map = doubleton_map.reshape(self.label_shape)            
            doubleton_map = np.pad(doubleton_map, ((self.bbox[2], self.brainimg.GetDepth() - self.bbox[5]),
                            (self.bbox[1], self.brainimg.GetHeight() - self.bbox[4]),
                            (self.bbox[0], self.brainimg.GetWidth() - self.bbox[3])),
                            "constant", constant_values=-10.0)
                            
            doubleton_image = sitk.GetImageFromArray(doubleton_map)
            doubleton_image.CopyInformation(self.brainimg) #copy the metadata
            self.potentials["doubleton_" + str(int(value))] = doubleton_image
            
            del doubleton_map, doubleton_image
            
            energy_map = np.zeros(self.probability.shape[1], dtype=np.float32) - 10.0 #get the doubleton map
            energy_map[self.lcv] = self.energy[np.where(self.label_values == value)][0]
            
            energy_map = energy_map.reshape(self.label_shape)            
            energy_map = np.pad(energy_map, ((self.bbox[2], self.brainimg.GetDepth() - self.bbox[5]),
                            (self.bbox[1], self.brainimg.GetHeight() - self.bbox[4]),
                            (self.bbox[0], self.brainimg.GetWidth() - self.bbox[3])),
                            "constant", constant_values=-10.0)
                            
            energy_image = sitk.GetImageFromArray(energy_map)
            energy_image.CopyInformation(self.brainimg) #copy the metadata
            self.potentials["energy_" + str(int(value))] = energy_image
            
            del energy_map, energy_image
                

if __name__ == "__main__":
    #AWoL-MRF parameters
    def positive_int(x): #avoid nonsense negative parameter values   
        x = int(x)
        if x < 0:
            raise ArgumentTypeError("%r is not a positive int"%(x,))
        return x
            
    def restricted_float(x): #avoid nonsense values for the threshold
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
        return x
    
    parser = ArgumentParser(description="""The AWoL-MRF algorithm organizes the low-confidence voxels in patches. 
                            In each patch, the labels for these voxels are updated in a sequence given
                            by Prim's algorithm using the Markov Random Field potentials.
                            
                            Key features of this version:
                            - Works with a single structural label
                            - Works with 2 or more separate structural labels
                            - Assumes the background label is 0 (or smaller than any structural label)
                            - Assumes strictly positive integer values for the structural labels
                            - Compatible with a Python notebook, see AWoL-MRF_on_notebook.ipynb
                            - Uses smart bounding boxes and label counting to reduce peak memory usage
                            - Defines the candidate labels for each voxel based on the MAGeT votes
                            - Uses MRF to update the prior label probabilities from the MAGeT votes
                            - Has an option to save the singleton, prior and doubleton potential maps
                            - Doubleton potential is distance-weighted with the 26-voxel neighborhood
                            - Output does not depend on the MRF updating sequence""")  
                            
    pg = parser.add_argument_group("AWoL-MRF parameters")
    pg.add_argument("-a", "--alpha", type=float, default=1.0,
                    help="[default = %(default)s]")
    pg.add_argument("-b", "--beta", type=float, default=2.7,
                    help="[default = %(default)s]")
    pg.add_argument("-p", "--patch_length", type=positive_int, default=5,
                    help="[default = %(default)s]")
    pg.add_argument("-t", "--threshold", type=restricted_float, default=0.2,
                    help="""[default = %(default)s]""")
                    
    #file manipulation arguments
    parser.add_argument("input_labels", nargs="+", type=str)
    parser.add_argument("--brain_image", type=str, required=True,
                        help="brain intensity image, required argument") #need this for the singleton potentials
    parser.add_argument("output_labels", type=str)
    
    cg = parser.add_mutually_exclusive_group()
    cg.add_argument("--clobber", dest="clobber", action="store_true",
                   help="clobber output file [default = %(default)s]")
    cg.add_argument("--no-clobber", dest="clobber", action="store_false",
                   help="opposite of '--clobber'")
    cg.set_defaults(clobber=False)
    parser.add_argument("--potential_maps", action="store_true", default=False,
                    help="keep the MRF potential maps")
                    
    #additional arguments to keep patch-wise information
    ag = parser.add_argument_group("Additional information:")
    ag.add_argument("--majority_labels", type=str, default=None)
    ag.add_argument("--manual_labels", type=str, default=None)
    ag.add_argument("--n_atlases", type=positive_int, default=None)
    ag.add_argument("--n_templates", type=positive_int, default=None)
    ag.add_argument("--output_data", type=str, default=None)
    ag.add_argument("--dataset", type=str, default=None)

    opt = parser.parse_args()

    if not(opt.clobber) and os.path.exists(opt.output_labels):
        sys.exit("Output file already exists; use --clobber to overwrite.")
    
    #load volumes from input files    
    labelimg_list = [] #list of candidate segmentation images
    
    #use this to verify if the voxel-wise computations make sense    
    def check_metadata(img, metadata, filename):
        if img.GetSize() != metadata["size"]:
            sys.exit("Size of {0} not the same as {1}".format(filename, opt.input_labels[0]))
        elif map(lambda x: round(x, 4), img.GetOrigin()) != metadata["origin"]:
            sys.exit("Origin of {0} not the same as {1}".format(filename, opt.input_labels[0]))
        elif img.GetSpacing() != metadata["spacing"]:
            sys.exit("Spacing of {0} not the same as {1}".format(filename, opt.input_labels[0]))
        elif img.GetDirection() != metadata["direction"]:
            sys.exit("Direction of {0} not the same as {1}".format(filename, opt.input_labels[0]))
    
    for filename in opt.input_labels:
        labelimg = sitk.ReadImage(filename) #get all the candidate segmentations
        
        structure = labelimg > 0 #find the structural voxels
        label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
        label_shape_analysis.SetBackgroundValue(0)
        label_shape_analysis.Execute(structure)
        b = label_shape_analysis.GetBoundingBox(1) #get the bounding box        
        
        if len(labelimg_list) == 0:        
            metadata = {} #get the metadata of the first image
            metadata["size"] = labelimg.GetSize()
            metadata["origin"] = map(lambda x: round(x, 4), labelimg.GetOrigin())
            metadata["spacing"] = labelimg.GetSpacing()
            metadata["direction"] = labelimg.GetDirection()
            
            #get the first bounding box
            bbox = [b[0], b[1], b[2], b[0]+b[3], b[1]+b[4], b[2]+b[5]]
                
        else: #check that the metadata is the same for each other image
            check_metadata(labelimg, metadata, filename)
            
            new_bbox = (b[0], b[1], b[2], b[0]+b[3], b[1]+b[4], b[2]+b[5])   
            for i in range(0,3): #for each minimum bounding box index
                if new_bbox[i] < bbox[i]:
                    bbox[i] = new_bbox[i] #keep the new minimum
            for i in range(3,6): #for each maximum bounding box index
                if new_bbox[i] > bbox[i]:
                    bbox[i] = new_bbox[i] #keep the new maximum
        
        labelimg_list.append(labelimg)
  
    brainimg = sitk.ReadImage(opt.brain_image) #get the subject brain intensity image
    check_metadata(brainimg, metadata, opt.brain_image)
        
    #go through the AWoL-MRF steps
    awolmrf = AWoL_MRF(labelimg_list, brainimg, bbox=np.asarray(bbox), alpha=opt.alpha, beta=opt.beta, 
                       patch_length=opt.patch_length, threshold=opt.threshold, potential_maps=opt.potential_maps,
                       majority_labels=opt.majority_labels, manual_labels=opt.manual_labels, dataset=opt.dataset,
                       n_atlases=opt.n_atlases, n_templates=opt.n_templates, output_data=opt.output_data)
                       
    del labelimg_list
      
    output_image = awolmrf.run()
    
    if opt.potential_maps:
        for name, image in awolmrf.potentials.items(): #write the potential map files
            filename, fileext = os.path.splitext(opt.output_labels)
            sitk.WriteImage(image, filename + "." + name + fileext, True)
    
    sitk.WriteImage(output_image, opt.output_labels, True) #save the result to the output file