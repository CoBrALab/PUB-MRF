#!/usr/bin/env python

import numpy as np
import SimpleITK as sitk

from argparse import ArgumentParser, ArgumentTypeError
from warnings import warn
import os.path
import sys

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
    - Does not require to compute minimum spanning trees for each patch
    - Updates the low-confidence voxels with minimum information loss
    - Defines the candidate labels for each voxel based on the local votes"""
    
    def __init__(self, labelimg_list, brainimg, bbox=None, beta=-.2, patch_length=5,
                 same_threshold=True, thresholds=[0.7, 0.7]): #use same defaults as the parser
                
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
        self.beta = float(beta)
        self.patch_length = positive_int(patch_length)
        for threshold in thresholds:
            threshold = restricted_float(threshold)

        if bbox is not None:        
            bbox[:3] -= self.patch_length #pad the bounding box with the patch length
            bbox[3:] += self.patch_length
                       
              
        nimg = len(labelimg_list)
        for n, img in enumerate(labelimg_list):
            if bbox is None:
                label_array = sitk.GetArrayFromImage(img) #get the label array from each image
            else: #get each array within the bounding box
                label_array = sitk.GetArrayFromImage(img)[bbox[2]:bbox[5], bbox[1]:bbox[4], bbox[0]:bbox[3]]
                
            if n == 0:
                self.label_values = np.unique(label_array) #obtain the list of labels
                self.label_shape = label_array.shape
                self.votes = np.zeros((self.label_values.shape[0], label_array.ravel().shape[0]), dtype=np.uint8)
            
            elif np.asarray(np.unique(label_array) != self.label_values).any(): #check label equivalence
                warn("Labels in image {} not the same as in image 1.".format(n))
            
            for i, value in enumerate(self.label_values):
                self.votes[i][np.where(label_array.ravel() == value)] += 1 #count the votes for each label
        
        if len(self.label_values) != len(thresholds):
            if not(same_threshold):
                raise AssertionError("Number of labels does not match number of thresholds.")
            else:
                while len(thresholds) < len(self.label_values):
                    thresholds.append(thresholds[-1]) #same threshold for each structural label
        
        self.mode = (np.argmax(self.votes, axis=0), np.amax(self.votes, axis=0)) #find the majority votes        
        self.labels = np.zeros(self.votes[0].shape, dtype=np.int16) - 1 #array of labels
        
        if bbox is None:
            self.intensity = sitk.GetArrayFromImage(brainimg).ravel() #array of intensities
        else:
            self.intensity = sitk.GetArrayFromImage(brainimg)[bbox[2]:bbox[5],bbox[1]:bbox[4],bbox[0]:bbox[3]].ravel()
        
        #find the high-confidence voxels for each label        
        for i, value in enumerate(self.label_values.tolist()):
            self.labels[np.where((self.mode[0] == i) & (self.mode[1] >= thresholds[i]*nimg))] = value
            
        self.bbox = bbox #keep the bounding box for the final fusion labels
        self.brainimg = brainimg #keep this to copy the metadata to the output image
            
    def run(self):
        """Find the seeds. For each seed, get the minimum spanning tree
        sequence, and compute the MRF potentials in that order. Then compute
        the final fusion labels."""
             
        self.find_lcv()
        
        if self.no_lcv:
            warn("No low-confidence voxel was found.")
            return self.get_output_image() #return majority vote output
        
        else:
            while any(self.confidence_level != -np.inf):
                self.next_voxel()
                self.get_patch_stats()
                self.mrf_potentials()
    
            return self.get_output_image()
        
    def find_lcv(self):
        """Find the low-confidence voxels. Also initialize the 6-voxel
        neighborhood and the 26-voxel neighborhood for each low-confidence
        voxel. Also computes the confidence level of each voxel."""
        
        self.lcv = np.where(self.labels == -1)[0] #find the low-confidence voxels
        
        if self.lcv.shape[0] == 0: #in this case we just want to return the majority vote
            self.no_lcv = True
            
        else:
            self.no_lcv = False
                
            self.neighbors_small = {}
            self.neighbors_big = {}
            self.patches = {}
            
            for lcv in self.lcv: #find the neighborhood of each lcv
                x, y, z = np.unravel_index(lcv, self.label_shape) #get coordinates of voxels
                
                small = []
                for coord in [(x-1,y,z),(x+1,y,z),(x,y-1,z),(x,y+1,z),(x,y,z-1),(x,y,z+1)]:
                    small.append(np.ravel_multi_index(coord, self.label_shape))
                self.neighbors_small[lcv] = small #6-voxel neighborhood
                
                big = []                
                for i in range(x-1, x+2):
                    for j in range(y-1, y+2):
                        for k in range(z-1, z+2):
                            if i != x or j != y or k != z:
                                big.append(np.ravel_multi_index((i,j,k), self.label_shape))
                self.neighbors_big[lcv] = big #26-voxel neighborhood
                
                patch = []                
                for i in range(x-self.patch_length, x+self.patch_length+1):
                    for j in range(y-self.patch_length, y+self.patch_length+1):
                        for k in range(z-self.patch_length, z+self.patch_length+1):
                            patch.append(np.ravel_multi_index((i,j,k), self.label_shape))
                self.patches[lcv] = patch #patch neighborhood
                  
            self.candidate_labels = {}
            self.confidence_level = []
            
            for lcv in self.lcv:
                #get the candidate labels for each low-confidence voxel
                local_votes = self.votes[:, np.asarray(self.neighbors_big[lcv])]
                self.candidate_labels[lcv] = self.label_values[np.unique(local_votes.nonzero()[0])]
                
                #find the confidence level of each low-confidence voxel 
                n_hcv = 0
                for value in self.label_values[1:]: #for each structural label
                    n_hcv += sum(self.labels[self.neighbors_big[lcv]] == value)
                self.confidence_level.append(n_hcv)
            
            self.confidence_level = np.asarray(self.confidence_level, dtype=np.float16)
            
            del self.label_values, self.votes
            
    def next_voxel(self):
        """Find the next voxel to be updated. The low-confidence voxels are
        updated from the highest confidence level to the lowest."""
        
        next_seed_index = np.argmax(np.asarray(self.confidence_level))
        self.voxel = self.lcv[next_seed_index]
        self.confidence_level[next_seed_index] = -np.inf
                
    def get_patch_stats(self):
        """Compute the patch stats for the current low-confidence voxels
        using mean and standard deviation. These stats are used to compute the
        MRF singleton potentials."""
        
        #keep the patch stats for each label that is in the patch
        self.patch_stats = {}
        patch = np.asarray(self.patches[self.voxel])
            
        for value in self.candidate_labels[self.voxel]:
            points = self.intensity[patch[self.labels[patch] == value]]
            if len(points) > 1: #need at least 2 points for the standard deviation
                mean, std = np.mean(points), np.std(points)
                if std != 0.0: #a standard deviation of 0 doesn't make sense for the singleton computation
                    self.patch_stats[value] = [mean, std]
                    
        del self.patches[self.voxel]
        
    def mrf_potentials(self):
        """Compute the Markov Random Field potential for the patch in the
        order given by the minimum spanning tree sequence. The weight of the
        doubleton potentials is determined by the beta patameter."""
        
        #compute doubleton and singleton potential
        lcv = self.voxel
        mrf_energy = np.inf
        n = self.neighbors_small[lcv]
        for value in self.patch_stats.keys():
            (mean, std) = self.patch_stats[value]
            mrf_single = (np.log(np.sqrt(2*np.pi)*std)) + (np.power(self.intensity[lcv]-mean,2))/(2*np.power(std,2))
            mrf_double = self.beta*(2*sum(self.labels[n] == value) + sum(self.labels[n] == -1) - len(n))
            #the formula for doubleton potentials is Li - L(not i) = 2Li + L(-1) - (nb of neighbors) 
            if (mrf_single + mrf_double) < mrf_energy:
                mrf_energy = mrf_single + mrf_double #minimize the MRF energy
                new_label = value #find the label with minimum energy
             
        self.labels[lcv] = new_label #update the label
        
        if new_label != 0: #increment the confidence level of all the neighbor low-confidence voxels
            neighbors = np.asarray(self.neighbors_big[self.voxel])
            self.confidence_level[np.searchsorted(self.lcv, neighbors[self.labels[neighbors] == -1])] += 1
        
        del self.candidate_labels[lcv], self.neighbors_big[lcv], self.neighbors_small[lcv], self.patch_stats
        
    def get_output_image(self):
        """Return the final AWoL-MRF output image with the final fusion label
        for each voxel."""
        
        self.labels = self.labels.reshape(self.label_shape)        
        
        #pad the bounding box with background labels
        if self.bbox is not None:        
            self.labels = np.pad(self.labels, ((self.bbox[2], self.brainimg.GetDepth() - self.bbox[5]),
                                  (self.bbox[1], self.brainimg.GetHeight() - self.bbox[4]),
                                  (self.bbox[0], self.brainimg.GetWidth() - self.bbox[3])),
                                  "constant", constant_values=0)
        
        #get the output SimpleITK image with fusion labels        
        output_image = sitk.GetImageFromArray(self.labels)
        output_image.CopyInformation(self.brainimg) #copy the metadata
        
        return output_image

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
                            - Does not require to compute minimum spanning trees for each patch
                            - Updates the low-confidence voxels with minimum information loss
                            - Defines the candidate labels for each voxel based on the local votes""")  
                            
    pg = parser.add_argument_group("AWoL-MRF parameters")
    pg.add_argument("-b", "--beta", type=float, default=-.2,
                    help="[default = %(default)s]")
    pg.add_argument("-p", "--patch_length", type=positive_int, default=5,
                    help="[default = %(default)s]")
    #the thresholds should be in the same order as the labels
    pg.add_argument("-t", "--thresholds", nargs="+", type=restricted_float, 
                    default=[0.7, 0.7], metavar=("T_BACKGROUND", "T_STRUCTURE"),
                    help="""A voxel V has high-confidence for any label L if the number of votes
                    it receives for that label is greater than (1/n(V) + T)*nimg, where n(V) is
                    the number of candidate labels at voxel V. [default = %(default)s]""")
    
    #user can decide if he wants different labels for each structural label
    tg = pg.add_mutually_exclusive_group()
    tg.add_argument("--same-threshold", dest="same_threshold", action="store_true",
                   help="same threshold for each structure [default = %(default)s]")
    tg.add_argument("--different-thresholds", dest="same_threshold", action="store_false",
                   help="opposite of '--same-threshold'")
    tg.set_defaults(same_threshold=True)
                    
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

    opt = parser.parse_args()
    

    if not(opt.clobber) and os.path.exists(opt.output_labels):
        sys.exit("Output file already exists; use --clobber to overwrite.")
    
    #load volumes from input files    
    labelimg_list = [] #list of candidate segmentation images
    
    #use this to verify if the voxel-wise computations make sense    
    def check_metadata(img, metadata, filename):
        if img.GetSize() != metadata["size"]:
            sys.exit("Size of {0} not the same as {1}".format(filename, opt.input_labels[0]))
        elif img.GetOrigin() != metadata["origin"]:
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
            metadata["origin"] = labelimg.GetOrigin()
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
    awolmrf = AWoL_MRF(labelimg_list, brainimg, bbox=np.asarray(bbox), beta=opt.beta, patch_length=opt.patch_length,
                       same_threshold=opt.same_threshold, thresholds=opt.thresholds)
                       
    del labelimg_list
      
    output_image = awolmrf.run()
    
    sitk.WriteImage(output_image, opt.output_labels, True) #save the result to the output file