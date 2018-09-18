from numpy import *
from scipy import stats
from scipy.sparse import csgraph
from scipy.spatial import KDTree

import SimpleITK as sitk

from argparse import ArgumentParser, ArgumentTypeError
import os.path
import sys

class AWoL_MRF:
    """The AWoL-MRF algorithm organizes the low-confidence voxels in patches. 
    In each patch, the labels for these voxels are updated in a sequence given
    by Prim's algorithm using the Markov Random Field potentials.
    
    Key features of this version:
    - Works with a single structural label
    - Works with 2 or more separate structural labels
    - Untested with 2 or more adjacent structural labels
    - Assumes the background label is 0 (or smaller than any structural label)
    - Assumes strictly positive integer values for the structural labels
    - Poor exception handling; the program may crash if given bad input
    - AWoL-MRF class not directly usable on a Python notebook"""
    
    def __init__(self, labels, label_values):
        self.labels = labels
        self.label_values = label_values
        
        #find the low-confidence voxels
        lcv = argwhere(self.labels == -1)
        tree_lcv = KDTree(lcv)
        self.lcv = ravel_multi_index(where(labels == -1), self.labels.shape)
        
        #create a tree with all the coordinates        
        self.tree = KDTree(argwhere(isfinite(self.labels)))
        
        def neighbors(tree_lcv, tree, dist_max):
            neighbors = {}            
            qbt = tree_lcv.query_ball_tree(tree, dist_max)
            
            #remove each point from its own list of neighbors
            for i, elt in enumerate(qbt):
                qbt[i].remove(self.lcv[i])
                neighbors[self.lcv[i]] = qbt[i]
                
            return neighbors
                
        self.neighbors_small = neighbors(tree_lcv, self.tree, 1) #6-voxel neighborhood        
        self.neighbors_big = neighbors(tree_lcv, self.tree, sqrt(3)) #26-voxel neighborhood
           
        self.label_count = zeros((self.label_values.shape[0], self.labels.ravel().shape[0]))
    
    def find_seeds(self, mixing_ratio, patch_length):
        """Find the seeds for the AWoL-MRF patches. We assume that each seed 
        needs a minimum number of high-confidence voxels in its 26-voxel
        neighbourhood, which is determined by the mixing ratio parameter."""
        
        self.seeds = []
        seed_coord = []
        lflat = self.labels.ravel()
        for lcv in self.lcv:
            for value in self.label_values[1:]: #for each structural label
                if sum(lflat[self.neighbors_big[lcv]] == value) > mixing_ratio:
                    self.seeds.append(lcv)
                    seed_coord.append(unravel_index(lcv, self.labels.shape))
                    
        del self.neighbors_big
                
        if len(self.seeds) > 500: #control for max number of seeds
            self.seeds = self.seeds[:500]
            seed_coord = seed_coord[:500]
        
        #find the patch for each seed using KDTree
        tree_seeds = KDTree(seed_coord)
        self.patches = tree_seeds.query_ball_tree(self.tree, patch_length)
        for i, elt in enumerate(self.patches):
            self.patches[i].remove(self.seeds[i])
            
        del self.tree
                
    def get_mst_sequence(self, intensity):
        """Finds the minimum spanning tree sequence for the low-confidence
        voxels in the patch around the seed. The dimensions of the patch are
        defined by the patch length parameter."""
        
        #find the low-confidence voxels in the patch     
        lflat = self.labels.ravel()
        lcvp = asarray(self.patches[0])[lflat[self.patches[0]] == -1]
        
        #keep the patch stats for each label that is in the patch
        self.patch_stats = {}
        iflat = intensity.ravel()
        for value in self.label_values:
            points = iflat[asarray(self.patches[0])[lflat[self.patches[0]] == value]]
            if len(points) > 0: #need at least 1 point to consider a label in the patch
                self.patch_stats[value] = [mean(points), std(points)]
            
        #fill the weight matrix for the MST
        n = shape(lcvp)[0] #the number of lcv in the patch
        weightm = zeros((n, n))
        for wx in range(0, n):
            for wy in range(wx, n):
                if lcvp[wx] in self.neighbors_small[lcvp[wy]]:
                    weightm[wx][wy] = abs(iflat[lcvp[wx]] - iflat[lcvp[wy]]) #intensity gradient
                else:
                    xcoord = asarray(unravel_index(lcvp[wx], self.labels.shape))
                    ycoord = asarray(unravel_index(lcvp[wy], self.labels.shape))
                    weightm[wx][wy] = 100*linalg.norm(xcoord - ycoord) #proportional to the norm
                    
        mst = csgraph.minimum_spanning_tree(weightm).toarray() #get the MST

        #find the MST sequence        
        self.seq = [0]
        edges = argwhere(mst).tolist() #edges of the MST
        
        while edges:
            mincost = amax(mst) + 1
            for e in edges:
                if (e[0] in self.seq or e[1] in self.seq) and mst[e[0]][e[1]] < mincost:
                    mincost = mst[e[0]][e[1]]
                    nextedge = e #find the edge with minimum cost
                      
            if nextedge[0] in self.seq:
                self.seq.append(nextedge[1])
            else: #if nextedge[1] in self.seq
                self.seq.append(nextedge[0])
            
            edges.remove(nextedge)

        self.seq = [lcvp[:][i] for i in self.seq] #get the ordered list of lcv
        
        del self.patches[0]
        del self.seeds[0]
        
    def mrf_potentials(self, beta, intensity):
        """Compute the Markov Random Field potential for the patch in the
        order given by the minimum spanning tree sequence. The weight of the
        doubleton potentials is determined by the beta patameter."""        
        
        iflat = intensity.ravel()
        lflat = self.labels.ravel()
        
        #compute doubleton and singleton potential
        for lcv in self.seq:
            mrf_energy = inf
            n = self.neighbors_small[lcv] 
            for value in self.label_values:
                if value in self.patch_stats.keys():
                    (mean, std) = self.patch_stats[value]
                    mrf_single = (log(sqrt(2*pi))*std) + (power(iflat[lcv]-mean,2))/(2*power(std,2))
                    mrf_double = beta*(2*sum(lflat[n] == value) + sum(lflat[n] == -1) - len(n))
                    #the formula for doubleton potentials is Li - L(not i) = 2Li + L(-1) - (nb of neighbors) 
                    if (mrf_single + mrf_double) < mrf_energy:
                        mrf_energy = mrf_single + mrf_double #minimize the MRF energy
                        new_label = value #find the label with minimum energy
            
            #update the label count
            index = where(self.label_values == new_label)[0][0]
            self.label_count[index][lcv] += 1
        
        del self.seq        
        del self.patch_stats
        
    def update_labels(self, mode):
        """Update the labels of the low-confidence voxels after the walk in
        each patch. If a voxel is not found in any patch, its final label is
        the majority vote label. Otherwise, its label is the most popular
        label throughout the patches."""
        
        #get the final fusion labels        
        new_labels = (amax(self.label_count, axis=0), argmax(self.label_count, axis=0))
        
        for lcv in self.lcv:
            if new_labels[0][lcv] > 0: #if the low-confidence voxel is in at least 1 patch
                self.labels.ravel()[lcv] = self.label_values[new_labels[1][lcv]]
            else:
                self.labels.ravel()[lcv] = mode[0][0].ravel()[lcv]
                

if __name__ == "__main__":
    #AWoL-MRF parameters
    def positive_int(x): #avoid nonsense negative parameter values
        x = int(x)
        if x < 0:
            raise ArgumentTypeError("%r is not a positive int"%(x,))
        return x
            
    def restricted_float(x): #avoid nonsense values for the thresholds
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
                            - Untested with 2 or more adjacent structural labels
                            - Assumes the background label is 0 (or smaller than any structural label)
                            - Assumes strictly positive integer values for the structural labels
                            - Poor exception handling; the program will crash if given bad input
                            - AWoL-MRF class not directly usable on a Python notebook""")    
    pg = parser.add_argument_group("AWoL-MRF parameters")
    pg.add_argument("-b", "--beta", type=float, default=-.2,
                    help="[default = %(default)s]")
    pg.add_argument("-p", "--patch_length", type=positive_int, default=5,
                    help="[default = %(default)s]")
    pg.add_argument("-r", "--mixing_ratio", type=positive_int, default=10,
                    help="[default = %(default)s]")
    #the thresholds should be in the same order as the labels
    pg.add_argument("-t", "--thresholds", nargs="+", type=restricted_float, 
                    default=[.8, .6], metavar=("T_BACKGROUND", "T_STRUCTURE"),
                    help="[default = %(default)s]")
    
    #user can decide if he wants different labels for each structural label
    tg = parser.add_mutually_exclusive_group()
    tg.add_argument("--same-threshold", dest="same_threshold", action="store_true",
                   help="same threshold for structural labels [default = %(default)s]")
    tg.add_argument("--different-thresholds", dest="same_threshold", action="store_false",
                   help="opposite of '--same-threshold'")
    tg.set_defaults(same_threshold=True)
                    
    #file manipulation arguments
    parser.add_argument("input_files", nargs="+", type=str)
    parser.add_argument("brain_file", type=str) #need this for the singleton potentials
    parser.add_argument("output_file", type=str)
    cg = parser.add_mutually_exclusive_group()
    cg.add_argument("--clobber", dest="clobber", action="store_true",
                   help="clobber output file [default = %(default)s]")
    cg.add_argument("--no-clobber", dest="clobber", action="store_false",
                   help="opposite of '--clobber'")
    cg.set_defaults(clobber=False)

    opt = parser.parse_args()

    if not(opt.clobber) and os.path.exists(opt.output_file):
        sys.exit("Output file already exists; use --clobber to overwrite.")
    
    #load volumes from input files    
    volhandles = []

    nfiles = len(opt.input_files)
    for filename in opt.input_files:
        label_image = sitk.ReadImage(filename) #get all the candidate segmentations
        label_array = sitk.GetArrayFromImage(label_image)
        
        if len(volhandles) == 0:        
            label_values = unique(label_array) #obtain the list of labels
        elif asarray(unique(label_array) != label_values).any(): #make sure that they are the same in each image
            sys.exit("Labels in {0} not the same as in {1}.".format(filename, opt.input_files[0]))
            
        volhandles.append(label_array)
        
    if len(label_values) != len(opt.thresholds):
        if not(opt.same_threshold):
            sys.exit("Number of labels does not match number of thresholds.")
        else: #if opt.same_threshold
            while len(opt.thresholds) < len(label_values):
                opt.thresholds.append(opt.thresholds[-1]) #same threshold for each structural label
        
    mode = stats.mode(volhandles)
    labels = zeros(volhandles[0].shape) - 1 #array of labels, -1 is for low-confidence voxels
    intensity = sitk.GetArrayFromImage(sitk.ReadImage(opt.brain_file)) #array of intensities
    
    for i, l in enumerate(label_values.tolist()):
        labels[(mode[0][0] == l) & (mode[1][0] > opt.thresholds[i]*nfiles)] = l
        
    #go through the AWoL-MRF steps
    awolmrf = AWoL_MRF(labels, label_values) #initialize the AWoL-MRF instance
    awolmrf.find_seeds(opt.mixing_ratio, opt.patch_length)
    
    while awolmrf.seeds: #for each seed
        awolmrf.get_mst_sequence(intensity)
        awolmrf.mrf_potentials(opt.beta, intensity)
    
    awolmrf.update_labels(mode) #get the fusion labels
    
    #save the label fusion result to the output file
    output = sitk.GetImageFromArray(awolmrf.labels)
    output.CopyInformation(label_image) #copy the metadata
    sitk.WriteImage(output, opt.output_file)