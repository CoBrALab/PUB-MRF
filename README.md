# PUB-MRF: a novel probabilistic method for label fusion in brain segmentation pipelines

The PUB-MRF algorithm uses a Markov Random Field model to update the
label probabilities obtained with a multi-atlas registration method. In
particular, the algorithm has been tested extensively with MAGeT-Brain.

PUB-MRF starts by subdividing the stereotaxic space into a high-confidence
and a low-confidence region. A voxel is included in the low-confidence
region if and only if for any label, the percentage of votes at that voxel
does not exceed a given threshold. 

Then, PUB-MRF iterates through the voxels in the low-confidence region. 
At each voxel, a local Markov Random Field is defined. We consider that
the set of all labels which receive at least one vote on a given voxel is 
a partition of the sample space.

The doubleton potentials are estimated using the segmentation votes at the
voxel itself, and in an immediate 26-voxel neighborhood. The singleton
potential for each label is estimated using the local intensity values in
the brain scan, under the assumption that with a large number of voxels, 
this will approximately correspond to a normal distribution.

For any voxel in the low-confidence region, the final label is the argmin 
of the Markov Random Field energies, which corresponds to the argmax of
the updated label probabilities. In the high-confidence region, the output 
labels are obtained using majority vote.

-------------------------------------------------------------------------    
                         PUB-MRF Parameters
-------------------------------------------------------------------------

self.threshold     : Let N be the number of labels which receive at least
                     one at voxel v. Then v is in the low-confidence
                     region if and only if, for any label l,
                     P(L(v) = l) < (1.0/N + self.threshold)

self.patch_length  : At each low-confidence voxel v, the region used to
                     compute the singleton potential is a cube with edge
                     length (2*self.patch_length + 1) centered at v.

self.alpha         : Corresponds to the relative weight of the doubleton
                     potential with respect to the singleton potential in
                     the MRF energy computation.

self.beta          : The weights in the 26-voxel neighborhood for the
                     doubleton potential are evaluated with an expotential
                     decay function with parameter self.beta, with respect
                     to the Euclidian norm.

Key features of this version:
- Works with any number of separate or adjacent labels
- Assumes strictly positive integer values for the structural labels
- Assumes that the background label is 0
- Uses smart bounding boxes to reduce peak memory usage

(C) Charles Lagace, Nikhil Bhagwat, Chakravarty Lab
http://www.douglas.qc.ca/researcher/mallar-chakravarty?locale=en
