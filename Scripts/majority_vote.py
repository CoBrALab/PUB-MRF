#!/usr/bin/env python

import SimpleITK as sitk
import numpy as np

from argparse import ArgumentParser
from warnings import warn
import os.path
import sys

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_labels", nargs="+", type=str)
    parser.add_argument("output_labels", type=str)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--clobber", dest="clobber", action="store_true",
                   help="clobber output file [default = %(default)s]")
    g.add_argument("--no-clobber", dest="clobber", action="store_false",
                   help="opposite of '--clobber'")
    g.set_defaults(clobber=False)

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
    
    nimg = len(labelimg_list)
    for n, img in enumerate(labelimg_list):
        label_array = sitk.GetArrayFromImage(img)[bbox[2]:bbox[5], bbox[1]:bbox[4], bbox[0]:bbox[3]]
        
        if n == 0:
            label_values = np.unique(label_array) #obtain the list of labels
            votes = np.zeros((label_values.shape[0], label_array.shape[0], 
                              label_array.shape[1], label_array.shape[2]))
        elif np.asarray(np.unique(label_array) != label_values).any(): #make sure that they are the same in each image
            warn("Labels in image {0} not the same as in image {1}.".format(opt.input_labels[n], opt.input_labels[0]))
        
        for i, value in enumerate(label_values):
            votes[i][np.where(label_array == value)] += 1 #count the votes for each label
    
    mode = np.argmax(votes, axis=0) #find the majority votes        
    labels = np.zeros(votes[0].shape, dtype=np.uint8) #array of labels
    
    for i, value in enumerate(label_values.tolist()):
        labels[np.where(mode == i)] = value #assign the majority vote to all voxels
    
    labels = np.pad(labels, ((bbox[2], labelimg.GetDepth() - bbox[5]),(bbox[1], labelimg.GetHeight() - bbox[4]),
                             (bbox[0], labelimg.GetWidth() - bbox[3])), "constant", constant_values=0)
    
    output_image = sitk.GetImageFromArray(labels)
    output_image.CopyInformation(labelimg) #copy the metadata
    
    sitk.WriteImage(output_image, opt.output_labels, True) #save the result to the output file
