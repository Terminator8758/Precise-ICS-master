import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import argparse
import sys
import random
import numpy as np
import math
import time
from sklearn.cluster.dbscan_ import dbscan
from scipy.spatial.distance import pdist, cdist, squareform

cudnn.benchmark=True
torch.multiprocessing.set_sharing_strategy('file_system')


def propagate_label(W, IDs, all_cams, associate_class_pair, step, max_step):
    # start label propagation
    print('Start associating ID...')

    # mask out intra-camera classes and lower-half
    for i in range(len(W)):
        W[i,np.where(all_cams==all_cams[i])[0]]=1000
        lower_ind=np.arange(0,i)
        W[i, lower_ind]=1000

    # cross-camera association
    associateMat=1000*np.ones(W.shape, W.dtype)

    # mask out intra-camera classes and lower-half
    for i in range(len(W)):
        W[i, np.where(all_cams == all_cams[i])[0]] = 1000
        lower_ind = np.arange(0, i)
        W[i, lower_ind] = 1000

    sorted_ind = np.argsort(W.flatten())[0:int(associate_class_pair)]
    row_ind = sorted_ind // W.shape[1]
    col_ind = sorted_ind % W.shape[1]

    C = len(np.unique(all_cams))
    cam_cover_info = np.zeros((len(W), C))
    associ_num, ignored_num = 0, 0
    associ_pos_num, ignored_pos_num = 0, 0
    print('  associate_class_pair: {}, step: {}, max_step: {}'.format(associate_class_pair, step, max_step))
    thresh = associate_class_pair * step / max_step
    print('  thresh= {}'.format(thresh))

    for m in range(len(row_ind)):
        cls1 = row_ind[m]
        cls2 = col_ind[m]
        assert (all_cams[cls1] != all_cams[cls2])
        check = (cam_cover_info[cls1, all_cams[cls2]] == 0 and cam_cover_info[cls2, all_cams[cls1]] == 0)
        if check:
            cam_cover_info[cls1, all_cams[cls2]] = 1
            cam_cover_info[cls2, all_cams[cls1]] = 1
            associateMat[cls1, cls2] = 0
            associateMat[cls2, cls1] = 0
            associ_num += 1
            if IDs[cls1] == IDs[cls2]:
                associ_pos_num += 1
        else:
            ignored_num += 1
            if IDs[cls1] == IDs[cls2]:
                ignored_pos_num += 1
        if associ_num >= thresh:
            break
    print('  associated class pairs: {}/{} correct, ignored class pairs: {}/{} correct'.
            format(associ_pos_num, associ_num, ignored_pos_num, ignored_num))
        
    # mask our diagnal elements
    for m in range(len(associateMat)):
        associateMat[m,m]=0

    _, new_merged_label = dbscan(associateMat, eps=3, min_samples=2, metric='precomputed')
    print('  length of merged_label= {}, min= {}, max= {}'.format(len(new_merged_label),np.min(new_merged_label),np.max(new_merged_label)))

    return new_merged_label



