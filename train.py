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
from reid.utils.logging import Logger
from reid.utils.image_data_loader import Loaders
from reid.learner import Learner


cudnn.benchmark=True
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Re-ID',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr',type=float,default=0.00035,help='learning rate')
    parser.add_argument('--lr_step_size',type=int, nargs='+', default=[40, 70])
    parser.add_argument('--class_per_batch',type=int,default=16)  # triplet sampling, number of IDs per batch
    parser.add_argument('--track_per_class',type=int,default=4)  # triplet sampling, number of images per ID per batch
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--n_epochs',type=int,default=50)
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--latent_dim',type=int,default=2048)
    parser.add_argument('--load_ckpt',type=str,default=None)
    parser.add_argument('--log_path',type=str,default='train.txt')
    parser.add_argument('--ckpt',type=str,default='logs')
    parser.add_argument('--model_type',type=str,default='ResNet50_STB')
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='market1501')
    parser.add_argument('--data_dir', type=str, metavar='PATH', default='/path/to/dataset/')
    parser.add_argument('--market_path', type=str, default='/path/to/dataset/Market1501/')
    parser.add_argument('--duke_path', type=str, default='/path/to/dataset/DukeMTMC-reID/')
    parser.add_argument('--msmt_path', type=str, default='/path/to/dataset/MSMT17/')
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--num_steps', type=int, default=2)  ## 
    parser.add_argument('--num_class', type=int, default=0)
    parser.add_argument('--evaluate_interval', type=int, default=50)
    parser.add_argument('--save_model_name',type=str, default='ICS_model')
    parser.add_argument('--save_model_interval', type=int, default=50)
    parser.add_argument('--associate_rate', type=float, default=1.0)  # range=[0.5,1.5], 1.5 for Market1501, 1.0 for DukeMTMC and MSMT17
    parser.add_argument('--use_propagate_data',type=bool,default=True)
    args = parser.parse_args()
    args.lr_step_size = [40, 70]

    return args




if __name__ == '__main__':

    #Parse args
    args = parse_args()

    # log
    sys.stdout = Logger(os.path.join(args.ckpt, args.log_path))
    
    for run_time in range(1, 2):
        print('Run time: {} --------------------------------------------------------'.format(run_time))

        # initialize learner
        learner = Learner(args)

        selected_idx = None
        new_labels = ''
        for step in range(1, args.num_steps+1):

            print('step {} training...'.format(step))

            # re-create dataloader
            if step == 1:
                learning_setting = 'semi_supervised'
            else:
                learning_setting = 'semi_association'
            loaders = Loaders(args, selected_idx, new_labels, learning_setting=learning_setting)

            # update dataloader for learner
            learner.update_dataloader(loaders)

            # model training
            if step == 1:
                learner.train_memory_bank(step, miu=0.5, t=15, run_time=run_time)
                load_model_name = os.path.join(args.ckpt, args.save_model_name + '_step1_epoch_' + str(args.n_epochs) + '_run_time_' + str(run_time) + '.pth')
            elif step == 2:
                learner.train_semi(step, new_labels, run_time=run_time, save_model_name=args.save_model_name, load_model_name=load_model_name)

            # association
            if step == 1:
                new_labels = learner.img_association(args.num_steps)
                selected_idx = np.where(new_labels>=0)[0]



