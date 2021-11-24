import os
import sys
import time
import numpy as np
import pandas as pd
import collections
import random
import math
## For torch lib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torch.nn.functional as F
## For Image lib
from PIL import Image
import scipy.io as sio


'''
For MARS,Video-based Re-ID
'''



def process_labels_semi_supervised(labels, cams, pred_labels='', selected_train=False):
    semi_labels = np.zeros(labels.shape, labels.dtype)
    accumulate_labels = np.zeros(labels.shape, labels.dtype)
    id_count_each_cam = []
    unique_cams = np.unique(cams)
    prev_id_count = 0
    print('  unique cameras= {}'.format(unique_cams))

    for this_cam in unique_cams:
        percam_labels = labels[cams==this_cam]
        unique_id = np.unique(percam_labels)
        id_count_each_cam.append(len(unique_id))
        id_dict = {ID: i for i, ID in enumerate(unique_id.tolist())}
        for i in range(len(percam_labels)):
            percam_labels[i] = id_dict[percam_labels[i]]

        semi_labels[cams==this_cam]=percam_labels
        accumulate_labels[cams==this_cam]=percam_labels+prev_id_count
        prev_id_count += len(unique_id)
    assert prev_id_count == (np.max(accumulate_labels)+1)

    if len(pred_labels)>0:
        assert(len(pred_labels) == (np.max(accumulate_labels)+1))  # number of accumulated IDs
        #print('  unique values in pred_labels= {}'.format(np.unique(pred_labels)))
        prev_id_count = len(np.unique(pred_labels[np.where(pred_labels>=0)]))

        instance_pred_labels = np.zeros(accumulate_labels.shape, accumulate_labels.dtype)
        for j in range(len(instance_pred_labels)):
            instance_pred_labels[j] = pred_labels[accumulate_labels[j]]

        if selected_train:
            selected_idx = np.where(instance_pred_labels>=0)[0]
            print('  selected index length= {}, values= {}'.format(len(selected_idx), selected_idx))
            instance_pred_labels = instance_pred_labels[selected_idx]
            semi_labels = semi_labels[selected_idx]
            accumulate_labels = accumulate_labels[selected_idx]
            return semi_labels, accumulate_labels, id_count_each_cam, prev_id_count, instance_pred_labels, selected_idx

        return semi_labels, accumulate_labels, id_count_each_cam, prev_id_count, instance_pred_labels

    return semi_labels, accumulate_labels, id_count_each_cam, prev_id_count



class Video_semi_train_Dataset(Dataset):
    def __init__(self, db_txt, info, transform, S=6, track_per_class=4, flip_p=0.5, delete_one_cam=False, target_camera=-1, cam_type='normal', pred_labels='', selected_train=False, max_frams_num=900):
        """
        :param db_txt: list of train image names(paths)
        :param info: train/test tracklet info, form as (start_ind, end_ind, ID_label, cam_label)
        """
        with open(db_txt,'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # For info (id,track)
        if delete_one_cam == True:  # delete IDs that only appear in one camera
            info = np.load(info)
            info[:,2],id_count = process_labels(info[:,2])
            for i in range(id_count):
                idx = np.where(info[:,2]==i)[0]
                if len(np.unique(info[idx,3])) ==1:
                    info = np.delete(info,idx,axis=0)
                    id_count -=1
            info[:,2],id_count = process_labels(info[:,2])
            #change from 625 to 619
        else:
            info = np.load(info)
            #info[:,2],id_count = process_labels(info[:,2])    # re-order Id_label to 0,1,2,3...,624
            if target_camera != -1:  # to get dataloader in a specific camera
                idx = np.where(info[:, 3] == target_camera)[0]
                info = info[idx]
                print('  Target camera: {}, tracklet number under the camera: {}'.format(target_camera, len(info)))
            
            print('  length of info= {}'.format(len(info)))
            processed_infos = process_labels_semi_supervised(info[:,2], info[:,3], pred_labels, selected_train)    # re-order Id_label separately for each camera
            semi_labels, accumulate_labels, id_count_each_cam, prev_id_count = processed_infos[0], processed_infos[1], processed_infos[2], processed_infos[3]
            print('  after label processing: id_count_each_cam={},  prev_id_count={}'.format(id_count_each_cam, prev_id_count))
            print('  length of semi_labels ={}, length of accumulate_labels = {}'.format(len(semi_labels), len(accumulate_labels)))

            info[:, 2], supervised_id_count = process_labels(info[:,2])
            print('  after label processing: supervised_id_count={}'.format(supervised_id_count))

            accumulate_labels = accumulate_labels.reshape((-1,1))
            semi_labels = semi_labels.reshape((-1,1))
            
            if len(pred_labels)>0 and selected_train:
                selected_idx = processed_infos[5]
                print('  length of selected index= {}'.format(len(selected_idx)))
                info = info[selected_idx]  # get instances from selected index
                #print(' info data shape= ',info.shape)
            info = np.concatenate((info, semi_labels, accumulate_labels),axis=1)

            if len(pred_labels)>0:
                instance_pred_labels = processed_infos[4]
                print('  length of instance_pred_labels = {}'.format(len(instance_pred_labels)))
                instance_pred_labels = instance_pred_labels.reshape((-1, 1))
                info = np.concatenate((info, instance_pred_labels), axis=1)
                # form as (start_ind, end_ind, supervised_label, camera, percam_ID_label, accumulate_label, predicted_label)
            #sio.savemat('semi_mars_train_info.mat', {'train_info': info})

        self.info = []
        for i in range(len(info)):
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F < S:
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(S-F)
                for s in range(S):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/S)
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(interval*S-F)
                for s in range(S):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            if len(pred_labels)>0:
                self.info.append(np.array([np.array(sample_clip),info[i][2],info[i][3],info[i][4],info[i][5],info[i][6]]))
            else:
                self.info.append(np.array([np.array(sample_clip),info[i][2],info[i][3],info[i][4],info[i][5]]))
            # self.info: (sampled_track, supervised_label, camera, percam_ID_label, accumulate_label, predicted_label)

        self.info = np.array(self.info)
        print('  length of self.info= {}'.format(len(self.info)))
        self.transform = transform
        self.supervised_n_id = supervised_id_count
        self.n_id = prev_id_count
        self.id_each_camera = id_count_each_cam
        self.n_tracklets = self.info.shape[0]
        self.flip_p = flip_p
        self.track_per_class = track_per_class
        self.cam_type = cam_type
        self.two_cam = False
        self.cross_cam = False
        self.have_predicted_label = (len(pred_labels)>0)
        self.selected_train = selected_train
        self.max_frame_num = max_frams_num

    def __getitem__(self,ID):
        if self.have_predicted_label:
            sub_info = self.info[self.info[:, 5] == ID]  # use predicted ID label to sample
        else:
            sub_info = self.info[self.info[:, 4] == ID]   # get item from global accumulated ID (semi-supervised)

        global_labels = ID*torch.ones(self.track_per_class,dtype=torch.int64)

        if self.cam_type == 'normal':  # randomly choose certain tracks of this ID
            #tracks_pool = list(np.random.choice(sub_info[:,0],self.track_per_class))

            sampled_ind = np.random.choice(np.arange(len(sub_info)), self.track_per_class)
            tracks_pool = list(sub_info[:,0][sampled_ind])

            super_labels = list(sub_info[:, 1][sampled_ind])
            super_labels = torch.tensor(super_labels, dtype=torch.int64)

            cams = list(sub_info[:,2][sampled_ind])
            cams = torch.tensor(cams,dtype=torch.int64)

            percam_labels = list(sub_info[:, 3][sampled_ind])
            percam_labels = torch.tensor(percam_labels, dtype=torch.int64)

            global_labels = list(sub_info[:, 4][sampled_ind])
            global_labels = torch.tensor(global_labels, dtype=torch.int64)

            if self.have_predicted_label:
                predicted_labels = list(sub_info[:, 5][sampled_ind])
                predicted_labels = torch.tensor(predicted_labels, dtype=torch.int64)

        elif self.cam_type == 'evaluation':
            #max_ind = min(20, len(sub_info))
            tracks_pool = list(sub_info[:, 0])

            super_labels = list(sub_info[:, 1])
            super_labels = torch.tensor(super_labels, dtype=torch.int64)

            cams = list(sub_info[:, 2])
            cams = torch.tensor(cams, dtype=torch.int64)

            percam_labels = list(sub_info[:, 3])
            percam_labels = torch.tensor(percam_labels, dtype=torch.int64)

            global_labels = list(sub_info[:, 4])
            global_labels = torch.tensor(global_labels, dtype=torch.int64)

            if self.have_predicted_label:
                predicted_labels = list(sub_info[:, 5])
                predicted_labels = torch.tensor(predicted_labels, dtype=torch.int64)

        elif self.cam_type == 'two_cam':  # choose from two cameras of this ID
            unique_cam = np.random.permutation(np.unique(sub_info[:,2]))[:2]
            tracks_pool = list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[0],0],1))+\
                list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[1],0],1))
        elif self.cam_type == 'cross_cam':  # sample from all cameras equally
            unique_cam = np.random.permutation(np.unique(sub_info[:,2]))
            while len(unique_cam) < self.track_per_class:
                unique_cam = np.append(unique_cam,unique_cam)
            unique_cam = unique_cam[:self.track_per_class]
            tracks_pool = []
            for i in range(self.track_per_class):
                tracks_pool += list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[i],0],1))

        one_id_tracks = []
        for track_pool in tracks_pool:
            if self.cam_type == 'evaluation':
                # for evaluation: choose indexes
                if track_pool.shape[0] * len(tracks_pool) > self.max_frame_num:
                    crop_len = int(self.max_frame_num / len(tracks_pool))
                    rand_ind = np.arange(len(track_pool), dtype=np.int64)
                    np.random.shuffle(rand_ind)
                    idx = np.zeros((crop_len,), dtype=np.int64)
                    number = track_pool[rand_ind[:crop_len], idx]
                else:
                    idx = np.zeros((track_pool.shape[0]), dtype=np.int64)  # fix-interval first-item sampling
                    number = track_pool[np.arange(len(track_pool)), idx]
            else:
                idx = np.random.choice(track_pool.shape[1],track_pool.shape[0]) # fix-interval random sampling a tracklet        
                number = track_pool[np.arange(len(track_pool)),idx]

            imgs = [self.transform(Image.open(path)) for path in self.imgs[number]]
            imgs = torch.stack(imgs,dim=0)

            random_p = random.random()
            if random_p  < self.flip_p:
                imgs = torch.flip(imgs,dims=[3])
            one_id_tracks.append(imgs)

        if self.have_predicted_label:
            return torch.stack(one_id_tracks,dim=0), super_labels, cams, percam_labels, global_labels, predicted_labels
        else:
            return torch.stack(one_id_tracks,dim=0), super_labels, cams, percam_labels, global_labels

    def __len__(self):
        return self.n_id


def Video_semi_train_collate_fn(data):
    if isinstance(data[0],collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value for key,value in zip(data[0].keys(),values)}
    else:
        imgs, super_labels, cams, percam_labels, global_labels = zip(*data)
        imgs = torch.cat(imgs,dim=0)
        super_labels = torch.cat(super_labels,dim=0)
        cams = torch.cat(cams, dim=0)
        percam_labels = torch.cat(percam_labels,dim=0)
        global_labels = torch.cat(global_labels,dim=0)
        return imgs, super_labels, cams, percam_labels, global_labels


def Video_semi_train_collate_fn_v2(data):
    if isinstance(data[0],collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value for key,value in zip(data[0].keys(),values)}
    else:
        imgs, super_labels, cams, percam_labels, global_labels, predicted_labels = zip(*data)
        imgs = torch.cat(imgs,dim=0)
        super_labels = torch.cat(super_labels,dim=0)
        cams = torch.cat(cams, dim=0)
        percam_labels = torch.cat(percam_labels,dim=0)
        global_labels = torch.cat(global_labels,dim=0)
        predicted_labels = torch.cat(predicted_labels,dim=0)
        return imgs, super_labels, cams, percam_labels, global_labels, predicted_labels


def Get_Video_semi_train_DataLoader(db_txt,info,transform,shuffle=True,num_workers=8,S=10,track_per_class=4,class_per_batch=8,target_camera=-1,flip_p=0.5,cam_type='normal',pred_labels='',selected_train=False):
    dataset = Video_semi_train_Dataset(db_txt,info,transform,S,track_per_class,target_camera=target_camera,flip_p=flip_p,
                                       cam_type=cam_type, pred_labels=pred_labels, selected_train=selected_train)

    if len(pred_labels)>0:
        collate_fn = Video_semi_train_collate_fn_v2
    else:
        collate_fn = Video_semi_train_collate_fn

    dataloader = DataLoader(dataset,batch_size=class_per_batch,collate_fn=collate_fn,shuffle=shuffle,
                            worker_init_fn=lambda _:np.random.seed(),drop_last=True,num_workers=num_workers)
    return dataloader




# --------------- the fully-supervised train data loader:

def process_labels(labels):
    unique_id = np.unique(labels)
    id_count = len(unique_id)
    id_dict = {ID:i for i, ID in enumerate(unique_id.tolist())}
    for i in range(len(labels)):
        labels[i] = id_dict[labels[i]]
    assert len(unique_id)-1 == np.max(labels)
    return labels,id_count



class Video_train_Dataset(Dataset):
    def __init__(self,db_txt,info,transform,S=6,track_per_class=4,flip_p=0.5,delete_one_cam=False,cam_type='normal'):
        """
        :param db_txt: list of train image names(paths)
        :param info: train/test tracklet info, form as (start_ind, end_ind, ID_label, cam_label)
        """
        with open(db_txt,'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # For info (id,track)
        if delete_one_cam == True:
            info = np.load(info)
            info[:,2],id_count = process_labels(info[:,2])
            for i in range(id_count):
                idx = np.where(info[:,2]==i)[0]
                if len(np.unique(info[idx,3])) ==1:
                    info = np.delete(info,idx,axis=0)
                    id_count -=1
            info[:,2],id_count = process_labels(info[:,2])
            #change from 625 to 619
        else:
            info = np.load(info)
            info[:,2],id_count = process_labels(info[:,2])    # re-order Id_label to 0,1,2,3...,624

        self.info = []
        for i in range(len(info)):
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F < S:
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(S-F)
                for s in range(S):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/S)
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(interval*S-F)
                for s in range(S):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip),info[i][2],info[i][3]]))  # self.info: (sampled_track, Id, camera)

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = id_count
        self.n_tracklets = self.info.shape[0]
        self.flip_p = flip_p
        self.track_per_class = track_per_class
        self.cam_type = cam_type
        self.two_cam = False
        self.cross_cam = False

    def __getitem__(self,ID):
        sub_info = self.info[self.info[:,1] == ID]

        if self.cam_type == 'normal':  # randomly choose certain tracks of this ID
            tracks_pool = list(np.random.choice(sub_info[:,0],self.track_per_class))

        elif self.cam_type == 'two_cam':  # choose from two cameras of this ID
            unique_cam = np.random.permutation(np.unique(sub_info[:,2]))[:2]
            tracks_pool = list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[0],0],1))+\
                list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[1],0],1))
        elif self.cam_type == 'cross_cam':  # sample from all cameras equally
            unique_cam = np.random.permutation(np.unique(sub_info[:,2]))
            while len(unique_cam) < self.track_per_class:
                unique_cam = np.append(unique_cam,unique_cam)
            unique_cam = unique_cam[:self.track_per_class]
            tracks_pool = []
            for i in range(self.track_per_class):
                tracks_pool += list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[i],0],1))

        one_id_tracks = []
        for track_pool in tracks_pool:
            idx = np.random.choice(track_pool.shape[1],track_pool.shape[0])
            number = track_pool[np.arange(len(track_pool)),idx]
            imgs = [self.transform(Image.open(path)) for path in self.imgs[number]]
            imgs = torch.stack(imgs,dim=0)

            random_p = random.random()
            if random_p  < self.flip_p:
                imgs = torch.flip(imgs,dims=[3])
            one_id_tracks.append(imgs)

        return torch.stack(one_id_tracks,dim=0), ID*torch.ones(self.track_per_class,dtype=torch.int64)   # (sampled tracks of one ID, track-level labels)

    def __len__(self):
        return self.n_id


def Video_train_collate_fn(data):
    if isinstance(data[0],collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key,value in zip(data[0].keys(),values)}
    else:
        imgs,labels = zip(*data)
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)
        return imgs,labels

def Get_Video_train_DataLoader(db_txt,info,transform,shuffle=True,num_workers=8,S=10,track_per_class=4,class_per_batch=8):
    dataset = Video_train_Dataset(db_txt,info,transform,S,track_per_class)
    dataloader = DataLoader(dataset,batch_size=class_per_batch,collate_fn=Video_train_collate_fn,shuffle=shuffle,worker_init_fn=lambda _:np.random.seed(),drop_last=True,num_workers=num_workers)
    return dataloader


class Video_test_Dataset(Dataset):
    def __init__(self,db_txt,info,query,transform,S=6,distractor=True):
        with open(db_txt,'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # info
        info = np.load(info)
        self.info = []
        for i in range(len(info)):
            if distractor == False and info[i][2]==0:
                continue
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F < S:
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(S-F)
                for s in range(S):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/S)
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(interval*S-F)
                for s in range(S):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip),info[i][2],info[i][3]]))

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = len(np.unique(self.info[:,1]))
        self.n_tracklets = self.info.shape[0]
        self.query_idx = np.load(query).reshape(-1)

        if distractor == False:
            zero = np.where(info[:,2]==0)[0]
            self.new_query = []
            for i in self.query_idx:
                if i < zero[0]:
                    self.new_query.append(i)
                elif i <= zero[-1]:
                    continue
                elif i > zero[-1]:
                    self.new_query.append(i-len(zero))
                else:
                    continue
            self.query_idx = np.array(self.new_query)
                
    def __getitem__(self,idx):
        clips = self.info[idx,0]
        imgs = [self.transform(Image.open(path)) for path in self.imgs[clips[:,0]]]
        imgs = torch.stack(imgs,dim=0)
        label = self.info[idx,1]*torch.ones(1,dtype=torch.int32)
        cam = self.info[idx,2]*torch.ones(1,dtype=torch.int32)
        return imgs,label,cam

    def __len__(self):
        return len(self.info)


def Video_test_collate_fn(data):
    if isinstance(data[0],collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key,value in zip(data[0].keys(),values)}
    else:
        imgs,label,cam= zip(*data)
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(label,dim=0)
        cams = torch.cat(cam,dim=0)
        return imgs,labels,cams

def Get_Video_test_DataLoader(db_txt,info,query,transform,batch_size=10,shuffle=False,num_workers=8,S=6,distractor=True):
    dataset = Video_test_Dataset(db_txt,info,query,transform,S,distractor=distractor)
    dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=Video_test_collate_fn,shuffle=shuffle,worker_init_fn=lambda _:np.random.seed(),num_workers=num_workers)
    return dataloader




