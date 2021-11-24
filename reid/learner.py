import numpy as np
import random
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import ics_net
import torchvision.transforms as transforms
from reid.utils.data import transforms as T
from reid.utils.evaluators import extract_features, Evaluator
from reid.utils.dataset import get_preprocessor
from reid.utils.cross_entropy_loss import CrossEntropyLabelSmooth, SoftCrossEntropyLoss
from reid.utils.triplet_loss_stb import TripletLoss
from reid.utils.exclusive_loss import ExLoss
from reid.utils.osutils import time_now
from reid.utils.meters import CatMeter
from reid.utils.metric import cosine_dist
from reid.utils.evaluation_metrics.retrieval import PersonReIDMAP
from reid.association import propagate_label
from scipy.spatial.distance import pdist, cdist, squareform
import scipy.io as sio
import os
import os.path as osp
import sys
import time
import pickle
from bisect import bisect_right

torch.autograd.set_detect_anomaly(True)


class Learner(object):
    def __init__(self, args):

        self.DATA_NAME = args.dataset

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.base_lr = args.lr
        self.weight_decay = 5e-4
        self.milestones = args.lr_step_size

        self.max_epoch = args.n_epochs
        self.evaluate_interval = args.evaluate_interval

        self.train_bs = args.class_per_batch*args.track_per_class
        self.test_bs = 32

        self.model_type = args.model_type
        self.num_classes = args.num_class

        self.criterion_ID = CrossEntropyLabelSmooth(self.num_classes)  
        self.criterion_triplet = TripletLoss(args.margin, 'euclidean')  # default margin=0.3

        self.device = 'cuda'
        self.load_ckpt = args.load_ckpt
        self.save_model_interval = args.save_model_interval
        self.ckpt_path = args.ckpt
        self.save_model_name = args.save_model_name

        self.use_propagate_data = args.use_propagate_data
        self.associate_rate = args.associate_rate
        self.num_steps = args.num_steps  # 10
        self.alpha = args.alpha  # for learning rate warmup


    def update(self, selected_idx):
        self.selected_idx = selected_idx


    def update_dataloader(self, loaders):

        self.train_loader = loaders.train_iter
        self.id_count_each_cam = loaders.id_count_each_cam
        self.id_count_each_cam = np.array(self.id_count_each_cam)  #[652, 541, 694, 241, 576, 558]
        self.associate_class_pair = self.associate_rate*np.sum(self.id_count_each_cam)
        print('  number of ID pair to be associated: {}'.format(self.associate_class_pair))

        self.img_count_each_cam = loaders.img_count_each_cam
        self.img_count_each_cam = np.array(self.img_count_each_cam)

        self.semi_label_each_cam = loaders.semi_label_each_cam  # a list of arrays, each array contains the semi_label of each intra-cam img

        self.total_train_sample_num = loaders.total_train_sample_num
        self.all_cams = np.arange(len(self.id_count_each_cam))
        self.all_cams = torch.tensor(self.all_cams)

        if self.use_propagate_data:
            self.propagate_loader = loaders.propagate_loader

        if self.DATA_NAME == 'market1501':
            self.query_data = loaders.market_query_loader
            self.gallery_data = loaders.market_gallery_loader
            self.query_samples = loaders.market_query_samples
            self.gallery_samples = loaders.market_gallery_samples
        if self.DATA_NAME == 'dukemtmc':
            self.query_data = loaders.duke_query_loader
            self.gallery_data = loaders.duke_gallery_loader
            self.query_samples = loaders.duke_query_samples
            self.gallery_samples = loaders.duke_gallery_samples
        if self.DATA_NAME == 'msmt17':
            self.query_data = loaders.msmt_query_loader
            self.gallery_data = loaders.msmt_gallery_loader
            self.query_samples = loaders.msmt_query_samples
            self.gallery_samples = loaders.msmt_gallery_samples


    def train_supervised_an_epoch(self, epoch):

        self.network.train()
        total_id_loss, total_trip_loss = 0, 0

        ### we assume 200 iterations as an epoch
        #num_batch = int(len(self.selected_idx)/self.train_bs)  # 12936/64=202.1
        num_batch = int(self.total_train_sample_num / self.train_bs)  # num_batch = 200
        for _ in range(num_batch):

            ### load a batch data
            all_data = self.train_loader.next_one()
            imgs = all_data[0]
            pids = all_data[1]
            imgs, pids = imgs.to(self.device), pids.to(self.device)

            ### forward
            features, cls_score = self.network(imgs)

            ### loss
            ide_loss = self.criterion_ID(cls_score, pids)
            if self.use_trip_loss:
                triplet_loss = self.criterion_triplet(features, features, features, pids, pids, pids)
            else:
                triplet_loss = torch.tensor(0)

            loss = ide_loss + triplet_loss
 
            total_id_loss += ide_loss.item()
            total_trip_loss += triplet_loss.item()

            ### optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print('Time: %s,  Epoch: %d,  Trip loss: %.3f,  ID loss: %.3f'% (time_now(), epoch, total_trip_loss/num_batch, total_id_loss/num_batch))


    def train_supervised(self, step, T=10, use_trip_loss=True):
        # create model
        print('Re-creating model {}'.format(self.model_type))
        if self.model_type == 'ResNet50_STB':
            self.network = ics_net.ModelV2(class_num=self.num_classes)
        else:
            print('Unknown model type for supervised training!')

        if self.device == 'cuda':
            self.network = torch.nn.DataParallel(self.network.to(self.device))

        if self.load_ckpt is not None:
            state = torch.load(self.load_ckpt)
            self.network.load_state_dict(state)
      
        self.use_trip_loss = use_trip_loss

        # define optimizer
        self._init_optimizer()

        # start training
        for epoch in range(self.max_epoch):
            self.lr_scheduler.step(epoch)
            self.train_supervised_an_epoch(epoch)

            if (epoch + 1) % self.evaluate_interval == 0:
                print('Epoch {} evaluation ========================================'.format(epoch))
                eval_results = self.test()
                print('Step: %d, epoch: %d, rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f' % (step, epoch, eval_results[1], eval_results[2], eval_results[3], eval_results[4], eval_results[0]))
        return eval_results


    def train_semi_an_epoch(self, epoch):
        self.network.train()
        total_id_loss, total_trip_loss = 0, 0

        num_batch = int(self.total_train_sample_num / self.train_bs)
        for _ in range(num_batch):

            ### load a batch data
            all_data = self.train_loader.next_one()  # [imgs, ID, cam, Tcam, semi_label, accum_label, img_idx, predicted_label]
            ori_data = all_data
            imgs = ori_data[0].to(self.device)
            cams = ori_data[2]
            Tcams = ori_data[3]  # for original images: Tcams=-1
            #img_idx = ori_data[-1]
            class_idxs = ori_data[5]  # global label
            labels = torch.tensor(self.new_labels[class_idxs]).long().to(self.device)  # predicted label

            ### forward
            features, cls_score = self.network(imgs)

            ### loss
            ide_loss = self.criterion_ID(cls_score, labels)
            triplet_loss = self.criterion_triplet(features, features, features, labels, labels, labels)
            loss = ide_loss + triplet_loss

            total_id_loss += ide_loss.item()
            total_trip_loss += triplet_loss.item()

            ### optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print('Time: %s,  Epoch: %d,  Trip loss: %.3f,  ID loss: %.3f'
              % (time_now(), epoch, total_trip_loss/num_batch, total_id_loss/num_batch))



    def train_semi(self, step=2, new_labels=[], save_model_name='checking', run_time=0, load_model_name=None, max_epoch=120):

        # create model
        print('Re-creating model {}'.format(self.model_type))
        if self.model_type == 'ResNet50_STB':
            if len(new_labels)>0:
                self.num_classes = len(np.unique(new_labels[new_labels>=0]))
                print('new num_classes = {}'.format(self.num_classes))
            self.network = ics_net.ModelV2(class_num=self.num_classes) # ModelV2: with explicit bottleneck layer
            self.criterion_ID = CrossEntropyLabelSmooth(self.num_classes)  # nn.CrossEntropyLoss()

        if self.device == 'cuda':
            self.network = torch.nn.DataParallel(self.network.to(self.device))

        if load_model_name is not None:
            self.load_ckpt = load_model_name
        if self.load_ckpt is not None:
            print('  Loading pre-trained model: {}'.format(self.load_ckpt))
            trained_dict = torch.load(self.load_ckpt)
            filtered_trained_dict = {k: v for k, v in trained_dict.items() if
                                  ( (not k.startswith('module.embeding')) and (not k.startswith('module.classifier')))}  
            model_dict = self.network.state_dict()
            model_dict.update(filtered_trained_dict)
            self.network.load_state_dict(model_dict)

        if step == 1:
            print('Skip training for the first step (initialization)')
            print('initial model evaluation ========================================')
            eval_results = self.test()
            print('Step: %d, rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f' % (step, eval_results[1], eval_results[2], eval_results[3], eval_results[4], eval_results[0]))
        else:
            # define optimizer
            self._init_optimizer()

            # update the target label
            self.new_labels = new_labels

            # start training
            evaluate_interval = max_epoch
            save_model_interval = max_epoch

            for epoch in range(max_epoch):
                self.lr_scheduler.step(epoch)
                self.train_semi_an_epoch(epoch)

                if (epoch + 1) % evaluate_interval == 0:
                    print('Epoch {} evaluation ========================================'.format(epoch))
                    eval_results = self.test()
                    print('Step: %d, epoch: %d, rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f' % (step, epoch, eval_results[1], eval_results[2], eval_results[3], eval_results[4], eval_results[0]))

                if (epoch+1) % evaluate_interval == 0:
                    temp_save_model_name= os.path.join(self.ckpt_path, save_model_name + '_step_' + str(step) + '_epoch_' + str(epoch + 1) + '_run_time_' + str(run_time) + '.pth')
                    torch.save(self.network.state_dict(), temp_save_model_name)
                    print('  Model saved as {}'.format(temp_save_model_name))


    def train_memory_bank_an_epoch(self, epoch):

        self.network.train()
        total_id_loss, total_trip_loss, total_memo_trip_loss = 0, 0, 0
        
        num_batch = int(self.total_train_sample_num/self.train_bs)
        for _ in range(num_batch):

            ### load a batch data
            all_data = self.train_loader.next_one()  # [img, ID, cam, Tcam, percam_label, accumulate_label, img_idx_labels]
            imgs = all_data[0].to(self.device)
            cams = all_data[2]
            #Tcams = all_data[3]  #.to(self.device)  # for original images: Tcams=-1
            labels = all_data[4].to(self.device)  #all_data[3].to(self.device)
            accum_labels = all_data[5]    #.to(self.device)
            img_index = all_data[6].to(self.device)

            ### forward
            all_output = self.network(imgs)
            features = all_output[0]
            embed_feat = all_output[1]

            # cache the per-camera memory bank before each batch training
            if self.memory_trip_loss:
                percam_V = []
                for ii in range(len(self.all_cams)):
                    percam_V.append(self.criterion_memo[ii].V.detach().clone())

            ### loss
            loss = 0  # reset at each batch

            # triplet loss + ID loss
            for ii in range(len(self.all_cams)):  # all_cams=[0,1,2,3,4,5], make sure all_cams contains all actual cam values
                target_cam = self.all_cams[ii]

                if torch.nonzero(cams == target_cam).size(0) > 0:
                    percam_feat = features[cams == target_cam]
                    percam_memo_feat = embed_feat[cams == target_cam]
                    percam_label = labels[cams == target_cam]

                    # GAP triplet loss
                    if self.gap_trip_loss:
                        triplet_loss = self.criterion_triplet(percam_feat, percam_feat, percam_feat, percam_label, percam_label, percam_label)
                        total_trip_loss += triplet_loss
                        loss += triplet_loss

                    # ID-memory classification loss
                    ide_loss = self.criterion_memo[ii](percam_memo_feat, percam_label, miu=self.miu)
                    total_id_loss += ide_loss
                    loss += ide_loss

                    # ID-memory-based triplet loss
                    if self.memory_trip_loss:
                        cam_memory_label = torch.arange(self.id_count_each_cam[ii]).long().to(self.device)
                        memo_trip_loss = self.criterion_triplet(percam_memo_feat, percam_V[ii].to(self.device), percam_V[ii].to(self.device), percam_label, cam_memory_label, cam_memory_label)          
                        total_memo_trip_loss += memo_trip_loss 
                        loss += memo_trip_loss

            ### optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()                                    
        print('Time: %s,  Epoch: %d,  Trip loss: %.3f, Classification loss: %.3f, Memory trip loss: %.3f'
            % (time_now(), epoch, total_trip_loss/num_batch, total_id_loss/num_batch, total_memo_trip_loss/num_batch))


    def train_memory_bank(self, step, run_time=0, miu=0.5, t=15, gap_trip_loss=True, memory_trip_loss=True):

        self.miu = miu
        self.T = t
        self.memory_trip_loss = memory_trip_loss
        self.gap_trip_loss = gap_trip_loss 

        # create model
        print('Re-creating memory bank model')
        self.network = ics_net.MemoryBankModel(out_dim=2048, embeding_fea_size=2048)
        self.criterion_memo = []
        for i in range(len(self.id_count_each_cam)):
            self.criterion_memo.append(ExLoss(2048, self.id_count_each_cam[i], t=t, label_smoothing=False).to(self.device))

        if self.device == 'cuda':
            self.network = torch.nn.DataParallel(self.network.to(self.device))

        if self.load_ckpt is not None:
            state = torch.load(self.load_ckpt)
            self.network.load_state_dict(state)

        if step == 0:
            print('Skip training for step-0 (initialization)')
            print('Pretrained model evaluation ========================================')
            eval_results = self.test()
            print('Step: %d, rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f' % (step, eval_results[1], eval_results[2], eval_results[3], eval_results[4], eval_results[0]))
        else:
            # define optimizer
            self._init_optimizer()

            # start training
            for epoch in range(self.max_epoch):

                self.lr_scheduler.step(epoch)
                self.train_memory_bank_an_epoch(epoch)

                if (epoch + 1) % self.evaluate_interval == 0:
                    print('Epoch {} evaluation ========================================'.format(epoch))
                    eval_results = self.test()
                    print('Step: %d, epoch: %d, rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f' % (step, epoch, eval_results[1], eval_results[2], eval_results[3], eval_results[4], eval_results[0]))
                if (epoch+1) % self.save_model_interval == 0:
                    temp_save_model_name = os.path.join(self.ckpt_path, self.save_model_name + '_step1_epoch_' +str(epoch + 1) + '_run_time_' + str(run_time) + '.pth')
                    torch.save(self.network.state_dict(), temp_save_model_name)
                    print('  Model saved as {}'.format(temp_save_model_name))
 

    def _init_optimizer(self):
        params = []
        for key, value in self.network.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.base_lr
            weight_decay = self.weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        self.optimizer = optim.Adam(params)
        self.lr_scheduler = WarmupMultiStepLR(self.optimizer, self.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)


    def test(self):

        self.network.eval()

        # meters
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

        # init dataset
        loaders = [self.query_data, self.gallery_data]

        # compute query and gallery features
        with torch.no_grad():
            for loader_id, loader in enumerate(loaders):
                for data in loader:
                    # compute feautres
                    # images, pids, cids = data
                    images = data[0]
                    pids = data[1]
                    cids = data[2]
                    features = self.network(images)
                    # save as query features
                    if loader_id == 0:
                        query_features_meter.update(features.data)
                        query_pids_meter.update(pids)
                        query_cids_meter.update(cids)
                    # save as gallery features
                    elif loader_id == 1:
                        gallery_features_meter.update(features.data)
                        gallery_pids_meter.update(pids)
                        gallery_cids_meter.update(cids)
        #
        query_features = query_features_meter.get_val_numpy()
        gallery_features = gallery_features_meter.get_val_numpy()

        # compute mAP and rank@k
        result = PersonReIDMAP(
            query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
            gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy(), dist='cosine')

        return result.mAP, result.CMC[0], result.CMC[4], result.CMC[9], result.CMC[19]



    def img_association(self, step):

        self.network.eval()
        print('Start Inference...')
        features = []
        global_labels = []
        ID_labels = []
        all_cams = []

        with torch.no_grad():
            for c, data in enumerate(self.propagate_loader):
                images = data[0]  # .cuda()
                label = data[1]  # ground truth ID label
                cams = data[2]
                g_label = data[5]
 
                feat = self.network(images)  # .detach().cpu().numpy() #[xx,128]
                features.append(feat.cpu())
                ID_labels.append(label)
                all_cams.append(cams)
                global_labels.append(g_label)

        features = torch.cat(features, dim=0).numpy()
        ID_labels = torch.cat(ID_labels, dim=0).numpy()
        all_cams = torch.cat(all_cams, dim=0).numpy()
        global_labels = torch.cat(global_labels, dim=0).numpy()
        print('  features: shape= {}'.format(features.shape))  # 1955 X 2048

        # compress intra-camera same-ID image features by average pooling
        new_features, new_cams, new_IDs = [], [], []
        for glab in np.unique(global_labels):
            idx = np.where(global_labels==glab)[0]
            new_features.append(np.mean(features[idx],axis=0))
               
            assert (len(np.unique(all_cams[idx])) == 1)
            assert (len(np.unique(ID_labels[idx])) == 1)
            new_cams.append(all_cams[idx[0]])
            new_IDs.append(ID_labels[idx[0]])

        new_features = np.array(new_features)
        new_cams = np.array(new_cams)
        new_IDs = np.array(new_IDs) 
        del features, ID_labels, all_cams

        # compute distance and association
        new_features = new_features / np.linalg.norm(new_features, axis=1, keepdims=True)  # n * d -> n
        W = cdist(new_features, new_features, 'sqeuclidean')
        print('  distance matrix: shape= {}'.format(W.shape))  # should be (num_total_classes, num_total_classes)
        
        # self-similarity for association
        print('  this is step {} of total {} steps...'.format(step, self.num_steps))
        updated_label = propagate_label(W, new_IDs, new_cams, self.associate_class_pair, step, self.num_steps)

        return updated_label


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]




