import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import random
import os.path as osp



def get_dataset(name):
    if name == 'MNIST':
        return get_MNIST()
    elif name == 'FashionMNIST':
        return get_FashionMNIST()
    elif name == 'SVHN':
        return get_SVHN()
    elif name == 'CIFAR10':
        return get_CIFAR10()

def get_MNIST():
    raw_tr = datasets.MNIST('./MNIST', train=True, download=True)
    raw_te = datasets.MNIST('./MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST():
    raw_tr = datasets.FashionMNIST('./FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST('./FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN():
    data_tr = datasets.SVHN('./SVHN', split='train', download=True)
    data_te = datasets.SVHN('./SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10():
    data_tr = datasets.CIFAR10('./data', train=True, download=True)
    data_te = datasets.CIFAR10('./data', train=False, download=True)
    X_tr = data_tr.train_data
    Y_tr = torch.from_numpy(np.array(data_tr.train_labels))
    X_te = data_te.test_data
    Y_te = torch.from_numpy(np.array(data_te.test_labels))
    return X_tr, Y_tr, X_te, Y_te

def get_preprocessor(name):
    if name == 'MNIST':
        return Preprocessor1
    elif name == 'FashionMNIST':
        return Preprocessor1
    elif name == 'SVHN':
        return Preprocessor2
    elif name == 'CIFAR10':
        return Preprocessor3
    elif name == 'market1501':
        return Preprocessor_market



class Preprocessor1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class Preprocessor2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class Preprocessor3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)




class Preprocessor_market(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor_market, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            print('batch indices are tuple (list)...')
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, videoid = self.dataset[index]
        fname = fname[0]  # fname eg:  (u'0075_03_0192_0000.jpg',)

        fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # for image dataset like market: video_frames shape=(1,h,w)
        # for video dataset like mars: video_frames shape=(selected_frames_num, h, w)
        pid = int(pid)
        camid = int(camid)
        return img, pid, camid    #, index, videoid


    '''
    def _get_single_item(self, index):
        images, pid, camid, videoid = self.dataset[index]
        image_str = "".join(images)

        # random select images if training
        if self.is_training:
            if len(images) >= self.selected_frames_num:
                images = random.sample(images, self.selected_frames_num)
            else:
                images = random.choices(images,  k=self.selected_frames_num)
            images.sort()

        else: # for evaluation, we use all the frames
            if len(images) > self.max_frames:  # to avoid the insufficient memory
                images = random.sample(images, self.max_frames)

        video_frames = []
        for fname in images:
            if self.root is not None:
                fpath = osp.join(self.root, fname)
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            video_frames.append(img)

        video_frames = torch.stack(video_frames, dim=0)
        # for image dataset like market: video_frames shape=(1,h,w)
        # for video dataset like mars: video_frames shape=(selected_frames_num, h, w)
        pid = int(pid)
        return video_frames, image_str, pid, index, videoid
    '''
