import sys
sys.path.append('../')
from .dataset import *
from .loader import *
import torchvision.transforms as transforms
from reid.utils.data import transforms as T


class Loaders:

    def __init__(self, args, selected_idx=None, predicted_label=None, learning_setting='semi_supervised', triplet_sampling=True):

        self.transform_train = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # dataset
        self.selected_idx = selected_idx
        self.predicted_label = predicted_label

        self.market_path = args.market_path
        self.duke_path = args.duke_path
        self.msmt_path = args.msmt_path
        self.dataset_name = args.dataset

        # add choice of using propagate train data (without shuffle or random transformation)
        self.use_propagate_data = args.use_propagate_data

        if args.dataset == 'market1501':
            self.train_dataset = 'market_train'
        elif args.dataset == 'dukemtmc':
            self.train_dataset = 'duke_train'
        elif args.dataset == 'msmt17':
            self.train_dataset = 'msmt_train'

        # batch size
        self.p = args.class_per_batch
        self.k = args.track_per_class

        # triplet sample dataloader or random shuffle
        self.triplet_sampling = triplet_sampling

        # choose which index to obtain the label and sample triplets
        print('  learning setting is: {}'.format(learning_setting))
        if learning_setting == 'supervised':
            self.label_position = 1  # ground truth ID label
        elif learning_setting == 'semi_supervised':
            self.label_position = 5  # semi_label
        elif learning_setting == 'semi_association':
            self.label_position = 7  # predicted_label (index=7 when img_idx is appended before predicted label)

        # dataset paths
        self.samples_path = {
            'market_train': os.path.join(self.market_path, 'bounding_box_train/'),
            'market_test_query': os.path.join(self.market_path, 'query/'),
            'market_test_gallery': os.path.join(self.market_path, 'bounding_box_test/'),
            'duke_train': os.path.join(self.duke_path, 'bounding_box_train/'),
            'duke_test_query': os.path.join(self.duke_path, 'query/'),
            'duke_test_gallery': os.path.join(self.duke_path, 'bounding_box_test/'),
            'msmt_train': os.path.join(self.msmt_path, 'bounding_box_train/'),
            'msmt_test_query': os.path.join(self.msmt_path, 'query/'),
            'msmt_test_gallery': os.path.join(self.msmt_path, 'bounding_box_test/'),}

        # load
        self._load()


    def _load(self):

        # train dataset and iter
        train_samples, id_count_each_cam, img_count_each_cam, semi_label_each_cam = self._get_train_samples(self.train_dataset)
        print('  original train dataset samples: {}, id_count_each_cam= {}, img_count_each_cam= {}'.format(len(train_samples), id_count_each_cam, img_count_each_cam))
        self.id_count_each_cam = id_count_each_cam
        self.img_count_each_cam = img_count_each_cam
        self.semi_label_each_cam = semi_label_each_cam

        if self.use_propagate_data:
            self.propagate_loader = self._get_loader(train_samples, self.transform_test, 128)

        # selected train: only use selected samples to train, and add predicted label to each sample
        if (self.selected_idx is not None) and (self.predicted_label is not None):
            # select from sample's global label
            selected_train_samples = []
            for sample in train_samples:
               if sample[5] in self.selected_idx:  # global_label: sample[5], keep the position fixed!!
                    assert(self.predicted_label[sample[5]] >= 0)
                    sample.append(self.predicted_label[sample[5]])  # append predicted label
                    selected_train_samples.append(sample)
            print('  {}/{} images are selected for training.'.format(len(selected_train_samples),len(train_samples)))
            train_samples = selected_train_samples

        self.total_train_sample_num = len(train_samples)  ### count total sample number, after all operations have been done

        if self.triplet_sampling:
            self.train_iter = self._get_uniform_iter(train_samples, self.transform_train, self.p, self.k)  # train_samples:[imgs, ID, cam, Tcam, semi_label, accum_label]
        else:
            print('  Creating random-shuffled dataloader with batch size= {}'.format(self.p*self.k))
            self.train_iter = self._get_random_iter(train_samples, self.transform_train, self.p*self.k)  # train_samples:[imgs, ID, cam, Tcam, semi_label, accum_label, pred_label]

        # market test dataset and loader
        if self.dataset_name == 'market1501':
            self.market_query_samples, self.market_gallery_samples = self._get_test_samples('market_test')
            self.market_query_loader = self._get_loader(self.market_query_samples, self.transform_test, 128)
            self.market_gallery_loader = self._get_loader(self.market_gallery_samples, self.transform_test, 128)

        # duke test dataset and loader
        if self.dataset_name == 'dukemtmc':
            self.duke_query_samples, self.duke_gallery_samples = self._get_test_samples('duke_test')
            self.duke_query_loader = self._get_loader(self.duke_query_samples, self.transform_test, 128)
            self.duke_gallery_loader = self._get_loader(self.duke_gallery_samples, self.transform_test, 128)

        # msmt test dataset and loader
        if self.dataset_name == 'msmt17':
            self.msmt_query_samples, self.msmt_gallery_samples = self._get_test_samples('msmt_test')
            self.msmt_query_loader = self._get_loader(self.msmt_query_samples, self.transform_test, 128)
            self.msmt_gallery_loader = self._get_loader(self.msmt_gallery_samples, self.transform_test, 128)


    def _get_train_samples(self, train_dataset):

        train_samples_path = self.samples_path[train_dataset]
        if train_dataset == 'market_train':
            reid_samples = Samples4Market(train_samples_path, save_semi_gt_ID=True)

        elif train_dataset == 'duke_train' or 'msmt_train':  # Duke and MSMT have similar name style, so Duke data loader can be used for MSMT
            reid_samples = Samples4Duke(train_samples_path, save_semi_gt_ID=True)

        samples = reid_samples.samples
        id_count_each_cam = reid_samples.id_count_each_cam
        img_count_each_cam = reid_samples.img_count_each_cam
        semi_label_each_cam = reid_samples.semi_label_each_cam

        return samples, id_count_each_cam, img_count_each_cam, semi_label_each_cam


    def _get_test_samples(self, test_dataset):

        query_data_path = self.samples_path[test_dataset + '_query']
        gallery_data_path = self.samples_path[test_dataset + '_gallery']

        print('  query_data_path: {},  gallery_data_path: {}'.format(query_data_path, gallery_data_path))
        if test_dataset == 'market_test':
            query_samples = Samples4Market(query_data_path, reorder=False).samples  #, get_semi_label=False
            gallery_samples = Samples4Market(gallery_data_path, reorder=False).samples  # without re-order, cams=1,2,3,4,5,6
        elif test_dataset == 'duke_test' or 'msmt_test':
            query_samples = Samples4Duke(query_data_path, reorder=False).samples
            gallery_samples = Samples4Duke(gallery_data_path, reorder=False).samples

        return query_samples, gallery_samples


    def _get_uniform_iter(self, samples, transform, p, k):
        '''
        load person reid data_loader from images_folder
        and uniformly sample according to class
        :param images_folder_path:
        :param transform:
        :param p:
        :param k:
        :return:
        '''
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False, sampler=ClassUniformlySampler(dataset, class_position=self.label_position, k=k))
        iters = IterLoader(loader)

        return iters


    def _get_random_iter(self, samples, transform, batch_size):

        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        iters = IterLoader(loader)

        return iters


    def _get_random_loader(self, samples, transform, batch_size):

        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        return loader


    def _get_loader(self, samples, transform, batch_size):

        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader

