from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import tqdm

import torch
from torch.nn import functional as F
from torch.backends import cudnn

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features["".join(elem[0])].unsqueeze(0) for elem in query], 0)
    y = torch.cat([features["".join(elem[0])].unsqueeze(0) for elem in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        #query_ids = [pid for _, pid, _, _ in query]
        #gallery_ids = [pid for _, pid, _, _ in gallery]
        #query_cams = [cam for _, _, cam, _ in query]
        #gallery_cams = [cam for _, _, cam, _ in gallery]
        query_ids = [elem[1] for elem in query]
        gallery_ids = [elem[1] for elem in gallery]
        query_cams = [elem[2] for elem in query]
        gallery_cams = [elem[2] for elem in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
#        'allshots': dict(separate_camera_set=False,
#                         single_gallery_shot=False,
#                         first_match_break=False),
#        'cuhk03': dict(separate_camera_set=True,
#                       single_gallery_shot=True,
#                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['market1501'][0]


def extract_features(model, data_loader, squeeze_input=False):
    cudnn.benchmark = False
    model.eval()

    features = OrderedDict()
    labels = OrderedDict()

    with tqdm.tqdm(total=len(data_loader)) as pbar:
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            if squeeze_input:
                imgs = imgs.squeeze(dim=1)  # added by wml, remove the 2nd dimension for image datasets like market1501

            outputs = extract_cnn_feature(model, imgs)
            #outputs = F.normalize(outputs, p=2, dim=1)  # added by wml

            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid
            pbar.update(1)

    print("Extract {} batch videos".format(len(data_loader)))
    cudnn.benchmark = True
    return features, labels




class Evaluator(object):
    def __init__(self, model, ):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, squeeze_input=False, metric=None):

        features, _ = extract_features(self.model, data_loader, squeeze_input=squeeze_input)
        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery)


