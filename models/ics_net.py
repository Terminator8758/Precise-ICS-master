import torch
import torch.nn as nn
import torchvision
from torch.nn import init
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



class ModelV2(nn.Module):

    def __init__(self, class_num):
        super(ModelV2, self).__init__()

        self.class_num = class_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        features = self.gap(self.resnet_conv(x)).squeeze()
        bn = self.bottleneck(features)
        cls_score = self.classifier(bn)

        if self.training:
            return features, cls_score
        else:
            return bn


class MemoryBankModel(nn.Module):
    def __init__(self, out_dim, embeding_fea_size=2048, dropout=0.5):
        super(MemoryBankModel,self).__init__()
        
        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(out_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.embeding_memo = nn.Linear(out_dim, embeding_fea_size)
        self.embeding_memo_bn = nn.BatchNorm1d(embeding_fea_size)
        init.kaiming_normal_(self.embeding_memo.weight, mode='fan_out')
        init.constant_(self.embeding_memo.bias, 0)
        init.constant_(self.embeding_memo_bn.weight, 1)
        init.constant_(self.embeding_memo_bn.bias, 0)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gap(self.resnet_conv(x)).squeeze()
        bn = self.bottleneck(x)

        if self.training == True:
            embed_feat = self.embeding_memo(bn)
            embed_feat = self.embeding_memo_bn(embed_feat)
            embed_feat = F.normalize(embed_feat, p=2, dim=1)
            embed_feat = self.drop(embed_feat)
            return x, embed_feat  # x for triplet loss, embed_feat for ID loss
        else:
            return bn  #, embed_feat



