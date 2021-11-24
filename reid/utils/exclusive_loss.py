from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd
#from reid.utils.cross_entropy_loss import SoftCrossEntropyLoss


class Exclusive(autograd.Function):
    def __init__(self, V, miu=0.5):
        super(Exclusive, self).__init__()
        self.V = V
        self.miu = miu

    def forward(self, inputs, targets):  # When forward takes N params, autograd expects backward to return N parameters
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.V.t())
        return outputs

    '''
    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = grad_outputs.mm(self.V) if self.needs_input_grad[0] else None
        uniq_y = torch.unique(targets)
        for y in uniq_y:
            match_ind = (targets==y)
            img_num = len(torch.nonzero(match_ind))
            mean_x = (inputs[match_ind]).mean(0)
            mean_x = F.normalize(mean_x, p=2, dim=0)  # normalize the averaged feature
            weighted_miu = self.miu**img_num    #self.miu*img_num/(1+img_num)
            self.V[y] = F.normalize( weighted_miu*self.V[y] + (1-weighted_miu)*mean_x, p=2, dim=0 )
        return grad_inputs, None    #, None
    '''
    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = grad_outputs.mm(self.V) if self.needs_input_grad[0] else None
        for x, y in zip(inputs, targets):
            self.V[y] = F.normalize( self.miu*self.V[y] + (1-self.miu)*x, p=2, dim=0)
        return grad_inputs, None    #, None
    


class ExLoss(nn.Module):
    def __init__(self, num_features, num_classes, t=1.0,
                 weight_mask=None, label_smoothing=False, batch_average=True):
        super(ExLoss, self).__init__()
        self.num_features = num_features
        self.t = t
        self.weight_mask = weight_mask
        self.register_buffer('V', torch.zeros(num_classes, num_features))
        self.epsilon = 0.1
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.logsoftmax = nn.LogSoftmax(dim=1)
        if batch_average:
            self.reduction = 'mean'
        else:
            self.reduction = 'none'

    def forward(self, inputs, targets, miu=0.5):
        outputs = Exclusive(self.V, miu)(inputs, targets) * self.t
        
        if self.label_smoothing:
            log_probs = self.logsoftmax(outputs)
            targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            targets = targets.to(torch.device('cuda'))
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = F.cross_entropy(outputs, targets, weight=self.weight_mask, reduction=self.reduction)
        return loss  #, outputs
    
    #def forward(self, inputs, targets, Tcams, miu=0.5):
    #    #print('  in ExLoss: Tcams= {}'.format(Tcams))
    #    outputs = Exclusive(self.V, miu)(inputs, targets, Tcams) * self.t
    #    loss = F.cross_entropy(outputs, targets, weight=self.weight_mask)
    #    return loss  #, outputs


class SoftExLoss(nn.Module):
    def __init__(self, num_features, num_classes, t=1.0,
                 weight_mask=None):
        super(SoftExLoss, self).__init__()
        self.num_features = num_features
        self.t = t
        self.weight_mask = weight_mask
        self.register_buffer('V', torch.zeros(num_classes, num_features))
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets1, targets2, miu=0.5):
        # targets1: global label, target2: updated soft label
        outputs = Exclusive(self.V, miu)(inputs, targets1) * self.t

        log_probs = self.logsoftmax(outputs)
        targets2 = targets2.cuda()
        loss = (- targets2 * log_probs).mean(0).sum()  #SoftCrossEntropyLoss(outputs, targets2)
        return loss




