import torch
from torch import nn
from torch.autograd import Variable


class SoftCrossEntropyLoss(nn.Module):
    """Cross entropy loss with soft label as target
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=False, batch_average=True):
        super(SoftCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.label_smooth = label_smooth
        self.batch_average = batch_average

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        #targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        if self.label_smooth:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(1)  # loss.sum(1).mean() is the same as: loss.mean(0).sum()
        if self.batch_average:
            loss = loss.mean()
        return loss



class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True, batch_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.label_smooth = label_smooth
        self.batch_average = batch_average

    def forward(self, inputs, targets):
        """
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.to(torch.device('cuda'))
        if self.label_smooth:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(1)    #mean(0).sum()
        if self.batch_average:
            loss = loss.mean()
        return loss
        
