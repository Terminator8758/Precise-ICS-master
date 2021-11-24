from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable

import torch
import numpy


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def extract_cnn_feature(model, inputs, modules=None, output_ind=0):
    with torch.no_grad():
        model.eval()
        inputs = to_torch(inputs)
        #inputs = Variable(inputs)
        inputs = inputs.to('cuda')
        #print('in extract_cnn_feature(): inputs shape= {}'.format(inputs.shape))
        if modules is None:
            outputs = model(inputs)
            if outputs.__class__.__name__ in ('list','tuple'):
                outputs = outputs[output_ind]
            outputs = outputs.data.cpu()
            return outputs
        # Register forward hook for each module
        outputs = OrderedDict()
        handles = []
        for m in modules:
            outputs[id(m)] = None
            def func(m, i, o): outputs[id(m)] = o.data.cpu()
            handles.append(m.register_forward_hook(func))
        model(inputs)
        for h in handles:
            h.remove()
    return list(outputs.values())
