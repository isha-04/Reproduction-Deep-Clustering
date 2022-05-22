"""
Function to compute the image features of the whole dataset
"""

import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from utilities import get_average_value

def compute_features(dataloader, model, N, batch, verbose):
    if verbose:
        print('Compute features')
    batch_time = get_average_value()
    end = time.time()
    model.eval()
    
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda())
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * batch: (i + 1) * batch] = aux
        else:
            # special treatment for final batch
            features[i * batch:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                .format(i, len(dataloader), batch_time=batch_time))
    return features