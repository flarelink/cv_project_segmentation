# =============================================================================
# loss.py - loss function from reference for semantic segmenting images
# References:
# - https://github.com/meetshah1995/pytorch-semseg
# =============================================================================

import torch.nn.functional as F
import matplotlib.pyplot as plt

# make sure user imports only the functions
__all__ = ['cross_entropy2d', 'loss_plotter']

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target.long(), weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def loss_plotter(loss_list, name):
    plt.figure(1)
    plt.title('Loss plot')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.plot(loss_list)
    plt.savefig('./loss_plots/' + name + '_loss_plot.png')
    plt.close('all')