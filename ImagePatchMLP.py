'''Module to linearly project the image patches.
'''

import torch
import torch.nn as nn



class ImagePatchMLP(nn.Module):
    '''A single fully-connected layer to project the flattened image patches into a lower/higher dimensional space.
    '''

    def __init__(self):
        '''Param init.
        '''


