'''Module to linearly project the image patches.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn



class ImagePatchMLP(nn.Module):
    '''A single fully-connected layer to project the flattened image patches into a lower/higher dimensional space.
    '''

    def __init__(self, flattened_img_dimension, output_dimension):
        '''Param init.
        '''
    
        super(ImagePatchMLP, self).__init__()
        
        self.patch_linear_layer = nn.Linear(in_features=flattened_img_dimension, out_features=output_dimension)
        
        
    def forward(self, x):
        '''Linearly projects the image patches (no activation function) into another vector space.
        
        Input:
           x -- A tensor of shape [batch size, total number of patches, a single flattened image dimension]

        The projection will preserve the total number of patches.
        '''

        return self.patch_linear_layer(x)

        



