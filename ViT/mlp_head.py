'''Classification mlp head module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import einops.layers.torch as einops_torch


class MLPHead(nn.Sequential):
    '''Final classification MLP layer.
    '''

    def __init__(self, patch_embedding_dim, num_classes):
        '''Param init.
        '''

        
        super().__init(
                        einops_torch.Reduce('b n e -> b e', reduction='mean'), #from [batch size, patch num, patch embedding] -> [batch size, patch embedding] by averaging the 2nd dim.
                        nn.LayerNorm(patch_embedding_dim),
                        nn.Linear(patch_embedding_dim, num_classes)
                      )


