'''Classification mlp head module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import einops.layers.torch as einops_torch


class MLPHead(nn.Module):
    '''Final classification MLP layer. This implementation make use of the entire output from the final transformer encoder instead of just the CLS token tensor.
    '''

    def __init__(self, patch_embedding_dim, num_classes, expansion_factor=2):
        '''Param init.
        '''
        super(MLPHead, self).__init__()


        self.classification_head = nn.Sequential(einops_torch.Reduce('b n e -> b e', reduction='mean'),
                                                 nn.LayerNorm(patch_embedding_dim),
                                                 nn.Linear(patch_embedding_dim, patch_embedding_dim*expansion_factor),
                                                 nn.GELU(),
                                                 nn.Linear(patch_embedding_dim*expansion_factor, num_classes))


    def forward(self, x):

        out = self.classification_head(x)

        return out



