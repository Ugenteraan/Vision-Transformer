'''Classification mlp head module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import einops.layers.torch as einops_torch


class MLPHead(nn.Module):
    '''Final classification MLP layer.
    '''

    def __init__(self, patch_embedding_dim, num_classes):
        '''Param init.
        '''
        super(MLPHead, self).__init__()


        self.layer_norm = nn.LayerNorm(patch_embedding_dim)
        self.classification_mlp = nn.Linear(patch_embedding_dim, num_classes)


    def forward(self, x):
        extracted_cls_token = x[:, 0, :].squeeze(1) #first position in the patch num dimension. That's where the CLS token was initialized. Should be the size of [batch size, patch embedding] since we removed the 1 in the first dimension.

        x = self.layer_norm(extracted_cls_token)
        x = self.classification_mlp(x)

        return x



