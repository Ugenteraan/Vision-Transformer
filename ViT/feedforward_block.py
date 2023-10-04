'''Feed forward block in a transformer encoder.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


class FeedForwardEncoderBlock(nn.Sequential):
    '''Feed Forward module.
    '''

    def __init__(self, patch_embedding_dim, feedforward_projection_dim, feedforward_dropout_prob):
        '''Param Init.
        '''
        
        #we can define the sequence in the super itself since we inherited the nn.sequential module.`
        super().__init__(
                nn.Linear(patch_embedding_dim, feedforward_projection_dim),
                nn.GELU(),
                nn.Dropout(feedforward_dropout_prob),
                nn.Linear(feedforward_projection_dim, patch_embedding_dim)
            )
