'''Feed forward block in a transformer encoder.
'''


import torch
import torch.nn as nn


class FeedForwardEncoderBlock(nn.Sequential):
    '''Feed Forward module.
    '''

    def __init__(self, patch_embedding_dimension, feedforward_projection_dimension, feedforward_dropout_prob):
        '''Param Init.
        '''
        
        #we can define the sequence in the super itself since we inherited the nn.sequential module.`
        super().__init__(
                nn.Linear(patch_embedding_dimension, feedforward_projection_dimension),
                nn.GELU(),
                nn.Dropout(feedforward_dropout_prob),
                nn.Linear(feedforward_projection_dimension, patch_embedding_dimension)
            )
