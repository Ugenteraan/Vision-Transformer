'''Transformer Encoder Network.
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn

from .transformer_encoder_block import TransformerEncoderBlock

class TransformerEncoderNetwork(nn.Sequential):
    '''Creates multiple transformer encoders to form a network.
    '''

    def __init__(self, transformer_network_depth, patch_embedding_dim, **kwargs):
        '''Use nn.Sequential to create the network.
        '''
        
        super().__init__(*[TransformerEncoderBlock(patch_embedding_dim=patch_embedding_dim, **kwargs) for _ in range(transformer_network_depth)])


