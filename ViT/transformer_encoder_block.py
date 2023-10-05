'''Transformer Encoder block module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from .multihead_attention import MultiHeadAttentionEinops
from .feedforward_block import FeedForwardEncoderBlock

class TransformerEncoderBlock(nn.Module):
    '''A single transformer encoder block.
    '''

    def __init__(self, patch_embedding_dim,  projection_dim_keys, projection_dim_values, num_heads, attn_dropout_prob, feedforward_projection_dim, feedforward_dropout_prob):

        super(TransformerEncoderBlock, self).__init__()
        

        self.multihead_attention_block = nn.Sequential(
                    nn.LayerNorm(patch_embedding_dim),
                    MultiHeadAttentionEinops(patch_embedding_dim=patch_embedding_dim, 
                                             projection_dim_keys=projection_dim_keys,
                                             projection_dim_values=projection_dim_values,
                                             num_heads=num_heads,
                                             attn_dropout_prob=attn_dropout_prob)
                )

        self.feedforward_block = nn.Sequential(
                    nn.LayerNorm(patch_embedding_dim),
                    FeedForwardEncoderBlock(patch_embedding_dim=patch_embedding_dim,
                                            feedforward_projection_dim=feedforward_projection_dim,
                                            feedforward_dropout_prob=feedforward_dropout_prob)
                    )





    def forward(self, x):
        '''Transformer encoder process.
        '''
        
        multihead_attention_output = self.multihead_attention_block(x)
        multihead_attention_output += x #skip connection

        feedforward_output = self.feedforward_block(multihead_attention_output)
        feedforward_output += multihead_attention_output #second skip connection

        return feedforward_output

'''
if __name__ == '__main__':
   
    x = torch.randn(2, 50, 128)

    h = TransformerEncoderBlock(patch_embedding_dim=128,  projection_dim_keys=128, projection_dim_values=128, num_heads=8, attn_dropout_prob=0, feedforward_projection_dim=128, feedforward_dropout_prob=0)

    out = h(x)
    print(out.size())
'''









