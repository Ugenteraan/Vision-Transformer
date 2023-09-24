'''Multi Head Attention Mechanism module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn




class MultiHeadAttention(nn.Module):
    '''Vanilla Vision Transformers Multi Head Attention Mechanism.
    '''

    
    def __init__(self, patch_dims, num_heads, projection_dim):
        '''Param init.
        '''

        super(MultiHeadAttention, self).__init__()

        

