'''Multi Head Attention Mechanism module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn




class SelfAttention(nn.Module):
    '''Vanilla Vision Transformers Self Attention Mechanism.
    '''

    
    def __init__(self, patch_dims, projection_dim):
        '''Param init.
        '''

        super(SelfAttention, self).__init__()
        
        self.projection_dim = projection_dim

        self.W_q = nn.Linear(in_features=patch_dims, out_features=self.projection_dim)
        self.W_k = nn.Linear(in_features=patch_dims, out_features=self.projection_dim)
        self.W_v = nn.Linear(in_features=patch_dims, out_features=self.projection_dim)



    def forward(self, X):
        '''Perform self-attention.
        '''

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        dot_product = torch.matmul(Q, K.transpose())
        scaled_dot_product = torch.div(dot_product, torch.sqrt(self.projection_dim))

        softmax_scaled_dot_product = nn.Softmax(scaled_dot_product, dim=-1)
        
        attention = torch.matmul(softmax_scaled_dot_product, V)

        return attention
