'''Multi Head Attention Mechanism module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torch.nn as nn
import einops.layers.torch as einops_torch



class MultiHeadAttentionTorch(nn.Module):
    '''Vanilla Vision Transformers Attention Mechanism without Einops. Einops version is also implemented in this class but commented out. We want to use the same nn.Linear weights for the verification.
    '''

    def __init__(self, patch_embedding_dim, projection_dim_keys, projection_dim_values, num_heads, attn_dropout_prob):
        '''Param Init.
        '''
        super(MultiHeadAttentionTorch, self).__init__()
        
        self.patch_embedding_dim = patch_embedding_dim
        self.projection_dim_keys = projection_dim_keys
        self.projection_dim_values = projection_dim_values
        self.num_heads = num_heads

        self.Wq_Wk = nn.Linear(patch_embedding_dim, projection_dim_keys*2) #weights to project the last dimension of the input tensor to a projected dimension for the query and keys. The multiplying by 2 is to project both query and key simultaneously.
        self.Wv = nn.Linear(patch_embedding_dim, projection_dim_values) #weights to project the last dimension of the input tensor to a projected dimension for the values.
        
        self.attention_dropout = nn.Dropout(attn_dropout_prob)
        self.W_o = nn.Linear(projection_dim_values, patch_embedding_dim)

        '''einops version for verification. 
        self.einops_rearrange_qk = einops_torch.Rearrange('b n (h e qk) -> (qk) b h n e', h=self.num_heads, qk=2)
        self.einops_rearrange_v = einops_torch.Rearrange('b n (h e k) -> k b h n e', h=self.num_heads, k=1) #this can be done without the k too since it's just 1.

        self.einops_mhsa_concat = einops_torch.Rearrange('b h n e -> b n (h e)') #for the concatenation of the heads.
        '''

    def forward(self, X):

        qk = self.Wq_Wk(x)
        v = self.Wv(X)

        qk_reshaped = qk.reshape(2, 50, self.num_heads, self.projection_dim_keys//self.num_heads, 2)
        v_reshaped = v.reshape(2, 50, self.num_heads, self.projection_dim_keys//self.num_heads, 1)

        qk_rearranged = qk_reshaped.permute(4, 0, 2, 1, 3)
        v_rearranged = v_reshaped.permute(4, 0, 2, 1, 3)

        queries, keys = qk_rearranged[0], qk_rearranged[1]
        values = v_rearranged[0]
        
        dot_projection = torch.matmul(queries, keys.transpose(3, 2))
        
        scaling_factor = self.projection_dim_keys ** 0.5

        scaled_dot_projection = F.softmax(dot_projection, dim=-1)/scaling_factor
        scaled_dot_projection = self.attention_dropout(scaled_dot_projection)

        attention_qkv = torch.matmul(scaled_dot_projection, values)

        permute_head = torch.permute(attention_qkv, (0, 2, 1, 3))
        multi_head_concat = permute_head.reshape(permute_head.size(0), permute_head.size(1), -1)

        multi_head_projection = self.W_o(multi_head_concat)
        

        '''Einops Version for verification.
        qk_rearranged_einops = self.einops_rearrange_qk(qk)
        v_rearranged_einops = self.einops_rearrange_v(v)
        Q, K = qk_rearranged_einops[0], qk_rearranged_einops[1]
        V = v_rearranged[0]

        dot_projection_einops = torch.einsum('bhqd, bhkd -> bhqk', Q, K) 
        
        scaled_dot_projection_einops = F.softmax(dot_projection_einops, dim=-1)/scaling_factor
        scaled_dot_projection_einops = self.attention_dropout(scaled_dot_projection_einops)

        attention_qkv_einops = torch.einsum('bhsl, bhlv -> bhsv', scaled_dot_projection_einops, V)
        multi_head_concat_einops = self.einops_mhsa_concat(attention_qkv_einops)

        multi_head_projection_einops = self.W_o(multi_head_concat)

        return multi_head_projection, multi_head_projection_einops 
        '''
        return multi_head_projection


class MultiHeadAttentionEinops(nn.Module):
    '''Implementation of MHSA with Einops and Einsum.
    '''


    def __init__(self, patch_embedding_dim, projection_dim_keys, projection_dim_values, num_heads, attn_dropout_prob):
        '''Param Init.
        '''
        super(MultiHeadAttentionEinops, self).__init__()
        
        self.patch_embedding_dim = patch_embedding_dim
        self.projection_dim_keys = projection_dim_keys
        self.projection_dim_values = projection_dim_values
        self.num_heads = num_heads

        self.Wq_Wk = nn.Linear(patch_embedding_dim, projection_dim_keys*2) #weights to project the last dimension of the input tensor to a projected dimension for the query and keys. The multiplying by 2 is to project both query and key simultaneously.
        self.Wv = nn.Linear(patch_embedding_dim, projection_dim_values) #weights to project the last dimension of the input tensor to a projected dimension for the values.
        
        self.attention_dropout = nn.Dropout(attn_dropout_prob)
        
        self.W_o = nn.Linear(projection_dim_values, patch_embedding_dim)
        
        #b: batch size
        #n: number of patches
        #h: number of heads
        #e: each patch embedding 
        #qk: query and key concatenated together.
        self.einops_rearrange_qk = einops_torch.Rearrange('b n (h e qk) -> (qk) b h n e', h=self.num_heads, qk=2)
        self.einops_rearrange_v = einops_torch.Rearrange('b n (h e k) -> k b h n e', h=self.num_heads, k=1) #this can be done without the k too since it's just 1.
        
        self.einops_mhsa_concat = einops_torch.Rearrange('b h n e -> b n (h e)') #for the concatenation of the heads.
    
    def forward(self, X):
        
        qk = self.Wq_Wk(x)
        v = self.Wv(x)
        qk_rearranged = self.einops_rearrange_qk(qk)
        v_rearranged = self.einops_rearrange_v(v)

        queries, keys = qk_rearranged[0], qk_rearranged[1]
        values = v_rearranged[0] 

        dot_projection = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        scaling_factor = self.projection_dim_keys ** 0.5
        
        #we want to only softmax the last dimension (the embeddings).
        scaled_dot_projection = F.softmax(dot_projection, dim=-1)/scaling_factor #it doesnt matter if we softmax first before the scaling or the other way around. Result will still be same.
        scaled_dot_projection = self.attention_dropout(scaled_dot_projection) #dropout
        
        attention_qkv = torch.einsum('bhsl, bhlv -> bhsv', scaled_dot_projection, values)

        multi_head_concat = self.einops_mhsa_concat(attention_qkv)

        multi_head_projection = self.W_o(multi_head_concat)

        return multi_head_projection 


'''code below was used to verify the output from both the implementations.
if __name__ == '__main__':
    
    x = torch.randn(2, 50, 128)
    mhsa = MultiHeadAttentionTorch(patch_embedding_dim=128, projection_dim_keys=128, projection_dim_values=128, num_heads=8, attn_dropout_prob=0)
    X, Y = mhsa(x)
    print(X.size(), Y.size())

    equal = torch.isclose(X, Y)
    print(equal)
'''
