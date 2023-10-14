'''Module to produce patch embeddings from the given image dataset.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

import einops
import einops.layers.torch as einops_torch

from .image_patch_mlp import ImagePatchMLP
from .positional_encoder import PositionalEncoder

class PatchEmbedding(nn.Module):
    '''Responsible for dividing the given image batch into patches and produce an embedding from them.
    '''


    def __init__(self, image_height=224, image_width=224, image_channel=3, patch_size=16, device='cpu'):
        '''Param init.
        '''
        super(PatchEmbedding, self).__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.patch_embedding_dim = patch_size * patch_size * image_channel
        self.device = device

        #initialize the linear projection module.
        self.patch_linear_module = ImagePatchMLP(flattened_img_dimension=(patch_size**2)*self.image_channel, output_dimension=self.patch_embedding_dim).to(device)

        #initialize the unfold function.
        self.unfolding_func = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

    def linear_projection_patches(self, patched_image_tensors):
        '''Linearly projects the image patches.
        '''

        return self.patch_linear_module(patched_image_tensors)

    def cls_token_concat(self, linear_projected_tensors):
        '''Receives the image patches that has been linearly projected and appends a learnable parameter tensor at the start of the input tensor.

        Input:
            patched_image_tensors -- A tensor of shape [batch size, total number of patches, a single flattened image dimension].
        '''
        batch_size = linear_projected_tensors.size(0)

        cls_token = nn.Parameter(torch.randn(1, 1, self.patch_embedding_dim))
        batched_cls_token = einops.repeat(cls_token, '() n e -> b n e', b=batch_size)

        cls_concat_tensor = torch.cat([batched_cls_token, linear_projected_tensors], dim=1)

        return cls_concat_tensor


    def get_non_overlapping_patches(self, inp):
        '''Break the batch image tensors to N non-overlapping patches.
        '''

        patched_image_tensors = self.unfolding_func(inp) #this will return a tensor of shape [batch size, a single flattened image patch dimension, total number of patches]

        #pure pytorch
#        rearranged_image_tensors = patched_image_tensors.permute(0, 2, 1) #to keep things consistent with the paper, we permute the dimensions to  [batch size, total number of patches, a single flattened image patch dimension]. Also, the linear projection happens to the image dimensions, not the number of patches. So this makes more sense.

        #einops equivalent
        rearranged_image_tensors = einops_torch.Rearrange('b e p -> b p e') #change the position of the embedding (flattened image patch) and the num of patch
        rearranged_image_tensors = rearranged_image_tensors(patched_image_tensors)


        return rearranged_image_tensors



    def __call__(self, batched_tensor_images):
        '''Given a batched images in a tensor, perform the Patch Embedding and return the result.

        Input:
            batched_tensor_images --  A tensor of shape [batch size, image channel, image height, image width]
        '''

        patched_image_tensors = self.get_non_overlapping_patches(inp=batched_tensor_images)
        linear_projected_tensors = self.linear_projection_patches(patched_image_tensors=patched_image_tensors)

        cls_token_concat_tensors = self.cls_token_concat(linear_projected_tensors=linear_projected_tensors)

        positional_encoding_module = PositionalEncoder(token_length=cls_token_concat_tensors.size(1),                        output_dim=cls_token_concat_tensors.size(2), n=1000, device=self.device)
        positional_encoding_tensor = positional_encoding_module() #tensor of size [num_patches+1, flattened image patch dimension]

        #in order to perform element-wise addition to the projected tensor with the CLS token, we're gonna have to stack up the positional encoding for every element in the batch.
        #stacked_pos_enc_tensor = positional_encoding_tensor.unsqueeze(0).repeat_interleave(cls_token_concat_tensors.size(0), dim=0)

        #einops equivalent
        stacked_pos_enc_tensor = einops.repeat(positional_encoding_tensor.unsqueeze(0), '() p e -> b p e', b=patched_image_tensors.size(0))


        patch_embeddings = torch.add(cls_token_concat_tensors, stacked_pos_enc_tensor)

        return patch_embeddings

