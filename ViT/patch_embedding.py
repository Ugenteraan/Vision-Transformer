'''Module to produce patch embeddings from the given image dataset.
'''
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    '''Responsible for dividing the given image batch into patches and produce an embedding from them.
    '''


    def __init__(self, image_height=224, image_width=224, image_channel=1, patch_size=16):
        '''Param init.
        '''
        super(PatchEmbedding, self).__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.stride = stride


        #initialize the unfold function.
        self.unfolding_func = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
    


    def get_non_overlapping_patches(self, inp):
        '''Break the batch image tensors to N non-overlapping patches.
        '''
        
        patched_image_tensors = self.unfolding_func(inp) #this will return a tensor of shape [batch size, a single flattened image dimension, total number of patches]
        patched_image_tensors = patched_image_tensors.permute(0, 2, 1) #to keep things consistent with the paper, we permute the dimensions to  [batch size, total number of patches, a single flattened image dimension]
        
        return patched_image_tensors


    
    def __call__(self, batched_tensor_images):
        '''Given a batched images in a tensor, perform the Patch Embedding and return the result.
        
        Input:
            batched_tensor_images --  A tensor of shape [batch size, image channel, image height, image width]
        '''

        patched_image_tensors = self.get_non_overlapping_patches(inp=batched_tensor_images)


