'''Combines all the modules to create the vanilla vision transformer architecture.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn

from .patch_embedding import PatchEmbedding
from .transformer_encoder import TransformerEncoderNetwork
from .mlp_head import MLPHead


class VisionTransformer(nn.Sequential):
    '''Full ViT architecture.
    '''

    def __init__(self, image_height, image_width, image_channel, patch_size, transformer_network_depth, num_classes, device, **kwargs):
        '''Combines all the modules in sequence.
        '''
        patch_embedding_dim = patch_size*patch_size*image_channel
        super().__init__(
                PatchEmbedding(image_height=image_height, image_width=image_width, image_channel=image_channel, patch_size=patch_size, device=device).to(device),
                TransformerEncoderNetwork(transformer_network_depth=transformer_network_depth, patch_embedding_dim=patch_embedding_dim, device=device, **kwargs).to(device),
                MLPHead(patch_embedding_dim=patch_embedding_dim, num_classes=num_classes).to(device)
                )

