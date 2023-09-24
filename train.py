'''Training module for the Vanilla Vision Transformer.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deeplake
import torch
from torchvision import transforms
from ViT import patch_embedding

embedding_class = patch_embedding.PatchEmbedding()
tform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((224,224))])

def collate_fn(batch):
    return {
            'images': torch.stack([x['images'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
            }

#load the dataset
places204_dataset = deeplake.load("hub://activeloop/places205")

dataloader = places204_dataset.dataloader().transform({'images':tform, 'labels':None}).batch(3).shuffle(False).pytorch(collate_fn=collate_fn, decode_method={'images':'pil'}) 


for idx, data in enumerate(dataloader):
    
    patch_embeddings = embedding_class(data['images'])
    
    


    break


def train():
    pass



if __name__ == '__main__':
    train()

