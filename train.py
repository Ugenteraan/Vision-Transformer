'''Training module for the Vanilla Vision Transformer.
'''


import deeplake
import torch
from torchvision import transforms


tform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

def collate_fn(batch):
    return {
            'images': torch.stack([x['images'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
            }

#load the dataset
places204_dataset = deeplake.load("hub://activeloop/places205")

dataloader = places204_dataset.dataloader().transform({'images':tform, 'labels':None}).batch(5).shuffle(False).pytorch(collate_fn=collate_fn, decode_method={'images':'pil'}) 


for idx, data in enumerate(dataloader):
    print(data['images'].size())
    print(data['labels'])
    break


def train():
    pass



if __name__ == '__main__':
    train()

