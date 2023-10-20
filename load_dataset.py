'''Module to load & transform dataset.
'''

import deeplake
import torch
from torchvision import transforms

import cred
import cfg




class LoadDeeplakeDataset:
    '''Load a dataset from the Deeplake API.
    '''

    def __init__(self, token, deeplake_ds_name, batch_size, shuffle, mode='train'):
        '''Param init.
        '''

        self.token = token
        self.deeplake_ds_name = deeplake_ds_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode


    def collate_fn(self, batch_data):
        '''Custom collate function to preprocess the batch dataset.
        '''
        return {
                'images': torch.stack([x['images'] for x in batch_data]),
                'labels': torch.stack([torch.from_numpy(x['labels']) for x in batch_data])
            }

    @staticmethod
    def training_transformation():

        return transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=cfg.COLOR_JITTER_BRIGHTNESS, hue=cfg.COLOR_JITTER_HUE),
            transforms.RandomAffine(degrees=cfg.RANDOM_AFFINE_ROTATION_RANGE, translate=cfg.RANDOM_AFFINE_TRANSLATE_RANGE, scale=cfg.RANDOM_AFFINE_SCALE_RANGE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1))
        ])

    @staticmethod
    def testing_transformation():
        return  transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1))
        ])

    def __call__(self):

        deeplake_dataset = deeplake.load(self.deeplake_ds_name, token=self.token)

        dataloader = None
        if self.mode == 'train':
            dataloader = deeplake_dataset.dataloader().transform({'images':self.training_transformation(), 'labels':None}).batch(self.batch_size).shuffle(self.shuffle).pytorch(collate_fn=self.collate_fn, decode_method={'images':'pil'})
        else:
            dataloader = deeplake_dataset.dataloader().transform({'images':self.testing_transformation(), 'labels':None}).batch(self.batch_size).shuffle(self.shuffle).pytorch(collate_fn=self.collate_fn, decode_method={'images':'pil'})

        return dataloader



















