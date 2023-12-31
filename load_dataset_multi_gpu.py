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

    def __init__(self, token, deeplake_ds_name, batch_size, shuffle, num_workers, world_size, rank, pin_memory=True, mode='train'):
        '''Param init.
        '''

        self.token = token
        self.deeplake_ds_name = deeplake_ds_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.world_size = world_size
        self.rank = rank


    def collate_fn(self, batch_data):
        '''Custom collate function to preprocess the batch dataset.
        '''

        return {
                #we perform the grayscale to rgb conversion here since lambda func throws an error in multi gpu process.
                'images': torch.stack([x['images'].repeat(int(3/x['images'].size(0)), 1, 1) for x in batch_data]),
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
            transforms.ToTensor()

        ])



    @staticmethod
    def testing_transformation():
        return  transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
            transforms.ToTensor()
        ])

    def __call__(self):

        deeplake_dataset = deeplake.load(self.deeplake_ds_name, token=self.token)
        sampler = torch.utils.data.distributed.DistributedSampler(deeplake_dataset, num_replicas=self.world_size, rank=self.rank)


        if self.mode == 'train':
            dataloader = deeplake_dataset.pytorch(batch_size=self.batch_size, distributed=True, shuffle=self.shuffle, num_workers=self.num_workers, transform={'images':self.training_transformation(), 'labels':None}, collate_fn=self.collate_fn, decode_method={'images':'pil'}, sampler=sampler)
        else:
            dataloader = deeplake_dataset.pytorch(batch_size=self.batch_size, distributed=False, shuffle=self.shuffle, num_workers=self.num_workers, transform={'images':self.testing_transformation(), 'labels':None}, collate_fn=self.collate_fn, decode_method={'images':'pil'}) #no need sampler here since we want to test the model with all the test data.
        # dataloader = None
        # if self.mode == 'train':
        #     dataloader = deeplake_dataset.dataloader().transform({'images':self.training_transformation(), 'labels':None}).batch(self.batch_size).shuffle(self.shuffle).num_workers(self.num_workers).pin_memory(self.pin_memory).sampler(train_sampler).pytorch(collate_fn=self.collate_fn, decode_method={'images':'pil'})
        # else:
        #     dataloader = deeplake_dataset.dataloader().transform({'images':self.testing_transformation(), 'labels':None}).batch(self.batch_size).shuffle(self.shuffle).num_workers(self.num_workers).pin_memory(self.pin_memory).pytorch(collate_fn=self.collate_fn, decode_method={'images':'pil'})

        return dataloader




















