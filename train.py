'''Training module for the Vanilla Vision Transformer.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import deeplake
import torch
import torch.nn as nn
from torchvision import transforms

from ViT.ViT import VisionTransformer
from load_dataset import LoadDeeplakeDataset
import cred
import cfg


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and cfg.DEVICE == 'gpu' else 'cpu')

MODEL = VisionTransformer(image_height=cfg.IMAGE_HEIGHT,
                          image_width=cfg.IMAGE_WIDTH,
                          image_channel=cfg.IMAGE_CHANNEL,
                          patch_size=cfg.PATCH_SIZE,
                          transformer_network_depth=cfg.TRANSFORMER_NETWORK_DEPTH,
                          num_classes=cfg.NUM_CLASSES,
                          projection_dim_keys=cfg.PROJECTION_DIM_KEYS,
                          projection_dim_values=cfg.PROJECTION_DIM_VALUES,
                          num_heads=cfg.NUM_HEADS,
                          attn_dropout_prob=cfg.ATTN_DROPOUT_PROB,
                          feedforward_projection_dim=cfg.FEEDFORWARD_PROJECTION_DIM,
                          feedforward_dropout_prob=cfg.FEEDFORWARD_DROPOUT_PROB)

CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
SCHEDULER = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SCHEDULER_STEP_SIZE, GAMMA=cfg.SCHEDULER_GAMMA)

TRAIN_DATALOADER = LoadDeeplakeDataset(token=cred.ACTIVELOOP_TOKEN, deeplake_ds_name="hub://activeloop/stanford-cars-train", batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE)()
TEST_DATALOADER = LoadDeeplakeDataset(token=cred.ACTIVELOOP_TOKEN, deeplake_ds_name="hub://activeloop/stanford-cars-test", batch_size=cfg.BATCH_SIZE, shuffle=False)()


for epoch_idx in tqdm(range(cfg.TRAIN_EPOCH)):
    
    MODEL.train() #set the model to training mode.
    for idx, data in enumerate(TRAIN_DATALOADER):
    
        X,Y = data['images'].to(DEVICE), data['labels'].to(DEVICE) 
        
        




def train():
    pass



if __name__ == '__main__':
    train()

