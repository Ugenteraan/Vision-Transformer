'''Training module for the Vanilla Vision Transformer.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import deeplake
import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary

from ViT.ViT import VisionTransformer
from load_dataset import LoadDeeplakeDataset
import cred
import cfg
import utils


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
                          feedforward_dropout_prob=cfg.FEEDFORWARD_DROPOUT_PROB,
                          device=DEVICE).to(DEVICE)

CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=cfg.LEARNING_RATE)
SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER, step_size=cfg.SCHEDULER_STEP_SIZE, gamma=cfg.SCHEDULER_GAMMA)

TRAIN_DATALOADER = LoadDeeplakeDataset(token=cred.ACTIVELOOP_TOKEN, deeplake_ds_name="hub://activeloop/stanford-cars-train", batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE)()
TEST_DATALOADER = LoadDeeplakeDataset(token=cred.ACTIVELOOP_TOKEN, deeplake_ds_name="hub://activeloop/stanford-cars-test", batch_size=cfg.BATCH_SIZE, shuffle=False)()

summary(MODEL, (cfg.IMAGE_CHANNEL, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))

def main():

    for epoch_idx in tqdm(range(cfg.TRAIN_EPOCH)):

        train_epoch_accuracy = 0
        train_epoch_loss = 0
        MODEL.train() #set the model to training mode.

        for idx, data in enumerate(TRAIN_DATALOADER):

            train_X, train_Y = data['images'].to(DEVICE), data['labels'].to(DEVICE)

            OPTIMIZER.zero_grad() #clear the optimizer.

            train_predictions = MODEL(train_X)
            train_batch_loss = CRITERION(train_predictions, train_Y.reshape(-1))
            train_batch_loss.backward()
            OPTIMIZER.step()


            train_batch_accuracy = utils.calculate_accuracy(batch_predictions=train_predictions, batch_targets=train_Y)
            train_epoch_accuracy += train_batch_accuracy/len(TRAIN_DATALOADER)
            train_epoch_loss += train_batch_loss/len(TRAIN_DATALOADER)

        print(f"Epoch {epoch_idx} :\nTraining Accuracy: {train_epoch_accuracy}\nTesting Loss: {train_epoch_loss}\n\n")

        test_epoch_accuracy = 0
        test_epoch_loss = 0
        MODEL.eval()
        for idx, data in enumerate(TEST_DATALOADER):

            test_X,test_Y = data['images'].to(DEVICE), data['labels'].to(DEVICE)

            test_predictions = MODEL(test_X)
            test_batch_loss = CRITERION(test_predictions, test_Y.reshape(-1))

            test_batch_accuracy = utils.calculate_accuracy(batch_predictions=test_predictions, batch_targets=test_Y)
            test_epoch_accuracy += test_batch_accuracy/len(TEST_DATALOADER)
            test_epoch_loss += test_batch_loss/len(TEST_DATALOADER)


        print(f"Epoch {epoch_idx} :\nTesting Accuracy: {test_epoch_accuracy}\nTesting Loss: {test_epoch_loss}\n\n")







if __name__ == '__main__':
    main()

