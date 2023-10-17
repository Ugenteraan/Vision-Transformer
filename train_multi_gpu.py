'''Training module for the Vanilla Vision Transformer.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from pathlib import Path
import deeplake
import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
import torch.distributed as dist
import torch.multiprocessing as mp


from ViT.ViT import VisionTransformer
from load_dataset_multi_gpu import LoadDeeplakeDataset
import cred
import cfg
import utils

mp.set_sharing_strategy('file_system') #to avoid shared memory handles limit when there are too many batches at DataLoader.



def main(gpu):

    RANK = gpu
    WORLD_SIZE = 2

    dist.init_process_group(backend='nccl', init_method='env://', world_size=WORLD_SIZE, rank=RANK)

    DEVICE = gpu
    torch.cuda.set_device(gpu)

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
                            device=DEVICE).cuda(DEVICE)

    CRITERION = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=cfg.LEARNING_RATE)
    SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER, step_size=cfg.SCHEDULER_STEP_SIZE, gamma=cfg.SCHEDULER_GAMMA)



    summary(MODEL, (cfg.IMAGE_CHANNEL, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))

    MODEL = nn.parallel.DistributedDataParallel(MODEL, device_ids=[DEVICE])



    TRAIN_DATALOADER = LoadDeeplakeDataset(token=cred.ACTIVELOOP_TOKEN, deeplake_ds_name="hub://activeloop/cifar10-train", batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE, world_size=WORLD_SIZE, rank=RANK, num_workers=cfg.NUM_WORKERS, pin_memory=True)()
    TEST_DATALOADER = LoadDeeplakeDataset(token=cred.ACTIVELOOP_TOKEN, deeplake_ds_name="hub://activeloop/cifar10-test", batch_size=cfg.BATCH_SIZE, shuffle=False, world_size=WORLD_SIZE, rank=RANK, num_workers=cfg.NUM_WORKERS, pin_memory=True)()

    best_accuracy = 0
    #create folders if doesn't exist.
    Path(f'{cfg.MODEL_SAVE_FOLDER}').mkdir(parents=True, exist_ok=True)
    Path(f'{cfg.GRAPH_SAVE_FOLDER}').mkdir(parents=True, exist_ok=True)

    total_train_epoch_accuracy = []
    total_train_epoch_loss = []
    total_test_epoch_accuracy = []
    total_test_epoch_loss = []

    plotting_epoch_step = 5 #plot at every n epoch.

    for epoch_idx in tqdm(range(cfg.TRAIN_EPOCH)):

        train_epoch_accuracy = 0
        train_epoch_loss = 0
        MODEL.train() #set the model to training mode.

        total_train_data=0 #to verify the distributed training.
        train_idx = 0
        print(f"Training has started for epoch {epoch_idx} with learning rate of {SCHEDULER.get_last_lr()}")
        for train_idx, data in enumerate(TRAIN_DATALOADER):

            train_X, train_Y = data['images'].cuda(non_blocking=True), data['labels'].cuda(non_blocking=True)
            train_batch_size = train_X.detach().cpu().size(0)

            OPTIMIZER.zero_grad() #clear the optimizer.

            train_predictions = MODEL(train_X)
            train_batch_loss = CRITERION(train_predictions, train_Y.reshape(-1))
            train_batch_loss.backward()
            OPTIMIZER.step()



            train_batch_accuracy = utils.calculate_accuracy(batch_predictions=train_predictions.detach(), batch_targets=train_Y.detach())
            train_epoch_accuracy += train_batch_accuracy
            train_epoch_loss += train_batch_loss.item()
            total_train_data += train_batch_size
        SCHEDULER.step()

        train_epoch_accuracy /= (train_idx+1)
        train_epoch_loss /= (train_idx+1)



        test_epoch_accuracy = 0
        test_epoch_loss = 0


        print(f"Epoch {epoch_idx} :\nTraining Accuracy: {train_epoch_accuracy}\nTraining Loss: {train_epoch_loss}\nTrained on: {total_train_data}\n ")
        #we don't want to perform testing at every epoch
        if not epoch_idx % plotting_epoch_step == 0:
            continue

        total_train_epoch_accuracy.append(train_epoch_accuracy)
        total_train_epoch_loss.append(train_epoch_loss)

        total_test_data = 0 #to verify the distributed training.
        test_idx = 0
        MODEL.eval() #change to eval mode for testing.
        with torch.no_grad():
            for test_idx, data in enumerate(TEST_DATALOADER):

                test_X,test_Y = data['images'].cuda(non_blocking=True), data['labels'].cuda(non_blocking=True)
                test_batch_size = test_X.detach().cpu().size(0)

                test_predictions = MODEL(test_X)
                test_batch_loss = CRITERION(test_predictions, test_Y.reshape(-1))

                test_batch_accuracy = utils.calculate_accuracy(batch_predictions=test_predictions.detach(), batch_targets=test_Y.detach())
                test_epoch_accuracy += test_batch_accuracy
                test_epoch_loss += test_batch_loss.item()
                total_test_data += test_batch_size

        test_epoch_accuracy /= (test_idx+1)
        test_epoch_loss /= (test_idx+1)


        total_test_epoch_accuracy.append(test_epoch_accuracy)
        total_test_epoch_loss.append(test_epoch_loss)

        print(f"Epoch {epoch_idx} :\nTesting Accuracy: {test_epoch_accuracy}\nTesting Loss: {test_epoch_loss} Tested on: {total_test_data}\n\n")

        #plot a graph of accuracy and loss for train vs test.
        utils.plot_loss_acc(path=cfg.GRAPH_SAVE_FOLDER,
                            num_epoch=epoch_idx,
                            train_accuracies=total_train_epoch_accuracy,
                            train_losses=total_train_epoch_loss,
                            test_accuracies=total_test_epoch_accuracy,
                            test_losses=total_test_epoch_loss,
                            rank=RANK,
                            epoch_step=plotting_epoch_step)

        #save the model with the best test accuracy.
        if test_epoch_accuracy > best_accuracy:
            torch.save(MODEL, f"{cfg.MODEL_SAVE_FOLDER}model.pth")
            best_accuracy = test_epoch_accuracy



os.environ['MASTER_ADDR'] = '10.100.146.181'
os.environ['MASTER_PORT'] = '8888'

if __name__ == '__main__':
    mp.spawn(main, nprocs=2)

