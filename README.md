# Vanilla Vision Transformer with PyTorch and Einops
## Introduction

The codes in this repository are free to be used in any way you like. The implementation here is based on [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929v2.pdf) research paper. The model is trained on CIFAR-10 dataset due to resource constraints. From [Understanding Why ViT Trains Badly on Small Datasets: An Intuitive Perspective](https://arxiv.org/pdf/2302.03751.pdf#:~:text=On%20the%20other%20hand%2C%20vision,is%20still%20lower%20than%20CNN's.), it is learned that the best accuracy a vanilla vision transformer can achieve in CIFAR-10 dataset is about 80%. In this implementation, the highest test accuracy achieved is about 70%. Perhaps with a more aggressive data augmentation and better hyperparameter selection, a better test accuracy can be achieved. 



## Architecture

![Vision Transformer Architecture](https://github.com/Ugenteraan/Vanilla-ViT/blob/main/readme_files/vit-arch.gif?raw=true "Vision Transformer Architecture")
Source: [VIT-PyTorch](https://github.com/lucidrains/vit-pytorch)

In this implementation, a transformer encoder network of depth 8 was used. Other configurations can be seen in [cfg.py](https://github.com/Ugenteraan/Vanilla-ViT/blob/main/cfg.py) file. 

As for the MLP Head, there are two different implementations. The main implementation in the main branch [mlp_head.py](https://github.com/Ugenteraan/Vanilla-ViT/blob/main/ViT/mlp_head.py) follows the architecture as depicted in the GIF above. In other words, only the CLS token is used for classification at the end. The second implementation resides in "full_mlp_head" branch [mlp_head.py](https://github.com/Ugenteraan/Vanilla-ViT/blob/full_mlp_head/ViT/mlp_head.py). In this implementation, all of the outputs from the last layer of transformer encoder is used for classification (including the CLS token). This was done by averaging the tensors in the 2nd dimension. 

## Dataset 

[DeepLake](https://www.deeplake.ai/) was used to load datasets at ease. In this experiment, CIFAR-10 dataset was used with image sizes of 32 x 32 x 3. This dataset was chosen due to the computing resource constraints. Using bigger size datasets (with higher image resolution) significantly lenghten the experiment process. Patience was also a resource constraint in this experiment. 

Some of the image augmentations used during training were: 
- Color Jittering
- Random Horizontal Flipping
- Random Affine Transformations

The configurations for each of the augmentations can be found in [load_dataset.py](https://github.com/Ugenteraan/Vanilla-ViT/blob/main/load_dataset.py).

## Training

In both mode of training, you'll first be required to create a ```cred.py``` file that consists of three variables.
- ACTIVELOOP_TOKEN - Retrieved from [DeepLake](https://www.deeplake.ai/).
- NEPTUNE_PROJECT - Retrieved from [Neptune.ai](https://neptune.ai/)
- NEPTUNE_API_TOKEN - Retrieved from [Neptune.ai](https://neptune.ai/)

All three variables above have to be retrieved from your account settings in their respective sites. Creating an account in those sites is an easy process and free options are available too.

Before running the trainings, make sure to change the values in [cfg.py](https://github.com/Ugenteraan/Vanilla-ViT/blob/main/cfg.py) file as per your dataset requirement and resource availability. Important paramters to double check is the image size, channels, and patch size.


### Single GPU Training

To perform the training using a single GPU, simply run 

```
python train.py
```

in the root folder. 

### Distributed Data Parallel Training

Due to the lack of patience as mentioned earlier, **DDP** was used in this experiment. Note that there are two extra files in the repository [train_multi_gpu.py](https://github.com/Ugenteraan/Vanilla-ViT/blob/main/train_multi_gpu.py) and [load_dataset_multi_gpu.py](https://github.com/Ugenteraan/Vanilla-ViT/blob/main/load_dataset_multi_gpu.py). Both these files have to be kept consistent with their counterparts except for the DDP logics in order to track the experiment smoothly. 

If you wish to train the model on more than 1 GPU (provided that you do have more than 1 GPU installed), change these parameters:
- WORLD_SIZE
- nprocs

in [train_multi_gpu.py](https://github.com/Ugenteraan/Vanilla-ViT/blob/main/train_multi_gpu.py) file.

To ensure that you are indeed training the model using multiple GPUs, check the utilization of your GPU cards with ```nvidia-smi``` command.

To start the training, simply run
```
python train_multi_gpu.py
```

### Experiment Tracking

To ease the tracking of the experiments (usually there'll be multiple experiments with different parameters), [Neptune.ai](https://neptune.ai/) is used. To know more about it, read their documentations. It's easy!

## Results

There are a total of two experiments conducted. 

We'll call the first experiement **CLS Token MLP Head**. This is directly from the implementation from the main branch where only the CLS token is used for classification.

The second experiment comes from the "full_mlp_head" branch where all of the output tensor from the final transformer encoder layer was used for classification. We'll name this **Full MLP Head**.

For both the experiments, the same parameters were used.

|  Parameters 	|  Value 	|   
|---	|---	|
| total train epoch  	|   1001	| 
| batch size   	|   128	| 
|  data shuffle 	|  True 	|
| image size | 32 x 32 x3 |
|patch size | 8 x 8 |
| learning rate | 1e-4|
| scheduler | StepLR |
| step size  | 200 |
| scheduler gamma | 0.5 |
| num of attention heads | 8 |
| transformer encoder depth | 8 |
| mlp head dropout rate | 0.1 |
|attention layer dropout rate | 0.1 |

### CLS Token MLP Head

Training Loss:

<img src="https://github.com/Ugenteraan/Vanilla-ViT/blob/main/readme_files/1st%20-%20train_loss-every-5-epoch.png?raw=true" width="550" height="250">

Training Accuracy:

<img src="https://github.com/Ugenteraan/Vanilla-ViT/blob/main/readme_files/1st%20-%20train_acc-every-5-epoch.png?raw=true" width="550" height="250">

Testing Loss:

<img src="https://github.com/Ugenteraan/Vanilla-ViT/blob/main/readme_files/1st%20-%20test_loss-every-5-epoch.png?raw=true" width="550" height="250">

Testing Accuracy:

<img src="https://github.com/Ugenteraan/Vanilla-ViT/blob/main/readme_files/1st%20-%20test_acc-every-5-epoch.png?raw=true" width="550" height="250">



### Full MLP Head

<img src="https://github.com/Ugenteraan/Vanilla-ViT/blob/main/readme_files/2nd%20-%20train_loss-every-5-epoch.png?raw=true" width="550" height="250">

Training Accuracy:

<img src="https://github.com/Ugenteraan/Vanilla-ViT/blob/main/readme_files/2nd%20-%20train_acc-every-5-epoch.png?raw=true" width="550" height="250">

Testing Loss:

<img src="https://github.com/Ugenteraan/Vanilla-ViT/blob/main/readme_files/2nd%20-%20test_loss-every-5-epoch.png?raw=true" width="550" height="250">

Testing Accuracy:

<img src="https://github.com/Ugenteraan/Vanilla-ViT/blob/main/readme_files/2nd%20-%20test_acc-every-5-epoch.png?raw=true" width="550" height="250">



## Conclusion

It is clear to see that the results from both the experiments are strikingly similar. The loss of the test set in both experiments starts to increase after a certain point. That probably indicates overfitting. Regardless, the highest accuracy for both the experiments were around 68%. 

Perhaps for small-scale expriment such as this, the implementation of the MLP head does not affect too much. Further experiments are needed to confirm the better implementation. Maybe the attention maps would be different when visualized. There's no point visualizing the attention maps in this experiment as it won't yield any insight due to the size of the dataset used as reported in [Understanding Why ViT Trains Badly on Small Datasets: An Intuitive Perspective](https://arxiv.org/pdf/2302.03751.pdf#:~:text=On%20the%20other%20hand%2C%20vision,is%20still%20lower%20than%20CNN's.).




## License

MIT

