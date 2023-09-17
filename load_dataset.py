import deeplake
import torch

places205_dataset = deeplake.load("hub://activeloop/places205")

print(places205_dataset)

dataloader = places205_dataset.pytorch(batch_size=3, num_workers=2, shuffle=False)
print(dataloader)
