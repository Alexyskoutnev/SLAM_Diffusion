import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

IMG_SIZE = 128
device = "cpu"

SAMPLE_PATH = "./samples/"

@torch.no_grad()
def save_plot_image(frames, itr):
    img_size = IMG_SIZE
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10 
    for i, frame in enumerate(frames):
        t = torch.full((1, ), i, device=device, dtype=torch.long)
        img = frame
        plt.subplot(1, num_images, i + 1)
        show_tensor_image(img.detach().cpu())
    plt.savefig(os.path.join(SAMPLE_PATH, datetime.datetime.now().strftime("_hallway-%Y-%m-%d-%H-%M-%S-%f") + ".png"), bbox_inches='tight')
    plt.close()
    

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2), \
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), \
        transforms.Lambda(lambda t: t * 255), \
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  \
        transforms.ToPILImage()
        ])
    
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))
    