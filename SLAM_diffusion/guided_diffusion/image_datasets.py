import math
import random
import os

from PIL import Image

import blobfile as bf
from mpi4py import MPI
import numpy as np
import h5py
import cv2 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision import transforms

IMG_SIZE = 64
hdf5_dir = Path("./data/hdf5/hdf5_64")
hdf5_dir.mkdir(parents=True, exist_ok=True)

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    # breakpoint()
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    # dataset = ImageDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    #     random_crop=random_crop,
    #     random_flip=random_flip,
    # )
    dataset = HDF5Dataset(data_dir, transforms=True)
    # breakpoint()
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class HDF5Dataset(Dataset):
    
    data_transform = transforms.Compose([
                       transforms.ToPILImage(), \
                       transforms.Resize((IMG_SIZE, IMG_SIZE)), \
                       transforms.RandomHorizontalFlip(), \
                       transforms.ToTensor(), ])
                    #    transforms.Lambda(lambda t: (t * 2) - 1)])
    
    def __init__(self, file_path, transforms=None):
        super().__init__()
        self.data = np.array([])
        self.size = 0
        
        p = Path(file_path)
        files = p.glob('*.h5')
        for file in files:
            h5py_ptr = h5py.File(str(file.resolve()), "r+")
            images = np.array(h5py_ptr["/images"]).astype("uint8")
            self.size += len(images)
            if len(self.data) == 0:
                self.data = images
            else:
                self.data = np.concatenate((self.data, images), axis=0)
        
        if transforms:
            self.transforms = HDF5Dataset.data_transform
        else:
            self.transforms = None  
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        if self.transforms:
            X = self.transforms(X)
            X = normalize(X)
        else:
            X = torch.from_numpy(X)
        return X, {}

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        breakpoint()
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def show_img(frame):
    plt.imshow(frame)
    plt.show()

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

def store_video(video_dir):
    images = []
    for video_path in os.listdir(video_dir):
        video_obj = cv2.VideoCapture(os.path.join(video_dir, video_path))
        while(video_obj.isOpened()):
            ret, frame = video_obj.read()
            if ret == False:
                break
            downsize_frame = cv2.flip(cv2.resize(frame, IMG_SIZE), 0)
            images.append(downsize_frame)
            
    num_images = len(images)
    file = h5py.File(hdf5_dir / f"{num_images}_hallway.h5", "w")
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    file.close()

def normalize(frame):
    return frame * 2 - 1