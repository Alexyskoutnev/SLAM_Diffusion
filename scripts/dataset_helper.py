import os
from pathlib import Path

import numpy as np
import h5py
import cv2

IMG_SIZE = 64
# rawdata_dir = Path("./rawdata/")
rawdata_dir = "./rawdata/"
hdf5_dir = Path("data/hdf5/")


def store_video(rawdatadir, hdf5_dir):
    images = []
    for video_path in os.listdir(rawdatadir):
        video_obj = cv2.VideoCapture(os.path.join(rawdatadir, video_path))
        while(video_obj.isOpened()):
            ret, frame = video_obj.read()
            if ret == False:
                break
            # downsize_frame = cv2.flip(cv2.resize(frame, IMG_SIZE), 0)
            downsize_frame = cv2.flip(frame, 0)
            images.append(downsize_frame)
            
    num_images = len(images)
    file = h5py.File(hdf5_dir / f"{num_images}_hallway.h5", "w")
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    file.close()
    
if __name__ == "__main__":
    #helper function to convert mp4 video to h5 dataset
    store_video(rawdata_dir, hdf5_dir)