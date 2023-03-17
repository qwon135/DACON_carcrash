import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader
from einops import rearrange
import random

import warnings
warnings.filterwarnings(action='ignore') 


class VideoDataset(Dataset):
    def __init__(self, df_for_dataset, transform=None, mode = 'train'):
        self.sample_id = df_for_dataset[:,0]
        self.video_path = df_for_dataset[:,1]
        self.label = df_for_dataset[:,2]
        self.label_split = np.array(df_for_dataset[:,3].tolist())
        self.transform = transform
        self.mode= mode

    def __len__(self):
        return len(self.sample_id)

    def __getitem__(self, idx):
        sample_id = self.sample_id[idx]
        video_path = self.video_path[idx]
        video_name = video_path.split('/')[-1]

        label = self.label[idx]

        label_split = self.label_split[idx]

        vr = VideoReader(video_path)
        video = vr.get_batch(range(50)).asnumpy()
        video = torch.from_numpy(video)
        video = rearrange(video, 't h w c -> c t h w')                

        if self.transform:
            video = self.transform(video)
        video = rearrange(video, 'c t h w -> t c h w')

        sample = {
            'sample_id':sample_id,
            'video_name' : video_name,
            'video':video,
            'label':label,
            'label_split':label_split
        }
        
        return sample        