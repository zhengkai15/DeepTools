from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

class mydataset(Dataset):
    def __init__(self, flag='train', config=None):
        self.config = config
        self.flag = flag
        self.init_times = self.generate_init_times()
        
    def get_fts(self, init_time):
        
        return torch.rand(2,2)

    def get_gts(self, init_time):
        
        return torch.rand(2,2)

    def generate_init_times(self):
        init_times = pd.date_range("2024-01-01 00:00:00", "2024-02-01 00:00:00")
        return init_times

    def __getitem__(self, index):
        init_time = self.init_times[index]
        ft_item = self.get_fts(init_time)
        gt_item = self.get_gts(init_time)
        return ft_item, gt_item, init_time.strftime("%Y-%m-%d-%H")



    def __len__(self):
        return len(self.init_times)