from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import os
import torch.utils.data as data


class PANO(data.Dataset):
    default_resolution = (512, 768)
    num_classes = 1 # 1 or 32

    def __init__(self, opt, split):
        super(PANO, self).__init__()
        self.split = split # split = test or train
        self.opt = opt
        self.data_dir = os.path.join(opt.data_dir, split)
        self.img_file_names = []

        for f in os.listdir(self.data_dir):
            if f[-3:] != 'txt' and 'thum' not in f:
                self.img_file_names.append(f)

        self.num_samples = len(self.img_file_names)
        self.max_objs = 32

    def __len__(self):
        return self.num_samples
