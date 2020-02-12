from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
import json
import cv2
import os
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import math


class PanoDataset(data.Dataset):

    def is_true(self, s):
        if 'T' in s or 't' in s:
            return True

        return False

    def pano_center_crop_and_resize(self, img):
        h, w = img.shape
        new_w = int(h * 1.5)
        margin = (w - new_w) // 2

        eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
        img = eq.apply(img)
        img = img[:, margin: margin + new_w]
        img = cv2.resize(img, dsize=(self.default_resolution[1], self.default_resolution[0]))
        img = (np.float64(img) - np.mean(img)) / np.std(img)

        return img

    def change_coords(self, x, y, H, W):
        new_w = int(H * 1.5)
        margin = (W - new_w) // 2
        x -= margin
        x /= new_w
        y /= H
        return (x, y)

    def process_anno(self, anno_file_name, H, W):
        w = W // 2
        h = H // 2
        annos = []
        df = pd.read_csv(os.path.join(self.data_dir, anno_file_name), header=None)

        for idx, row in df.iterrows():
            if self.num_classes == 1 and not self.is_true(row[1]):
                continue

            tooth_num = int(row[0])
            tooth_class = (tooth_num // 10) * 8 + tooth_num % 10 - 9

            x_max = y_max = -math.inf
            x_min = y_min = math.inf
            j = 3
            while j < 19:
                x = int(row[j]) + w
                y = int(row[j+1]) + h
                j += 2
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                x_min = min(x_min, x)
                y_min = min(y_min, y)

            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            x_alveolar = w + int(row[27])
            y_alveolar = h + int(row[28])
            x_crown = w + (int(row[3]) + int(row[17])) // 2
            y_crown = h + (int(row[4]) + int(row[18])) // 2
            x_root = w + (int(row[9]) + int(row[11])) // 2
            y_root = h + (int(row[10]) + int(row[12])) // 2

            tooth_width = (x_max - x_min) / (H * 1.5)
            tooth_height = (y_max - y_min) / H

            x_center, y_center = self.change_coords(x_center, y_center, H, W)
            x_alveolar, y_alveolar = self.change_coords(x_alveolar, y_alveolar, H, W)
            x_crown, y_crown = self.change_coords(x_crown, y_crown, H, W)
            x_root, y_root = self.change_coords(x_root, y_root, H, W)

            annos.append({
                'tooth_class': tooth_class,
                'tooth_size': (tooth_width, tooth_height),
                'extreme_points': [
                                   [x_center, y_center],
                                   [x_alveolar, y_alveolar],
                                   [x_crown, y_crown],
                                   [x_root, y_root]
                                  ]
            })

        return annos

    def __getitem__(self, index):
        img_file_name = self.img_file_names[index]
        img_path = os.path.join(self.data_dir, img_file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape

        inp = self.pano_center_crop_and_resize(img)
        inp = np.expand_dims(inp, 0)
        output_h = self.opt.output_h
        output_w = self.opt.output_w

        hm_center = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        ind_center = np.zeros((self.max_objs), dtype=np.int64)
        tooth_wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        if not self.opt.test:
            anno_file_name = img_file_name[:-3] + 'txt'
            annos = self.process_anno(anno_file_name, H, W)
            num_objs = min(len(annos), self.max_objs)
            draw_gaussian = draw_umich_gaussian

            for k in range(num_objs):
                anno = annos[k]
                cls_id = anno['tooth_class']
                pts = np.array(anno['extreme_points'], dtype=np.float32) * [output_w, output_h]
                pt_int = pts.astype(np.int32)
                tooth_w, tooth_h = anno['tooth_size']

                tooth_wh[k] = tooth_w, tooth_h
                radius = gaussian_radius((math.ceil(tooth_h * output_h), math.ceil(tooth_w * output_w)))
                radius = max(0, int(radius))

                if self.num_classes == 1:
                    draw_gaussian(hm_center[0], pt_int[0], radius)
                else:
                    draw_gaussian(hm_center[cls_id], pt_int[0], radius)

                ind_center[k] = pt_int[0, 1] * output_w + pt_int[0, 0]
                reg_mask[k] = 1

        ret = {
            'input': inp,
            'img_id': img_file_name[:-4],
            'original_wh': (W, H),
            'hm_center': hm_center,
            'reg_mask': reg_mask,
            'tooth_wh': tooth_wh,
            'ind_center': ind_center
        }

        return ret
