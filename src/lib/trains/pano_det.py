from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
import cv2
import math
import json

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import pano_decode_32_reg, pano_decode_1_class
from models.utils import _sigmoid
from utils.debugger import Debugger
from .base_trainer import BaseTrainer


class PanoLoss(torch.nn.Module):

    def __init__(self, opt):
        super(PanoLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                        RegLoss() if opt.reg_loss == 'sl1' else None
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss = 0
        wh_loss = 0
        off_loss = 0

        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm_center'] = _sigmoid(output['hm_center'])
            hm_loss += self.crit(output['hm_center'], batch['hm_center']) / opt.num_stacks

            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['tooth_wh'], batch['reg_mask'],
                    batch['ind_center'], batch['tooth_wh']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss}
        return loss, loss_stats


class PanoTrainer(BaseTrainer):

    def __init__(self, opt, model, optimizer=None):
        super(PanoTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss']
        loss = PanoLoss(opt)
        return loss_states, loss

    def pano_center_crop_and_resize(self, img):
        h, w, _ = img.shape
        new_w = int(h * 1.5)
        margin = (w - new_w) // 2
        img = img[:, margin: margin + new_w, :]
        img = cv2.resize(img, dsize=(self.opt.input_w, self.opt.input_h))
        return img, 512 * margin / h

    def draw_result_32_reg(self, batch, dets, save_json=False, save_img=False, draw_aabb=False):
        img_id = batch['img_id'][0]
        if self.opt.val:
            split = 'val'
            epoch = self.opt.val
        elif self.opt.test:
            split = 'test'
            epoch = self.opt.test

        vis_dir = os.path.join(self.opt.save_dir, 'vis_' + split + '_' + epoch)
        img_path = os.path.join(self.opt.data_dir, split, img_id)
        if os.path.exists(img_path + '.jpg'):
            img_path = img_path + '.jpg'
        elif os.path.exists(img_path + '.bmp'):
            img_path = img_path + '.bmp'
        elif os.path.exists(img_path + '.BMP'):
            img_path = img_path + '.BMP'
        else:
            return

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_H, orig_W, _ = img.shape
        img, margin = self.pano_center_crop_and_resize(img)
        H, W, _ = img.shape
        img = np.uint8(img * 0.5)
        output_list = []

        for i in range(32):
            tooth_num = (i // 8) * 10 + (i % 8) + 11

            x_center = float(dets[i][0]) * 4
            y_center = float(dets[i][1]) * 4
            tooth_w = float(dets[i][3]) * H * 1.5 / 2
            tooth_h = float(dets[i][4]) * H / 2
            x1 = int(x_center - tooth_w)
            x2 = int(x_center + tooth_w)
            y1 = int(y_center - tooth_h)
            y2 = int(y_center + tooth_h)

            if save_img:
                cv2.circle(img, (int(x_center), int(y_center)), 3, (127,0,255), -1)
                cv2.putText(img, str(tooth_num), (int(x_center - 15), int(y_center - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                if draw_aabb:
                    cv2.line(img, (x1, y1), (x1, y2), (0, 255, 0))
                    cv2.line(img, (x1, y1), (x2, y1), (0, 255, 0))
                    cv2.line(img, (x2, y2), (x2, y1), (0, 255, 0))
                    cv2.line(img, (x2, y2), (x1, y2), (0, 255, 0))

            if save_json:
                x_center = int((x_center + margin) * orig_W / (W + 2 * margin))
                y_center = int(y_center * orig_H / H)
                tooth_w = int(tooth_w * 2 * orig_W / W)
                tooth_h = int(tooth_h * 2 * orig_H / H)

                cur_dict = {}
                cur_dict['tooth_num'] = tooth_num
                cur_dict['center'] = (x_center, y_center)
                cur_dict['bbox_wh'] = (tooth_w, tooth_h)
                output_list.append(cur_dict)

        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        if save_img:
            cv2.imwrite(os.path.join(vis_dir, img_id + '_result.jpg'), img)

        if save_json:
            with open(os.path.join(vis_dir, img_id + '_result.json'), 'w') as f:
                json.dump(output_list, f)

    def draw_1_class(self, batch, dets, thresh=0, save_json=False, save_img=False, draw_aabb=False):
        img_id = batch['img_id'][0]
        if self.opt.val:
            split = 'val'
            epoch = self.opt.val
        elif self.opt.test:
            split = 'test'
            epoch = self.opt.test

        vis_dir = os.path.join(self.opt.save_dir, 'vis_' + split + '_' + epoch)
        img_path = os.path.join(self.opt.data_dir, split, img_id)
        if os.path.exists(img_path + '.jpg'):
            img_path = img_path + '.jpg'
        elif os.path.exists(img_path + '.bmp'):
            img_path = img_path + '.bmp'
        elif os.path.exists(img_path + '.BMP'):
            img_path = img_path + '.BMP'
        else:
            return

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_H, orig_W, _ = img.shape
        img, margin = self.pano_center_crop_and_resize(img)
        H, W, _ = img.shape
        output_list = []

        for i, d in enumerate(dets):
            score = float(d[2])
            if score < thresh:
                continue

            x_center = int(float(d[0]) * 4)
            y_center = int(float(d[1]) * 4)
            tooth_w = float(d[3]) * H * 1.5 / 2
            tooth_h = float(d[4]) * H / 2
            x1 = int(x_center - tooth_w)
            x2 = int(x_center + tooth_w)
            y1 = int(y_center - tooth_h)
            y2 = int(y_center + tooth_h)

            if save_img:
                cv2.circle(img, (x_center, y_center), 3, (255,0,0), -1)
                if draw_aabb:
                    cv2.line(img, (x1, y1), (x1, y2), (0, 255, 0))
                    cv2.line(img, (x1, y1), (x2, y1), (0, 255, 0))
                    cv2.line(img, (x2, y2), (x2, y1), (0, 255, 0))
                    cv2.line(img, (x2, y2), (x1, y2), (0, 255, 0))

            if save_json:
                x_center = int((x_center + margin) * orig_W / (W + 2 * margin))
                y_center = int(y_center * orig_H / H)
                tooth_w = int(tooth_w * 2 * orig_W / W)
                tooth_h = int(tooth_h * 2 * orig_H / H)

                cur_dict = {}
                cur_dict['center'] = (x_center, y_center)
                cur_dict['bbox_wh'] = (tooth_w, tooth_h)
                cur_dict['score'] = score
                output_list.append(cur_dict)

        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        if save_img:
            cv2.imwrite(os.path.join(vis_dir, img_id + '_result.jpg'), img)

        if save_json:
            with open(os.path.join(vis_dir, img_id + '_result.json'), 'w') as f:
                json.dump(output_list, f)

    def save_result(self, output, batch, results):
        hm_center = _sigmoid(output['hm_center'])
        tooth_wh = output['tooth_wh']

        if self.opt.num_classes == 1:
            dets= pano_decode_1_class(hm_center, 0, tooth_wh=tooth_wh, K=32)
            self.draw_1_class(batch, dets[0].detach().cpu().numpy(), save_img=True, draw_aabb=True, save_json=True)
        else:
            dets = pano_decode_32_reg(hm_center, tooth_wh=tooth_wh)
            self.draw_result_32_reg(batch, dets[0].detach().cpu().numpy(), save_img=True, draw_aabb=True, save_json=True)
