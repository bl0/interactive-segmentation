import argparse
from os import path as osp

import cv2
import numpy as np


def linear_combine(x, y, alpha):
    return x * alpha + y * (1 - alpha)


def bool_str(s):
    s = s.lower()
    return s == 'yes' or s == 'y' or s == 'ok' or s == 'true' or s == '1'


def load_image(img_path, max_height, max_width):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    scale = min(max_height / h, max_width / w)
    if scale < 1:
        img = cv2.resize(img, dsize=(0, 0), fx=scale,
                         fy=scale, interpolation=cv2.INTER_CUBIC)

    return img


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass


COLOR_BG = (0, 0, 0)
COLOR_FG = (0, 255, 0)


def load_mask(mask_path, img_shape, use_prev_mask=True):
    m = np.zeros(img_shape[:2], 'uint8')
    m[:] = cv2.GC_PR_BGD
    if use_prev_mask and osp.exists(mask_path):
        mask_restored = color2mask(cv2.imread(mask_path))
        if mask_restored.shape[:2] == img_shape[:2]:
            m = mask_restored

    return m


def mask2color(m):
    r, c = m.shape[:2]
    color = np.zeros((r, c, 3), np.uint8)
    color[np.where((m == 0) | (m == 2))] = COLOR_BG
    color[np.where((m == 1) | (m == 3))] = COLOR_FG
    return color


def color2mask(color):
    m = np.zeros(color.shape[:2], np.uint8)
    # in case image color changed
    m[np.abs((color - COLOR_FG).sum(axis=2)) < 150] = 1
    return m
