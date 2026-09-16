"""Lightweight OpenCV augmentations (no extra dependency).

Geometric transforms are applied identically to the image and its masks.
"""
import random

import cv2
import numpy as np


def _affine(img, M, is_mask):
    h, w = img.shape[:2]
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def augment(img, masks=(), strong=True):
    masks = list(masks)
    if random.random() < 0.5:
        img = img[:, ::-1]
        masks = [m[:, ::-1] for m in masks]
    if random.random() < 0.3:
        img = img[::-1]
        masks = [m[::-1] for m in masks]
    if random.random() < 0.7:
        h, w = img.shape[:2]
        angle = random.uniform(-20, 20)
        scale = random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += (random.uniform(-0.05, 0.05) * w, random.uniform(-0.05, 0.05) * h)
        img = _affine(np.ascontiguousarray(img), M, False)
        masks = [_affine(np.ascontiguousarray(m), M, True) for m in masks]
    if strong:
        x = img.astype(np.float32)
        if random.random() < 0.8:  # brightness / contrast
            x = x * random.uniform(0.75, 1.25) + random.uniform(-25, 25)
        if random.random() < 0.3:  # gamma (simulates low-contrast captures)
            x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** random.uniform(0.7, 1.5)
        if random.random() < 0.2:
            x = cv2.GaussianBlur(x, (5, 5), 0)
        if random.random() < 0.2:
            x = x + np.random.normal(0, 6, x.shape)
        img = np.clip(x, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), [np.ascontiguousarray(m) for m in masks]
