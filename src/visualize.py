"""Overlay and ROI helpers (no torch dependency)."""
import cv2
import numpy as np


def make_overlay(rgb, disc, cup, alpha=0.45):
    """Green disc, red cup, with outlines."""
    over = rgb.copy()
    color = np.zeros_like(rgb)
    color[disc.astype(bool)] = (40, 200, 90)
    color[cup.astype(bool)] = (235, 60, 70)
    m = (disc | cup).astype(bool)
    over[m] = (rgb[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)
    th = max(1, rgb.shape[0] // 300)
    for mask, col in ((disc, (40, 200, 90)), (cup, (235, 60, 70))):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cs, -1, col, th)
    return over


def crop_roi(img, disc, pad=0.6):
    """Crop around the optic disc for a zoomed view."""
    ys, xs = np.nonzero(disc)
    if len(xs) == 0:
        return img
    cx, cy = xs.mean(), ys.mean()
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) * (0.5 + pad)
    h, w = img.shape[:2]
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + r))
    return img[y0:y1, x0:x1]
