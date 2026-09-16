"""Post-processing of predicted masks to reduce false positives
(mentor feedback): keep the largest component, fill holes, force the cup
inside the disc and optionally smooth boundaries with an ellipse fit."""
import cv2
import numpy as np


def largest_component(mask):
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def fill_holes(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out


def ellipse_fit(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return mask
    out = np.zeros_like(mask, dtype=np.uint8)
    cv2.ellipse(out, cv2.fitEllipse(c), 1, thickness=cv2.FILLED)
    return out


def clean_masks(disc_prob, cup_prob, t_disc=0.5, t_cup=0.5, use_ellipse=True, min_area=30):
    """Probability maps (HxW float) -> cleaned binary disc and cup masks."""
    disc = (disc_prob >= t_disc).astype(np.uint8)
    cup = (cup_prob >= t_cup).astype(np.uint8)

    disc = fill_holes(largest_component(disc))
    if disc.sum() < min_area:
        return np.zeros_like(disc), np.zeros_like(cup)
    if use_ellipse:
        disc = ellipse_fit(disc)

    cup = cup & disc                      # cup must lie inside the disc
    cup = fill_holes(largest_component(cup))
    if use_ellipse and cup.sum() >= min_area:
        cup = ellipse_fit(cup) & disc
    return disc.astype(np.uint8), cup.astype(np.uint8)
