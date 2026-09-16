"""Image loading and preprocessing (CLAHE contrast enhancement, resizing)."""
import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_rgb(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a readable image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path, shape=None):
    """Binary mask (uint8 0/1). Any non-zero pixel counts as foreground."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if shape is not None and m.shape[:2] != tuple(shape[:2]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def apply_clahe(rgb, clip=2.0, grid=8):
    """CLAHE on the L channel: sharpens disc/cup boundaries in low-contrast images."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def resize(img, size, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, (size, size), interpolation=interp)


def normalize(rgb):
    """uint8 HWC RGB -> float32 CHW, ImageNet-normalised."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def prepare_seg_input(rgb, size):
    return normalize(resize(apply_clahe(rgb), size))


def prepare_cls_input(rgb, size):
    # resize first, then CLAHE (fast; identical to what the classifier sees in training)
    return normalize(apply_clahe(resize(rgb, size)))
