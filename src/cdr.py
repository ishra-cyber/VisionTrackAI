"""Cup-to-Disc Ratio estimation (vertical, horizontal, area)."""
import numpy as np


def _extent(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0, 0
    return int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)


def compute_cdr(disc, cup):
    """Returns dict with vertical_cdr, horizontal_cdr, area_cdr (None if no disc)."""
    dh, dw = _extent(disc)
    ch, cw = _extent(cup)
    da, ca = int(disc.sum()), int(cup.sum())
    if dh == 0 or dw == 0 or da == 0:
        return {"vertical_cdr": None, "horizontal_cdr": None, "area_cdr": None,
                "disc_height": 0, "disc_width": 0, "cup_height": ch, "cup_width": cw,
                "disc_area": da, "cup_area": ca}
    return {
        "vertical_cdr": round(ch / dh, 4),
        "horizontal_cdr": round(cw / dw, 4),
        "area_cdr": round(ca / da, 4),
        "disc_height": dh, "disc_width": dw, "cup_height": ch, "cup_width": cw,
        "disc_area": da, "cup_area": ca,
    }
