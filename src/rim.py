"""Neuroretinal rim analysis - slide 04, step 4 ("nerve fibre and rim features").

The rim is the tissue between the cup edge and the disc edge: it is the axons
themselves, and it thins before the cup visibly enlarges. Everything here is
computed from the disc and cup masks Component 1 already produces, so it needs no
new data and no new model.

What is measured, by casting rays out from the disc centre:

  * rim width in each of four sectors (superior, inferior, nasal, temporal)
  * the ISNT rule - in a healthy nerve the rim is widest inferiorly, then
    superiorly, then nasally, then temporally. Loss of that order is an early
    glaucoma sign.
  * the thinnest rim anywhere on the disc, and where it is
  * rim area and rim-to-disc area ratio

A caveat that is stated rather than hidden: nasal and temporal depend on which
eye it is and on the image orientation. The convention here is that for a right
eye (OD) the nasal side of the disc is toward the left of the image; pass
`nasal_left=False` to flip it. The inferior-versus-superior comparison, which is
the clinically important half of the ISNT rule, does not depend on that choice.

Pure numpy - no torch, no scipy.
"""
import numpy as np

SECTORS = ("superior", "inferior", "nasal", "temporal")
N_RAYS = 360


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _radial_profile(mask, cx, cy, n_rays=N_RAYS, max_r=None):
    """Outer radius of `mask` along each of n_rays directions (pixels)."""
    h, w = mask.shape
    if max_r is None:
        max_r = int(np.hypot(max(cx, w - cx), max(cy, h - cy))) + 2
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.arange(1, max_r)
    # sample every ray at every radius in one shot
    xs = np.clip(np.round(cx + np.cos(angles)[:, None] * radii[None, :]).astype(int), 0, w - 1)
    ys = np.clip(np.round(cy - np.sin(angles)[:, None] * radii[None, :]).astype(int), 0, h - 1)
    hit = mask[ys, xs] > 0
    # outer edge = largest radius still inside the mask along that ray
    out = np.zeros(n_rays, float)
    any_hit = hit.any(axis=1)
    last = np.where(hit, radii[None, :], 0).max(axis=1)
    out[any_hit] = last[any_hit]
    return angles, out


def _sector_masks(angles, nasal_left):
    """Angle is measured anticlockwise from the +x axis, with +y upward."""
    deg = np.degrees(angles) % 360
    sup = (deg >= 45) & (deg < 135)
    inf = (deg >= 225) & (deg < 315)
    left = (deg >= 135) & (deg < 225)
    right = (deg < 45) | (deg >= 315)
    nasal, temporal = (left, right) if nasal_left else (right, left)
    return {"superior": sup, "inferior": inf, "nasal": nasal, "temporal": temporal}


def rim_analysis(disc, cup, eye="OD", nasal_left=None, n_rays=N_RAYS):
    """Rim geometry from the disc and cup masks. Returns None if there is no disc."""
    disc = np.asarray(disc) > 0
    cup = np.asarray(cup) > 0
    c = _centroid(disc)
    if c is None or disc.sum() == 0:
        return None
    cx, cy = c
    if nasal_left is None:
        nasal_left = str(eye).upper() != "OS"

    angles, r_disc = _radial_profile(disc, cx, cy, n_rays)
    if cup.sum() > 0:
        _, r_cup = _radial_profile(cup, cx, cy, n_rays)
    else:
        r_cup = np.zeros_like(r_disc)

    valid = r_disc > 0
    rim = np.clip(r_disc - r_cup, 0, None)
    rim_ratio = np.where(valid, rim / np.maximum(r_disc, 1e-6), np.nan)

    secs = _sector_masks(angles, nasal_left)
    sector_px, sector_ratio = {}, {}
    for name, m in secs.items():
        sel = m & valid
        sector_px[name] = float(np.nanmean(rim[sel])) if sel.any() else None
        sector_ratio[name] = float(np.nanmean(rim_ratio[sel])) if sel.any() else None

    order = [sector_px[s] for s in ("inferior", "superior", "nasal", "temporal")]
    isnt_ok = all(a is not None and b is not None and a >= b for a, b in zip(order, order[1:]))
    inf_sup_ok = (sector_px["inferior"] is not None and sector_px["superior"] is not None
                  and sector_px["inferior"] >= sector_px["superior"])
    breaches = []
    names = ("inferior", "superior", "nasal", "temporal")
    for i in range(3):
        a, b = order[i], order[i + 1]
        if a is not None and b is not None and a < b:
            breaches.append(f"{names[i]} rim thinner than {names[i + 1]}")

    idx = int(np.nanargmin(np.where(valid, rim_ratio, np.nan))) if valid.any() else 0
    min_angle = float(np.degrees(angles[idx]) % 360)
    min_sector = next((s for s, m in secs.items() if m[idx]), None)

    disc_area, cup_area = int(disc.sum()), int(cup.sum())
    rim_area = max(0, disc_area - cup_area)
    return {
        "sector_rim_px": {k: (round(v, 2) if v is not None else None) for k, v in sector_px.items()},
        "sector_rim_ratio": {k: (round(v, 4) if v is not None else None) for k, v in sector_ratio.items()},
        "isnt_respected": bool(isnt_ok),
        "inferior_ge_superior": bool(inf_sup_ok),
        "isnt_breaches": breaches,
        "min_rim_to_disc_ratio": float(round(np.nanmin(rim_ratio[valid]), 4)) if valid.any() else None,
        "min_rim_sector": min_sector,
        "min_rim_angle_deg": round(min_angle, 1),
        "rim_area_px": rim_area,
        "rim_to_disc_area_ratio": round(rim_area / disc_area, 4) if disc_area else None,
        "eye": str(eye).upper(),
        "nasal_side": "image-left" if nasal_left else "image-right",
        "note": ("Nasal/temporal assume standard fundus orientation for this eye; "
                 "the inferior-vs-superior comparison does not depend on that."),
    }


def rim_findings(r, cdr=None):
    """Plain-language findings for the report."""
    if not r:
        return []
    out = []
    s = r["sector_rim_ratio"]
    if r["isnt_respected"]:
        out.append("Neuroretinal rim follows the ISNT rule (inferior thickest, temporal thinnest).")
    else:
        out.append("ISNT rule not respected: " + "; ".join(r["isnt_breaches"]) +
                   ". Loss of the ISNT order is an early glaucomatous sign.")
    if r["min_rim_to_disc_ratio"] is not None:
        where = r["min_rim_sector"] or "an unclassified sector"
        out.append(f"Thinnest rim is {r['min_rim_to_disc_ratio']:.2f} of the disc radius, "
                   f"in the {where} sector.")
        if r["min_rim_to_disc_ratio"] < 0.10:
            out.append("Rim approaches the disc margin at its thinnest point, "
                       "consistent with focal rim loss or a notch.")
    if s.get("inferior") is not None and s.get("superior") is not None and not r["inferior_ge_superior"]:
        out.append(f"Inferior rim ({s['inferior']:.2f}) is thinner than superior ({s['superior']:.2f}); "
                   "the inferior pole is where glaucomatous loss most often begins.")
    return out


__all__ = ["rim_analysis", "rim_findings", "SECTORS"]
