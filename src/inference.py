"""End-to-end VisionTrack AI Phase-1 pipeline:
fundus image -> U-Net disc/cup -> post-processing -> CDR -> EfficientNet-B0 risk.

The dict returned by `analyze` is the documented interface (schema v1.0)
that Component 2 (visual-field integration) will consume."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from .cdr import compute_cdr
from .models import load_classifier, load_unet
from .optics import physical_sizes
from .postprocess import clean_masks
from .rim import rim_analysis, rim_findings
from .preprocessing import prepare_cls_input, prepare_seg_input
from .visualize import crop_roi, make_overlay  # noqa: F401  (re-exported)

SCHEMA_VERSION = "2.0"


def load_thresholds():
    p = config.CHECKPOINT_DIR / "thresholds.json"
    t = {"disc": config.SEG_THRESH_DISC, "cup": config.SEG_THRESH_CUP, "cls": config.CLS_THRESH}
    if p.exists():
        t.update(json.loads(p.read_text()))
    return t


def risk_level(prob):
    for lo, hi, name in config.RISK_BANDS:
        if lo <= prob < hi:
            return name
    return "High"


class VisionTrackPipeline:
    def __init__(self, seg_ckpt=config.SEG_CKPT, cls_ckpt=config.CLS_CKPT, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not Path(seg_ckpt).exists():
            raise FileNotFoundError(2, "Segmentation checkpoint not found", str(seg_ckpt))
        self.seg, seg_cfg = load_unet(seg_ckpt, self.device)
        self.seg_size = seg_cfg.get("img_size", config.SEG_IMG_SIZE)
        self.cls = None
        self.cls_size = config.CLS_IMG_SIZE
        if cls_ckpt and Path(cls_ckpt).exists():
            self.cls, cls_cfg = load_classifier(cls_ckpt, self.device)
            self.cls_size = cls_cfg.get("img_size", config.CLS_IMG_SIZE)
        self.thr = load_thresholds()

    @torch.no_grad()
    def segment_probs(self, rgb, tta=True):
        """Returns (disc_prob, cup_prob) at the original image resolution."""
        x = torch.from_numpy(prepare_seg_input(rgb, self.seg_size))[None].to(self.device)
        prob = torch.sigmoid(self.seg(x))
        if tta:
            prob_f = torch.sigmoid(self.seg(torch.flip(x, dims=[3])))
            prob = (prob + torch.flip(prob_f, dims=[3])) / 2
        prob = F.interpolate(prob, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        prob = prob[0].cpu().numpy()
        return prob[0], prob[1]

    def segment(self, rgb, tta=True, use_ellipse=True):
        dp, cp = self.segment_probs(rgb, tta)
        return clean_masks(dp, cp, self.thr["disc"], self.thr["cup"], use_ellipse=use_ellipse)

    @torch.no_grad()
    def glaucoma_probability(self, rgb, tta=True):
        if self.cls is None:
            return None
        x = torch.from_numpy(prepare_cls_input(rgb, self.cls_size))[None].to(self.device)
        p = torch.sigmoid(self.cls(x)).item()
        if tta:
            p = (p + torch.sigmoid(self.cls(torch.flip(x, dims=[3]))).item()) / 2
        return float(p)

    def analyze(self, rgb, image_id="image", eye="OD", axial_length_mm=None):
        disc, cup = self.segment(rgb)
        cdr = compute_cdr(disc, cup)
        prob = self.glaucoma_probability(rgb)
        rim = rim_analysis(disc, cup, eye)
        sizes = physical_sizes(cdr, rgb.shape[1], axial_length_mm)

        flags = []
        if disc.sum() == 0:
            flags.append("Optic disc not detected - check image quality / field of view")
        elif cup.sum() == 0:
            flags.append("Optic cup not detected - CDR may be unreliable")
        vcdr = cdr["vertical_cdr"]
        if vcdr is not None and vcdr >= config.CDR_SUSPICIOUS:
            flags.append(f"Vertical CDR >= {config.CDR_SUSPICIOUS} (suspicious cupping)")
        if rim and not rim["isnt_respected"]:
            flags.append("Neuroretinal rim does not follow the ISNT rule")
        if rim and rim["min_rim_to_disc_ratio"] is not None and rim["min_rim_to_disc_ratio"] < 0.10:
            flags.append(f"Focal rim thinning in the {rim['min_rim_sector']} sector")
        if sizes and not sizes["plausible"]:
            flags.append("Physical size estimate outside the usual range - check the camera assumptions")

        classification = None
        if prob is not None:
            classification = {
                "glaucoma_probability": round(prob, 4),
                "prediction": "Glaucoma" if prob >= self.thr["cls"] else "Non-Glaucoma",
                "threshold": round(self.thr["cls"], 4),
                "risk_level": risk_level(prob),
            }
        ys, xs = np.nonzero(disc)
        center = [int(xs.mean()), int(ys.mean())] if len(xs) else None
        result = {
            "schema_version": SCHEMA_VERSION,
            "image_id": image_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "component1": {"disc_center_xy": center, "cdr": cdr, "rim": rim,
                           "physical": sizes,
                           "rim_findings": rim_findings(rim, cdr.get("vertical_cdr"))},
            "classification": classification,
            "flags": flags,
            "disclaimer": "Research prototype - not for clinical diagnosis.",
        }
        return result, disc, cup
