"""Populate the app with patients that have several visits, so the Progression
page has something real to analyse during a demonstration.

IMPORTANT, and stated on screen and in the run sheet: nobody in PAPILA or SMDG was
imaged twice, so a genuine follow-up series does not exist in this data. Each demo
patient here is assembled from DIFFERENT real eyes, ordered by cup-to-disc ratio,
and given plausible visit dates. Every number on the screen is a real model output
on a real fundus image; the sequence is what is constructed. Patient IDs are
prefixed DEMO- and every stored record carries a "demo_series" note.

    python scripts/seed_demo_patients.py                 # 3 patients x 5 visits
    python scripts/seed_demo_patients.py --reset         # clear DEMO- patients first
Requires data/cohort_master.csv (scripts/build_unified_cohort.py).
"""
import argparse
import json
import sqlite3
import uuid
from datetime import date, timedelta
from pathlib import Path

import _path  # noqa: F401
import cv2
import numpy as np
import pandas as pd

import config  # noqa: E402
from src.inference import VisionTrackPipeline  # noqa: E402
from src.preprocessing import read_rgb  # noqa: E402
from src.staging import combined_stage  # noqa: E402
from src.visualize import crop_roi, make_overlay  # noqa: E402

APP_DIR = config.ROOT / "app"
RESULTS = APP_DIR / "static" / "results"
DB_PATH = APP_DIR / "visiontrack.db"
NOTE = ("Demonstration series: each visit is a different real eye, ordered by cup-to-disc ratio. "
        "Model outputs are genuine; the follow-up sequence is constructed because the source datasets "
        "are cross-sectional.")

# Each visit targets a cup-to-disc ratio AND a visual-field Mean Defect, because
# picking eyes on CDR alone gave demo patients whose field jumped around at random
# - the eyes were ordered by structure, so their function was in no order at all.
PROFILES = [
    {"id": "DEMO-PROG-01", "eye": "OD", "label": "progressing",
     "cdr": [0.45, 0.52, 0.60, 0.68, 0.76], "md": [-1.0, -2.5, -4.5, -7.0, -10.0]},
    {"id": "DEMO-STABLE-02", "eye": "OD", "label": "stable",
     "cdr": [0.54, 0.52, 0.56, 0.53, 0.55], "md": [-1.5, -1.0, -2.0, -1.2, -1.8]},
    {"id": "DEMO-SLOW-03", "eye": "OS", "label": "slowly changing",
     "cdr": [0.50, 0.53, 0.55, 0.58, 0.61], "md": [-1.0, -1.8, -2.2, -3.0, -3.8]},
]


def ensure_schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS analyses (
        id TEXT PRIMARY KEY, patient_id TEXT, eye TEXT, visit_date TEXT, filename TEXT,
        vcdr REAL, hcdr REAL, acdr REAL, prob REAL, prediction TEXT, risk TEXT, result_json TEXT,
        created TEXT DEFAULT CURRENT_TIMESTAMP)""")
    have = {r[1] for r in conn.execute("PRAGMA table_info(analyses)")}
    for col, decl in (("md", "REAL"), ("stage", "INTEGER"), ("stage_label", "TEXT"),
                      ("md_source", "TEXT"), ("damage_prob", "REAL")):
        if col not in have:
            conn.execute(f"ALTER TABLE analyses ADD COLUMN {col} {decl}")
    conn.commit()


def pick(cohort, target_cdr, used, target_md=None, md_weight=0.35):
    """Closest unused eye to the target CDR and, when asked, the target Mean Defect.

    CDR and MD are on different scales, so MD is divided by 10 dB before the two
    distances are combined - a 10 dB error in field counts about as much as a
    0.1 error in ratio.
    """
    free = cohort[~cohort.index.isin(used)]
    if free.empty:
        return None
    cost = (free.vcdr - target_cdr).abs()
    if target_md is not None and "mean_defect" in free.columns:
        with_md = free[free.mean_defect.notna()]
        if len(with_md) > 5:
            cost = ((with_md.vcdr - target_cdr).abs()
                    + md_weight * (with_md.mean_defect - target_md).abs() / 10.0)
    i = cost.idxmin()
    used.add(i)
    return cohort.loc[i]


def store(conn, pipe, row, patient_id, eye, visit_date, profile_label):
    path = Path(row.path)
    if not path.exists():
        return None
    rgb = read_rgb(path)
    result, disc, cup = pipe.analyze(rgb, path.stem)
    aid = uuid.uuid4().hex[:12]
    md = None if pd.isna(row.mean_defect) else float(row.mean_defect)
    cdr = result["component1"]["cdr"]
    prob = (result.get("classification") or {}).get("glaucoma_probability")
    stage = combined_stage(cdr["vertical_cdr"], md, prob)
    result.update({"analysis_id": aid, "patient_id": patient_id, "eye": eye, "visit_date": visit_date,
                   "schema_version": "2.0", "component2": stage,
                   "demo_series": {"note": NOTE, "profile": profile_label, "source_image": path.name,
                                   "source_dataset": row.source}})
    if md is not None:
        result["visual_field"] = {"mean_defect_db": md, "source": row.source}

    scale = min(1.0, 1024 / max(rgb.shape[:2]))

    def small(im, nearest=False):
        if scale >= 1:
            return im
        return cv2.resize(im, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)

    over = make_overlay(rgb, disc, cup)
    d = RESULTS / aid
    d.mkdir(parents=True, exist_ok=True)
    for k, im in {"original": small(rgb), "overlay": small(over), "roi": crop_roi(over, disc),
                  "disc": small(disc * 255, True), "cup": small(cup * 255, True)}.items():
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(d / f"{k}.png"), im)
    (d / "result.json").write_text(json.dumps(result, indent=2))

    k = result.get("classification") or {}
    conn.execute("INSERT INTO analyses (id, patient_id, eye, visit_date, filename, vcdr, hcdr, acdr, prob, "
                 "prediction, risk, result_json, md, stage, stage_label, md_source, damage_prob) "
                 "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (aid, patient_id, eye, visit_date, path.name, cdr["vertical_cdr"], cdr["horizontal_cdr"],
                  cdr["area_cdr"], k.get("glaucoma_probability"), k.get("prediction"), k.get("risk_level"),
                  json.dumps(result), md, stage.get("stage"), stage.get("stage_label"),
                  "measured" if md is not None else "estimated",
                  (stage.get("damage_probability") or {}).get("probability")))
    return cdr["vertical_cdr"], md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(config.ROOT / "data" / "cohort_master.csv"))
    ap.add_argument("--months", type=int, default=11, help="spacing between visits")
    ap.add_argument("--prefer", default="PAPILA", help="dataset to draw from (has real visual fields)")
    ap.add_argument("--reset", action="store_true", help="delete existing DEMO- patients first")
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort).dropna(subset=["vcdr"])
    cohort = cohort[cohort.path.astype(str).str.len() > 0].reset_index(drop=True)
    preferred = cohort[cohort.source == args.prefer]
    pool = preferred if len(preferred) >= 20 else cohort
    print(f"drawing from {len(pool)} eyes ({pool.source.iloc[0]}), "
          f"{pool.mean_defect.notna().sum()} of them with a measured visual field")

    pipe = VisionTrackPipeline()
    RESULTS.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    if args.reset:
        n = conn.execute("DELETE FROM analyses WHERE patient_id LIKE 'DEMO-%'").rowcount
        conn.commit()
        print(f"removed {n} existing demo studies")

    used, today = set(), date.today()
    for prof in PROFILES:
        print(f"\n{prof['id']} ({prof['label']})")
        n = len(prof["cdr"])
        mds = prof.get("md") or [None] * n
        for i, target in enumerate(prof["cdr"]):
            row = pick(pool, target, used, mds[i] if i < len(mds) else None)
            if row is None:
                break
            visit = (today - timedelta(days=int(args.months * 30.4 * (n - 1 - i)))).isoformat()
            got = store(conn, pipe, row, prof["id"], prof["eye"], visit, prof["label"])
            if got:
                v, md = got
                tmd = mds[i] if i < len(mds) else None
                print(f"  {visit}  target CDR {target:.2f}"
                      f"{'' if tmd is None else f' / MD {tmd:+.1f}'} -> measured {v:.3f}"
                      f"{'' if md is None else f', MD {md:+.2f} dB'}  [{Path(row.path).name}]")
        conn.commit()

    rows = conn.execute("SELECT patient_id, COUNT(*) n, MIN(visit_date), MAX(visit_date) "
                        "FROM analyses WHERE patient_id LIKE 'DEMO-%' GROUP BY patient_id").fetchall()
    print("\nseeded:")
    for r in rows:
        print(f"  {r[0]}: {r[1]} visits, {r[2]} to {r[3]}")
    conn.close()
    print(f"\nOpen the app and go to Progression -> {PROFILES[0]['id']}")
    print("Remember to say on the day: the follow-up sequence is constructed, the measurements are real.")


if __name__ == "__main__":
    main()
