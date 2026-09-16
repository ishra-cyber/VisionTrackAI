"""VisionTrack AI - Flask web application (Phase 1).

    python app/app.py            -> http://127.0.0.1:5000
"""
import json
import re
import sqlite3
import sys
import uuid
from datetime import date
from pathlib import Path

import cv2
from flask import Flask, abort, g, jsonify, redirect, render_template, request, send_from_directory, url_for

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.visualize import crop_roi, make_overlay  # noqa: E402
from src.preprocessing import decode_rgb  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
RESULTS = APP_DIR / "static" / "results"
DB_PATH = APP_DIR / "visiontrack.db"
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def create_app(pipeline=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    RESULTS.mkdir(parents=True, exist_ok=True)
    state = {"pipeline": pipeline, "error": None}

    def get_pipeline():
        if state["pipeline"] is None and state["error"] is None:
            try:
                from src.inference import VisionTrackPipeline
                state["pipeline"] = VisionTrackPipeline()
            except FileNotFoundError as e:
                state["error"] = (f"Model checkpoint not found ({e.filename}). Train the models first "
                                  "or copy unet_disc_cup.pt / effnet_b0_glaucoma.pt into checkpoints/.")
            except Exception as e:  # noqa: BLE001
                state["error"] = f"Could not load models: {e}"
        return state["pipeline"]

    # ------------------------------------------------------------ database
    def db():
        if "db" not in g:
            g.db = sqlite3.connect(DB_PATH)
            g.db.row_factory = sqlite3.Row
            g.db.execute("""CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY, patient_id TEXT, eye TEXT, visit_date TEXT,
                filename TEXT, vcdr REAL, hcdr REAL, acdr REAL, prob REAL,
                prediction TEXT, risk TEXT, result_json TEXT,
                created TEXT DEFAULT CURRENT_TIMESTAMP)""")
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    # ------------------------------------------------------------ core
    def run_analysis(file_storage, patient_id, eye, visit_date, extra=None):
        name = file_storage.filename or "upload.png"
        if Path(name).suffix.lower() not in ALLOWED:
            raise ValueError("Unsupported file type. Upload a PNG / JPG / TIFF fundus image.")
        rgb = decode_rgb(file_storage.read())
        pipe = get_pipeline()
        if pipe is None:
            raise RuntimeError(state["error"])
        aid = uuid.uuid4().hex[:12]
        result, disc, cup = pipe.analyze(rgb, image_id=Path(name).stem)
        result.update({"analysis_id": aid, "patient_id": patient_id, "eye": eye, "visit_date": visit_date})
        if extra:
            result.update(extra)

        # save images (cap long side to 1024 px for the browser)
        scale = min(1.0, 1024 / max(rgb.shape[:2]))
        def small(im, nearest=False):
            if scale >= 1:
                return im
            return cv2.resize(im, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
        over = make_overlay(rgb, disc, cup)
        outs = {
            "original": small(rgb), "overlay": small(over),
            "roi": crop_roi(over, disc), "disc": small(disc * 255, True), "cup": small(cup * 255, True),
        }
        d = RESULTS / aid
        d.mkdir(parents=True, exist_ok=True)
        for k, im in outs.items():
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(d / f"{k}.png"), im)
        (d / "result.json").write_text(json.dumps(result, indent=2))

        c, k = result["component1"]["cdr"], result["classification"] or {}
        db().execute("INSERT INTO analyses (id, patient_id, eye, visit_date, filename, vcdr, hcdr, acdr, prob, "
                     "prediction, risk, result_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (aid, patient_id, eye, visit_date, name, c["vertical_cdr"], c["horizontal_cdr"],
                      c["area_cdr"], k.get("glaucoma_probability"), k.get("prediction"), k.get("risk_level"),
                      json.dumps(result)))
        db().commit()
        return result

    def form_fields():
        pid = (request.form.get("patient_id") or "").strip()[:40] or "anonymous"
        eye = request.form.get("eye", "OD") if request.form.get("eye") in ("OD", "OS") else "OD"
        vd = (request.form.get("visit_date") or date.today().isoformat())[:10]
        return pid, eye, vd

    # ------------------------------------------------------------ routes
    @app.context_processor
    def inject_status():
        p = state["pipeline"]
        return {"model_ready": p is not None,
                "cls_ready": bool(p is not None and p.cls is not None)}

    def render_index(error=None, code=200):
        get_pipeline()
        conn = db()
        recent = conn.execute("SELECT * FROM analyses ORDER BY created DESC LIMIT 12").fetchall()
        stats = conn.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT patient_id) AS p, "
                             "SUM(CASE WHEN risk='High' THEN 1 ELSE 0 END) AS high, "
                             "SUM(CASE WHEN vcdr >= ? THEN 1 ELSE 0 END) AS cdr "
                             "FROM analyses", (config.CDR_SUSPICIOUS,)).fetchone()
        return render_template("index.html", error=error or state["error"], recent=recent, stats=stats,
                               today=date.today().isoformat(), cdr_flag=config.CDR_SUSPICIOUS), code

    @app.get("/")
    def index():
        return render_index()

    @app.post("/analyze")
    def analyze():
        f = request.files.get("image")
        if not f or not f.filename:
            return render_index("Please choose a fundus image.", 400)
        try:
            result = run_analysis(f, *form_fields())
        except (ValueError, RuntimeError) as e:
            return render_index(str(e), 400)
        return redirect(url_for("result", aid=result["analysis_id"]))

    @app.get("/result/<aid>")
    def result(aid):
        row = db().execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
        if row is None:
            abort(404)
        r = json.loads(row["result_json"])
        visits = db().execute("SELECT COUNT(*) FROM analyses WHERE patient_id=?", (row["patient_id"],)).fetchone()[0]
        return render_template("result.html", r=r, aid=aid, row=row, visits=visits,
                               cdr_flag=config.CDR_SUSPICIOUS, bands=config.RISK_BANDS)

    @app.get("/result/<aid>/json")
    def result_json(aid):
        if not (RESULTS / aid / "result.json").exists() or not aid.isalnum():
            abort(404)
        return send_from_directory(RESULTS / aid, "result.json", as_attachment=True,
                                   download_name=f"visiontrack_{aid}.json")

    @app.get("/history")
    def history():
        pid = request.args.get("patient_id", "").strip()
        patients = [r["patient_id"] for r in db().execute(
            "SELECT DISTINCT patient_id FROM analyses ORDER BY patient_id").fetchall()]
        rows = []
        if pid:
            rows = db().execute("SELECT * FROM analyses WHERE patient_id=? ORDER BY visit_date, created",
                                (pid,)).fetchall()
        series = {}
        for r in rows:
            if r["vcdr"] is not None:
                series.setdefault(r["eye"], []).append({"date": r["visit_date"], "vcdr": r["vcdr"]})
        return render_template("history.html", pid=pid, patients=patients, rows=rows, series=series)

    # ------------------------------------------------------------ visual field (PAPILA)
    VF_CSV = config.ROOT / "data" / "papila_vf_matched.csv"
    PAPILA_NAME = re.compile(r"^RET(\d{3})(OD|OS)\.(jpg|jpeg|png|tif|tiff|bmp)$", re.I)

    def load_vf():
        if not VF_CSV.exists():
            return None
        import pandas as pd
        df = pd.read_csv(VF_CSV)
        return df.where(df.notna(), None)

    def find_papila_image(name):
        if not PAPILA_NAME.match(name):
            return None
        root = config.PAPILA_DIR
        if not root.exists():
            return None
        return next((p for p in root.rglob(name) if p.is_file()), None)

    @app.get("/visual-field")
    def visual_field():
        df = load_vf()
        if df is None:
            return render_template("visual_field.html", ready=False, csv=VF_CSV)
        import numpy as np
        rows = df.to_dict("records")
        pts = [r for r in rows if r.get("vertical_cdr") is not None and r.get("mean_defect") is not None]
        stats = {"n": len(rows), "patients": df["patient"].nunique() if "patient" in df else len(rows),
                 "od": int((df["eye"] == "OD").sum()), "os": int((df["eye"] == "OS").sum()),
                 "md_mean": float(df["mean_defect"].mean()) if len(df) else None,
                 "cdr_mean": float(df["vertical_cdr"].dropna().mean()) if df["vertical_cdr"].notna().any() else None,
                 "r": None, "slope": None, "icpt": None,
                 "dx": {int(k): int(v) for k, v in df["diagnosis"].dropna().astype(int).value_counts().items()}
                 if "diagnosis" in df else {}}
        if len(pts) > 2:
            x = np.array([p["vertical_cdr"] for p in pts], float)
            y = np.array([p["mean_defect"] for p in pts], float)
            if x.std() > 0 and y.std() > 0:
                stats["r"] = float(np.corrcoef(x, y)[0, 1])
                stats["slope"], stats["icpt"] = (float(v) for v in np.polyfit(x, y, 1))
        md_min = min([p["mean_defect"] for p in pts] + [-5.0])
        md_lo = float(np.floor(md_min / 5.0) * 5)
        md_hi = max(5.0, float(np.ceil(max([p["mean_defect"] for p in pts] + [0.0]) / 5.0) * 5))
        val, figs = None, []
        vdir = APP_DIR / "static" / "validation"
        if (vdir / "summary.json").exists():
            val = json.loads((vdir / "summary.json").read_text(encoding="utf-8"))
            figs = sorted(p.name for p in vdir.glob("*.png"))
        return render_template("visual_field.html", ready=True, rows=rows, pts=pts, stats=stats, val=val, figs=figs,
                               md_lo=md_lo, md_hi=md_hi, cdr_flag=config.CDR_SUSPICIOUS,
                               images_ok=config.PAPILA_DIR.exists())

    @app.post("/visual-field/analyze/<name>")
    def visual_field_analyze(name):
        path = find_papila_image(name)
        df = load_vf()
        if path is None or df is None:
            abort(404)
        match = df[df["image"] == name]
        rec = match.iloc[0].to_dict() if len(match) else {}
        m = PAPILA_NAME.match(name)

        class _File:
            filename = name
            def read(self):
                return path.read_bytes()

        extra = {"visual_field": {
            "source": "PAPILA",
            "mean_defect_db": rec.get("mean_defect"),
            "diagnosis": rec.get("diagnosis"),
            "age": rec.get("age"),
            "iop": rec.get("iop"),
        }}
        try:
            result = run_analysis(_File(), f"PAPILA-{m.group(1)}", m.group(2).upper(),
                                  date.today().isoformat(), extra)
        except (ValueError, RuntimeError) as e:
            return render_index(str(e), 400)
        return redirect(url_for("result", aid=result["analysis_id"]))

    @app.post("/api/analyze")
    def api_analyze():
        f = request.files.get("image")
        if not f:
            return jsonify(error="multipart field 'image' is required"), 400
        try:
            return jsonify(run_analysis(f, *form_fields()))
        except (ValueError, RuntimeError) as e:
            return jsonify(error=str(e)), 400

    @app.get("/health")
    def health():
        p = get_pipeline()
        return jsonify(status="ok" if p else "models_missing", error=state["error"],
                       classifier_loaded=bool(p and p.cls is not None))

    @app.errorhandler(413)
    def too_large(_e):
        return render_index("File too large (maximum 20 MB).", 413)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
