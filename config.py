"""Central configuration for VisionTrack AI (Phase 1)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- data -----------------------------------------------------------------
# Folder where the SMDG-19 Kaggle dataset is unzipped. It must contain
# full-fundus/, optic-disc/, optic-cup/ and metadata - standardized.csv
DATA_DIR = ROOT / "data" / "smdg"
SPLITS_DIR = ROOT / "data" / "splits"
PAPILA_DIR = ROOT / "data" / "papila"

CHECKPOINT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"
SEG_CKPT = CHECKPOINT_DIR / "unet_disc_cup.pt"
CLS_CKPT = CHECKPOINT_DIR / "effnet_b0_glaucoma.pt"

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ---- segmentation (Component 1) -------------------------------------------
SEG_IMG_SIZE = 512
SEG_BASE_CH = 32
SEG_BATCH = 8
SEG_EPOCHS = 40
SEG_LR = 3e-4
SEG_THRESH_DISC = 0.5
SEG_THRESH_CUP = 0.5          # tuned on validation set by evaluate.py --tune

# ---- classification (glaucoma risk) ---------------------------------------
CLS_IMG_SIZE = 384
CLS_BATCH = 24
CLS_EPOCHS = 25
CLS_LR = 3e-4
CLS_THRESH = 0.5              # tuned on validation set (Youden index)

# ---- risk bands shown in the app ------------------------------------------
RISK_BANDS = [(0.0, 0.35, "Low"), (0.35, 0.65, "Moderate"), (0.65, 1.01, "High")]
CDR_SUSPICIOUS = 0.6          # clinical rule-of-thumb flag (vertical CDR)
