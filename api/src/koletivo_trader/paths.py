from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
API_DIR = ROOT / "api"
CONFIGS_DIR = ROOT / "configs"
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "studies" / "results"
JOURNAL_DIR = ROOT / "journal"
UI_DIR = ROOT / "ui"
UI_DIST = UI_DIR / "dist"
UI_PUBLIC = UI_DIR / "public"
