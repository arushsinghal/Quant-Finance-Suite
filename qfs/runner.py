import os, runpy, contextlib, io, sys
from pathlib import Path

@contextlib.contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def _find_entry(app_dir: Path, candidates=None) -> Path:
    # Try common entry filenames first
    candidates = candidates or [
        "streamlit_app.py", "app.py", "main.py", "Home.py", "home.py",
        "BlackScholes.py", "BlackScholes_Model.py", "VaR.py", "StockSense.py"
    ]
    for name in candidates:
        p = app_dir / name
        if p.exists():
            return p
    # Fallback: first .py file containing "import streamlit"
    for p in app_dir.rglob("*.py"):
        try:
            if "import streamlit" in p.read_text(encoding="utf-8", errors="ignore"):
                return p
        except Exception:
            pass
    raise FileNotFoundError(f"No Streamlit entry script found in {app_dir}")

def mount_subapp(app_folder_name: str, entry_candidates=None):
    """
    Execute a Streamlit sub-app unchanged inside the current page.
    - app_folder_name: folder under ../subapps/
    - entry_candidates: optional list of filenames to try as the entry script
    """
    # Resolve paths
    this_file = Path(__file__).resolve()
    root = this_file.parents[1]          # project root
    app_dir = (root / "subapps" / app_folder_name).resolve()
    entry = _find_entry(app_dir, entry_candidates)

    # Run the sub-app as if it were __main__, with working dir set
    with _chdir(app_dir):
        runpy.run_path(str(entry), run_name="__main__")
