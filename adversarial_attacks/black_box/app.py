
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import streamlit as st

# ---- App setup ----
st.set_page_config(page_title="Adversarial Attacks Dashboard", layout="wide")

# Ensure current directory is on sys.path so local modules import properly
if str(Path(".").resolve()) not in sys.path:
    sys.path.insert(0, str(Path(".").resolve()))

# ---- Logo setup ----
LOGO_PATH = Path(__file__).parent / "logo.png"   # Put your logo.png next to app.py


def _render_sidebar_logo():
    from pathlib import Path
    p = LOGO_PATH
    try:
        if p.exists():
            try:
                st.sidebar.image(str(p), use_column_width=True)
            except Exception as e:
                st.sidebar.markdown("### Adversarial Attack Generator")
                st.sidebar.caption(f"Logo error: {e}")
        else:
            st.sidebar.markdown("### Adversarial Attack Generator")
    except Exception as e:
        st.sidebar.markdown("### Adversarial Attack Generator")
        st.sidebar.caption(f"Logo missing or unreadable: {e}")



def _render_home_logo():
    from pathlib import Path
    p = LOGO_PATH
    try:
        if p.exists():
            try:
                st.image(str(p), width=420)
            except Exception as e:
                st.markdown("## 🔐 Adversarial Attack Generator")
                st.caption(f"Logo error: {e}")
        else:
            st.markdown("## 🔐 Adversarial Attack Generator")
    except Exception as e:
        st.markdown("## 🔐 Adversarial Attack Generator")
        st.caption(f"Logo missing or unreadable: {e}")


# ---- Filesystem snapshot/diff helpers ----
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".streamlit"}
MAX_RECURSE_FILES = 5000  # safety cap

def _iter_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if len(files) >= MAX_RECURSE_FILES:
            break
        if p.is_file():
            parts = set(p.parts)
            if parts & EXCLUDE_DIRS:
                continue
            files.append(p)
    return files

def _snapshot(root: Path) -> Dict[str, Tuple[float, int]]:
    snap = {}
    for f in _iter_files(root):
        try:
            stat = f.stat()
            snap[str(f)] = (stat.st_mtime, stat.st_size)
        except Exception:
            pass
    return snap

def _diff_files(before: Dict[str, Tuple[float, int]], root: Path) -> List[Path]:
    changed = []
    now = {}
    for f in _iter_files(root):
        try:
            stat = f.stat()
            now[str(f)] = (stat.st_mtime, stat.st_size)
        except Exception:
            pass
    for path, (mt, sz) in now.items():
        if path not in before:
            changed.append(Path(path))
        else:
            bmt, bsz = before[path]
            if mt > bmt + 1e-6 or sz != bsz:
                changed.append(Path(path))
    return sorted(changed, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

# ---- Module / IO helpers ----
def _import_modules():
    try:
        import black_box_attack_other  # white-box under black-box scenario
    except Exception as e:
        st.warning(f"Could not import black_box_attack_other.py: {e}")
        black_box_attack_other = None
    try:
        import black_box_attacks        # pure black-box attacks
    except Exception as e:
        st.warning(f"Could not import black_box_attacks.py: {e}")
        black_box_attacks = None
    return black_box_attack_other, black_box_attacks

def _read_config(uploaded_cfg) -> Dict[str, Any]:
    try:
        return json.load(uploaded_cfg)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return {}

def _save_upload(uploaded_file, dest_folder: Path) -> Path:
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest = dest_folder / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest

def _ensure_defaults(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Provide silent default output paths for modules that expect them."""
    cfg = dict(config or {})
    ts = time.strftime("%Y%m%d-%H%M%S")
    cfg.setdefault("evaluation_output_path", f"runs/{mode}/{ts}/eval")
    cfg.setdefault("adversarial_dataset_output_path", f"runs/{mode}/{ts}/adv")
    Path(cfg["evaluation_output_path"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["adversarial_dataset_output_path"]).mkdir(parents=True, exist_ok=True)
    return cfg

def _validate_config(cfg: Dict[str, Any]) -> bool:
    ds = cfg.get("dataset_path")
    if not ds:
        st.error("No dataset specified. Please set `dataset_path` or upload a CSV.")
        return False
    p = Path(ds)
    if not p.exists():
        st.error(f"dataset_path does not exist: {p}")
        return False
    return True

def _run_and_capture(fn, config: Dict[str, Any]) -> str:
    # Capture stdout + traceback
    import traceback
    from contextlib import redirect_stdout
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            fn(config)
    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Run failed: {e}")
        st.code(tb, language="python")
    return buf.getvalue()

def _present_downloads(files: List[Path], title: str = "New/Updated Files"):
    st.markdown(f"### {title}")
    if not files:
        st.write("No new or modified files were detected.")
        return
    exts = st.multiselect(
        "Filter by extension (optional)",
        options=sorted({p.suffix for p in files if p.suffix}),
        default=[]
    )
    shown = [p for p in files if (not exts or p.suffix in exts)]
    for p in shown:
        try:
            with open(p, "rb") as f:
                st.download_button(
                    label=f"Download {p.name} ({p.stat().st_size} bytes)",
                    data=f,
                    file_name=p.name,
                    mime="application/octet-stream",
                    key=f"dl_{p.as_posix()}_{p.stat().st_mtime}"
                )
        except Exception as e:
            st.caption(f"Skipping {p}: {e}")

# ---- Label normalization (with manual picker fallback) ----
def _pick_label_column(df, prior_guess: str = None) -> str:
    st.markdown("**Select the label column**")
    cols = list(df.columns)
    default_index = cols.index(prior_guess) if prior_guess in cols else 0
    return st.selectbox("Label column", cols, index=default_index, key="label_picker")

def _normalize_label_column_inplace(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the dataset has a 'Label' column by trying common alternatives.
    If needed, writes a normalized CSV next to the original and updates config['dataset_path'].
    """
    ds_path = config.get("dataset_path")
    if not ds_path or not Path(ds_path).exists():
        return config  # nothing to do here

    try:
        import pandas as pd
        df = pd.read_csv(ds_path)
    except Exception as e:
        st.warning(f"Could not read dataset at {ds_path}: {e}")
        return config

    if "Label" in df.columns:
        return config

    candidates = ["label", "target", "y", "class", "Class", "category", "Category"]
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break

    if found is None:
        last = df.columns[-1]
        if df[last].nunique() <= max(100, int(0.2 * len(df))):
            found = last

    if found is None:
        st.warning("No 'Label' column found. Choose the correct column below and we'll normalize it.")
        chosen = _pick_label_column(df)
        df = df.rename(columns={chosen: 'Label'})
    else:
        df = df.rename(columns={found: "Label"})

    norm_path = Path(ds_path).with_name(f"{Path(ds_path).stem}_normalized.csv")
    try:
        df.to_csv(norm_path, index=False)
        config = dict(config)
        config['dataset_path'] = str(norm_path)
        st.info(f"Dataset normalized. Using: {norm_path}")
    except Exception as e:
        st.warning(f"Failed to save normalized dataset: {e}")

    return config

# ---- UI Pages ----
def home():
    _render_home_logo()  # big logo at center of home
    st.caption("White-box ⇄ Black-box evaluation hub")
    st.markdown("### Choose a path")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("White-box under Black-box Scenario", use_container_width=True, type="primary"):
            st.session_state['page'] = "white_under_black"
    with c2:
        if st.button("Black-box Attacks", use_container_width=True):
            st.session_state['page'] = "black_box"

    st.divider()
    st.markdown("**Tip:** You can upload a `config.json` below to prefill parameters.")
    st.divider()

    config_upload = st.file_uploader("Upload config.json (optional)", type=["json"], key="home_cfg")
    if config_upload:
        st.session_state['uploaded_config'] = _read_config(config_upload)
        st.success("Config loaded into session. Navigate to a module to review & run.")

def _config_editor(base_cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    st.markdown("#### Configuration")
    cfg = dict(base_cfg) if base_cfg else {}

    # Common fields
    dataset_path = cfg.get("dataset_path", "")
    dataset_name = cfg.get("dataset_name", "dataset")
    columns_to_drop = cfg.get("columns_to_drop", [])

    # Dataset CSV override
    up = st.file_uploader("Upload dataset CSV (optional to override `dataset_path`)", type=["csv"], key=f"csv_{mode}")
    if up:
        saved = _save_upload(up, Path("uploaded"))
        dataset_path = str(saved)
        st.info(f"Dataset saved to {saved}")

    dataset_path = st.text_input("dataset_path", value=dataset_path)
    dataset_name = st.text_input("dataset_name", value=dataset_name)
    cols_str = st.text_input("columns_to_drop (comma-separated)", value=",".join(map(str, columns_to_drop)))
    columns_to_drop = [c.strip() for c in cols_str.split(",")] if cols_str.strip() else []

    if mode == "white_under_black":
        eps = float(st.number_input("eps (epsilon) for FGSM/PGD/BIM", min_value=0.0, value=float(cfg.get("eps", 0.1)), step=0.01))
        type_of_attack = st.multiselect("type_of_attack", ["fgsm", "pgd", "bim", "carlini", "jsma"], default=cfg.get("type_of_attack", ["fgsm"]))
        jsma_theta = float(st.number_input("jsma_theta", min_value=0.0, value=float(cfg.get("jsma_theta", 1.0)), step=0.1))
        jsma_gamma = float(st.number_input("jsma_gamma", min_value=0.0, value=float(cfg.get("jsma_gamma", 0.1)), step=0.05))

        cfg.update(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            columns_to_drop=columns_to_drop,
            eps=eps,
            type_of_attack=type_of_attack,
            jsma_theta=jsma_theta,
            jsma_gamma=jsma_gamma,
        )
    elif mode == "black_box":
        attacks = ["zoo", "hsj", "boundary", "square"]
        type_of_attack = st.multiselect("type_of_attack (black-box)", attacks, default=cfg.get("type_of_attack", ["zoo"]))
        cfg.update(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            columns_to_drop=columns_to_drop,
            type_of_attack=type_of_attack,
        )
    return cfg

def page_white_under_black():
    st.markdown("## White-box Attacks under Black-box Scenario")
    st.caption("Runs FGSM / PGD / BIM / Carlini / JSMA on a simple surrogate as in `black_box_attack_other.py`.")
    base_cfg = st.session_state.get('uploaded_config', {})
    cfg = _config_editor(base_cfg, mode="white_under_black")
    cfg = _normalize_label_column_inplace(cfg)
    cfg = _ensure_defaults(cfg, 'white_under_black')
    if not _validate_config(cfg):
        return

    run = st.button("Run attacks", type="primary")
    if run:
        mod_other, _ = _import_modules()
        if mod_other is None or not hasattr(mod_other, "main"):
            st.error("Module `black_box_attack_other.py` not available or missing `main(config)`")
            return

        root = Path(".").resolve()
        before = _snapshot(root)
        logs = _run_and_capture(mod_other.main, cfg)
        changed = _diff_files(before, root)

        st.markdown("### Logs")
        st.code(logs or "<no output>", language="text")
        _present_downloads(changed, title="New/Updated Files from White-under-Black run")

def page_black_box():
    st.markdown("## Black-box Attacks")
    st.caption("Runs ZOO, HopSkipJump, Boundary, or Square Attack via `black_box_attacks.py`.")
    base_cfg = st.session_state.get('uploaded_config', {})
    cfg = _config_editor(base_cfg, mode="black_box")
    cfg = _normalize_label_column_inplace(cfg)
    cfg = _ensure_defaults(cfg, 'black_box')
    if not _validate_config(cfg):
        return

    run = st.button("Run attacks", type="primary")
    if run:
        _, mod_bb = _import_modules()
        if mod_bb is None or not hasattr(mod_bb, "main"):
            st.error("Module `black_box_attacks.py` not available or missing `main(config)`")
            return

        root = Path(".").resolve()
        before = _snapshot(root)
        logs = _run_and_capture(mod_bb.main, cfg)
        changed = _diff_files(before, root)

        st.markdown("### Logs")
        st.code(logs or "<no output>", language="text")
        _present_downloads(changed, title="New/Updated Files from Black-box run")

# ---- Sidebar + Routing ----
_render_sidebar_logo()
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Home", "White-box under Black-box", "Black-box Attacks"], index=0)

# Maintain page state
if 'page' not in st.session_state:
    st.session_state['page'] = "home"
if choice == "Home":
    st.session_state['page'] = "home"
elif choice == "White-box under Black-box":
    st.session_state['page'] = "white_under_black"
elif choice == "Black-box Attacks":
    st.session_state['page'] = "black_box"

# Render selected page
if st.session_state['page'] == "home":
    home()
elif st.session_state['page'] == "white_under_black":
    page_white_under_black()
elif st.session_state['page'] == "black_box":
    page_black_box()
