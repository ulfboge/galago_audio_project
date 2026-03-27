"""
Pull .keras + JSON weights into the repo layout when GALAGO_HF_MODEL_REPO is set.

Intended for Hugging Face Spaces: store large files in a separate Model (or Dataset)
repo and download them on container start. Local dev leaves the variable unset and
uses existing files under models/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths relative to repo root and to the Hub repo (same layout in both).
REQUIRED_REL_PATHS: tuple[str, ...] = (
    "models/detector/galago_detector_best.keras",
    "models/all_species/galago_cnn_all_19classes_improved_best.keras",
    "models/all_species/class_names_19.json",
)

OPTIONAL_REL_PATHS: tuple[str, ...] = (
    "models/all_species/galago_cnn_malawi_spnov_ft4_best.keras",
    "data/species_ranges.json",
)


def _needs_download(path: Path) -> bool:
    return not path.is_file() or path.stat().st_size == 0


def ensure_hf_hub_models() -> None:
    repo_id = os.environ.get("GALAGO_HF_MODEL_REPO", "").strip()
    if not repo_id:
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print(
            "[galago] GALAGO_HF_MODEL_REPO is set but huggingface_hub is not installed:",
            exc,
            file=sys.stderr,
            flush=True,
        )
        return

    revision = os.environ.get("GALAGO_HF_MODEL_REVISION", "").strip() or None
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    def pull(rel: str, optional: bool) -> None:
        dest = PROJECT_ROOT / rel
        if not _needs_download(dest):
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=rel,
                revision=revision,
                token=token,
                local_dir=str(PROJECT_ROOT),
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            if optional:
                print(f"[galago] Optional asset skipped ({rel}): {exc}", flush=True)
            else:
                raise

    for rel in REQUIRED_REL_PATHS:
        pull(rel, optional=False)
    for rel in OPTIONAL_REL_PATHS:
        pull(rel, optional=True)
