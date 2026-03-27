"""
Load detector + classifier once; run run_single_wav from predict_3stage_with_context.

Used by the Gradio demo to avoid subprocess + model reload per click.
"""
from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"

import sys

sys.path.insert(0, str(SCRIPTS))

from predict_3stage_with_context import (  # noqa: E402
    CLASSIFIER_PATH,
    CONTEXT_ALPHA,
    DEFAULT_PROFILES_JSON,
    DETECTOR_PATH,
    resolve_class_names_for_classifier,
    run_single_wav,
)


def _load_profiles() -> dict:
    if not DEFAULT_PROFILES_JSON.exists():
        return {}
    return json.loads(DEFAULT_PROFILES_JSON.read_text(encoding="utf-8"))


class CachedThreeStagePredictor:
    def __init__(self, classifier_path: Path | None = None):
        self._classifier_path = Path(classifier_path) if classifier_path else CLASSIFIER_PATH
        self._detector = None
        self._classifier = None
        self._class_names: list | None = None
        self._profile_name: str | None = None

    def ensure_models(self) -> None:
        if self._detector is None:
            if not DETECTOR_PATH.exists():
                raise FileNotFoundError(f"Detector missing: {DETECTOR_PATH}")
            if not self._classifier_path.exists():
                raise FileNotFoundError(f"Classifier missing: {self._classifier_path}")
            self._detector = tf.keras.models.load_model(DETECTOR_PATH)
            self._classifier = tf.keras.models.load_model(self._classifier_path)
            self._class_names, _ = resolve_class_names_for_classifier(self._classifier)

    def set_deployment_profile(self, name: str) -> None:
        data = _load_profiles()
        prof = data.get(name)
        if not isinstance(prof, dict):
            avail = ", ".join(sorted(data.keys())) if data else "(empty)"
            raise ValueError(f"Unknown profile '{name}'. Available: {avail}")
        self._profile_name = name
        self._prof = prof
        # Match CLI: profile may pin a classifier (e.g. Malawi ft4)
        cm = prof.get("classifier_model")
        if cm:
            p = Path(cm)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            target = p
        else:
            target = Path(CLASSIFIER_PATH)
        if self._classifier_path.resolve() != target.resolve():
            self._classifier_path = target
            self._classifier = None
            self._class_names = None

    def predict(
        self,
        wav_path: Path | str,
        *,
        infer_location_from_filename: bool = True,
        lat: float | None = None,
        lon: float | None = None,
        location: str | None = None,
        postprocess_mode_override: str | None = None,
    ) -> dict:
        self.ensure_models()
        if self._profile_name is None:
            self.set_deployment_profile("balanced")
        prof = getattr(self, "_prof", {})
        postprocess_mode = (
            postprocess_mode_override
            if postprocess_mode_override is not None
            else str(prof.get("postprocess_mode", "none"))
        )
        ca = float(prof.get("context_alpha", CONTEXT_ALPHA))
        return run_single_wav(
            Path(wav_path),
            detector=self._detector,
            classifier=self._classifier,
            class_names=self._class_names or [],
            location=location,
            month=None,
            hour=None,
            lat=lat,
            lon=lon,
            location_map=None,
            infer_location_from_filename=infer_location_from_filename,
            detector_threshold=float(prof.get("detector_threshold", 0.3)),
            classifier_threshold=float(prof.get("classifier_threshold", 0.35)),
            pool_topk=int(prof.get("pool_topk_windows", 3)),
            rms_gate_rel=float(prof.get("rms_gate_rel", 0.2)),
            rms_gate_abs=float(prof.get("rms_gate_abs", 0.0001)),
            classifier_temperature=float(prof.get("temperature", 0.212)),
            consensus_min_count=int(prof.get("consensus_min_count", 0)),
            context_alpha=ca,
            threshold_on="raw",
            platt_coef=None,
            platt_intercept=None,
            postprocess_mode=postprocess_mode,
        )


_SINGLETON: CachedThreeStagePredictor | None = None


def get_cached_predictor() -> CachedThreeStagePredictor:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = CachedThreeStagePredictor()
    return _SINGLETON
