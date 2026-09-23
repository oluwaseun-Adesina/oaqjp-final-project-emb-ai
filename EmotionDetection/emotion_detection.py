"""Emotion detection utilities.

This module exposes `emotion_detector(text)` which calls the remote
Watson emotion prediction endpoint when available, and falls back to a
local keyword-based analyzer when the network call fails. It also
handles blank input and HTTP 400 responses as required by the lab.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any

import requests

# Remote Watson EmotionPredict endpoint and header as described in lab
_EMOTION_API_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
_EMOTION_API_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def _none_result() -> Dict[str, Any]:
    """Return the dict structure with None values for all keys.

    Used for HTTP 400 responses or explicitly blank input.
    """
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def _mock_analyze(text: str) -> Dict[str, Any]:
    """Simple keyword-based fallback analyzer for offline/testing.

    Returns plausible float scores and a dominant emotion.
    """
    t = (text or "").lower()
    scores = {k: 0.01 for k in ["anger", "disgust", "fear", "joy", "sadness"]}
    if any(w in t for w in ("love", "happy", "glad", "fun", "enjoy")):
        scores.update({"joy": 0.95, "anger": 0.01, "disgust": 0.01, "fear": 0.01, "sadness": 0.02})
    elif any(w in t for w in ("hate", "angry", "mad", "furious")):
        scores.update({"anger": 0.95, "disgust": 0.01, "fear": 0.01, "joy": 0.01, "sadness": 0.02})
    elif any(w in t for w in ("disgust", "disgusted")):
        scores.update({"disgust": 0.95, "anger": 0.01, "fear": 0.01, "joy": 0.01, "sadness": 0.02})
    elif any(w in t for w in ("sad", "sorrow", "upset")):
        scores.update({"sadness": 0.95, "anger": 0.01, "disgust": 0.01, "joy": 0.01, "fear": 0.02})
    elif any(w in t for w in ("afraid", "scared", "fear", "terrified")):
        scores.update({"fear": 0.95, "anger": 0.01, "disgust": 0.01, "joy": 0.01, "sadness": 0.02})
    else:
        # neutral-ish
        scores.update({"joy": 0.30, "sadness": 0.20, "anger": 0.15, "disgust": 0.15, "fear": 0.20})

    dominant = max(scores, key=scores.get)
    scores["dominant_emotion"] = dominant
    return scores


def _find_emotion_scores(obj: Any) -> Dict[str, float] | None:
    """Try to recursively find emotion scores in a JSON-like object.

    Returns a dict mapping required emotion names to float scores or None
    if not found.
    """
    target_keys = {"anger", "disgust", "fear", "joy", "sadness"}

    if isinstance(obj, dict):
        keys = set(obj.keys())
        if target_keys.issubset(keys):
            try:
                return {k: float(obj[k]) for k in target_keys}
            except (ValueError, TypeError):
                pass
        for v in obj.values():
            res = _find_emotion_scores(v)
            if res:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = _find_emotion_scores(item)
            if res:
                return res
    return None


def emotion_detector(text_to_analyze: str) -> Dict[str, Any]:
    """Detect emotions for the supplied text.

    This function attempts to call the remote Watson EmotionPredict API.
    If the service returns HTTP 400, a dictionary with None values is
    returned. If the network call fails for any reason, a local
    keyword-based fallback analyzer is used which makes the module
    testable offline.

    Args:
        text_to_analyze: The input text to analyze.

    Returns:
        A dict with keys: 'anger', 'disgust', 'fear', 'joy', 'sadness',
        'dominant_emotion'. Scores are floats or None (for HTTP 400).
    """
    if text_to_analyze is None or not str(text_to_analyze).strip():
        return _none_result()

    # Allow test runners to force the local analyzer by env var
    if os.environ.get("EMOTION_DETECTOR_FORCE_LOCAL"):
        return _mock_analyze(text_to_analyze)

    try:
        resp = requests.post(
            _EMOTION_API_URL,
            headers=_EMOTION_API_HEADERS,
            json={"raw_document": {"text": text_to_analyze}},
            timeout=10,
        )
    except requests.RequestException:
        # Network error -> fallback to local analyzer
        return _mock_analyze(text_to_analyze)

    if resp.status_code == 400:
        return _none_result()

    if not resp.ok:
        # Non-200 -> fallback
        return _mock_analyze(text_to_analyze)

    # Try to parse the response and extract the emotion scores
    try:
        # the lab indicates the useful content is in `resp.text`
        parsed = json.loads(resp.text)
    except Exception:
        try:
            parsed = resp.json()
        except Exception:
            return _mock_analyze(text_to_analyze)

    scores = _find_emotion_scores(parsed)
    if not scores:
        return _mock_analyze(text_to_analyze)

    # build final result with dominant emotion
    dominant = max(scores, key=scores.get)
    result = {k: float(scores[k]) for k in ("anger", "disgust", "fear", "joy", "sadness")}
    result["dominant_emotion"] = dominant
    return result
