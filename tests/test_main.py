from collections import deque

import numpy as np
import pytest

import main
from core.embedder import EMBEDDING_MODEL, EMBEDDING_DIM


def _vector(rng):
    v = rng.random(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _meta(vec, threshold=None):
    meta = {"embedding": vec, "samples": 1, "model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM}
    if threshold is not None:
        meta["threshold"] = threshold
    return meta


def test_recognize_accepts_stored_per_embedding_threshold():
    rng = np.random.default_rng(1)
    v = _vector(rng)
    embeddings = {"1111222233334444": _meta(v)}
    name, nik, _ = main.recognize(v.copy(), embeddings, main.COSINE_THRESHOLD)
    assert nik == "1111222233334444"


def test_recognize_rejected_by_stricter_stored_threshold():
    rng = np.random.default_rng(2)
    v = _vector(rng)
    approx = _vector(rng)
    approx = approx + 0.6 * v
    approx /= np.linalg.norm(approx)
    embeddings = {"1111222233334444": _meta(v, threshold=0.99)}
    name, nik, pct = main.recognize(approx, embeddings, main.COSINE_THRESHOLD)
    assert nik is None
    assert pct == 0.0


def test_recognize_global_threshold_blocks_weak_match():
    rng = np.random.default_rng(3)
    v = _vector(rng)
    other = _vector(rng)
    embeddings = {"1111222233334444": _meta(v)}
    name, nik, pct = main.recognize(other, embeddings, 0.99)
    assert nik is None


def test_process_frame_no_face_appends_history():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    history = deque(maxlen=main.STABILITY_FRAMES)
    out, count = main.process_frame(frame, {}, main.COSINE_THRESHOLD, history=history)
    assert count == 0
    assert list(history) == [None]