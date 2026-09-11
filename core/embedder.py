import json
import os

import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
ADAFACE_PATH = os.path.join(MODELS_DIR, "adaface_ir_101.onnx")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "..", "data", "embeddings.json")
FACES_DIR = os.path.join(BASE_DIR, "..", "data", "faces")

EMBEDDING_MODEL = "adaface-ir101"  # tag stored with each embedding; mismatches are rejected
EMBEDDING_DIM = 512

REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_session = None


def _get_session():
    if not os.path.exists(ADAFACE_PATH):
        raise FileNotFoundError("AdaFace model missing. Run: python setup_models.py")
    global _session
    if _session is None:
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        _session = ort.InferenceSession(ADAFACE_PATH, providers=providers)
    return _session


def _umeyama(src, dst, with_scale=True):
    """Least-squares similarity transform (skimage-compatible)."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = dst_c.T @ src_c / n
    U, D, Vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        s[1, 1] = -1
    R = U @ s @ Vt
    scale = np.trace(np.diag(D) @ s) / (np.sum(src_c ** 2) / n) if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = t
    return M


def align_face(frame, face):
    """Warp the face region to a 112x112 aligned crop using the 5 YuNet landmarks."""
    pts = np.array(face[4:14], dtype=np.float32).reshape(5, 2)
    matrix = _umeyama(pts, REFERENCE_LANDMARKS)
    return cv2.warpAffine(frame, matrix[:2], (112, 112), borderValue=0.0)


def embed(crop):
    """AdaFace embedding (512-d, L2-normalized) from a face crop."""
    blaze = cv2.dnn.blobFromImage(crop, scalefactor=1.0 / 127.5, size=(112, 112),
                                  mean=(127.5, 127.5, 127.5), swapRB=False)
    session = _get_session()
    out = session.run(None, {session.get_inputs()[0].name: blaze})[0]
    feature = out.reshape(-1).astype(np.float32)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature


def mean_embedding(person_dir):
    vectors = []
    for fname in sorted(os.listdir(person_dir)):
        crop = cv2.imread(os.path.join(person_dir, fname))
        if crop is None:
            continue
        vectors.append(embed(crop))
    if not vectors:
        return None
    mean = np.mean(np.array(vectors), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        return {}
    with open(EMBEDDINGS_PATH, "r") as f:
        raw = json.load(f)
    out = {}
    for name, meta in raw.items():
        item = {"embedding": np.array(meta["embedding"], dtype=np.float32), "samples": meta["samples"]}
        for field in ("model", "dim", "name"):
            if meta.get(field):
                item[field] = meta[field]
        out[name] = item
    return out


def save_embeddings(embeddings):
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    raw = {}
    for name, meta in embeddings.items():
        item = {"embedding": meta["embedding"].tolist(), "samples": meta["samples"]}
        item["model"] = meta.get("model", EMBEDDING_MODEL)
        item["dim"] = meta.get("dim", EMBEDDING_DIM)
        if meta.get("name"):
            item["name"] = meta["name"]
        raw[name] = item
    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(raw, f, indent=2)


def embedding_compatible(meta):
    return meta.get("model") == EMBEDDING_MODEL and meta.get("dim") == EMBEDDING_DIM


def cosine_similarity(a, b):
    return float(np.clip(np.dot(a, b), -1.0, 1.0))