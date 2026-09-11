import cv2
import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "..", "data", "embeddings.json")
FACES_DIR = os.path.join(BASE_DIR, "..", "data", "faces")

_recognizer = None

def _get_recognizer():
    if not os.path.exists(SFACE_PATH):
        raise FileNotFoundError("SFace model missing. Run: python setup_models.py")
    global _recognizer
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF_create(SFACE_PATH, "")
    return _recognizer

def align_face(frame, face):
    return _get_recognizer().alignCrop(frame, face)

def embed(aligned):
    feature = _get_recognizer().feature(aligned)
    feature = feature.reshape(-1).astype(np.float32)
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
        if meta.get("name"):
            item["name"] = meta["name"]
        out[name] = item
    return out

def save_embeddings(embeddings):
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    raw = {}
    for name, meta in embeddings.items():
        item = {"embedding": meta["embedding"].tolist(), "samples": meta["samples"]}
        if meta.get("name"):
            item["name"] = meta["name"]
        raw[name] = item
    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(raw, f, indent=2)

def cosine_similarity(a, b):
    return float(np.clip(np.dot(a, b), -1.0, 1.0))