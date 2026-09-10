import os
import shutil
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

import cv2
import core.embedder as emb
from core.detector import YUNET_PATH
from core.embedder import SFACE_PATH

FACE_URLS = {
    "lena": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "messi": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
}


@pytest.fixture(scope="session")
def models_ok():
    return os.path.exists(SFACE_PATH) and os.path.exists(YUNET_PATH)


def models_available():
    return os.path.exists(SFACE_PATH) and os.path.exists(YUNET_PATH)


@pytest.fixture(scope="session")
def face_images():
    d = tempfile.mkdtemp(prefix="frtest_")
    paths = {}
    for name, url in FACE_URLS.items():
        target = os.path.join(d, name + ".jpg")
        try:
            urllib.request.urlretrieve(url, target)
            if cv2.imread(target) is None:
                raise OSError("bad image")
            paths[name] = target
        except Exception:
            raise pytest.skip(f"could not download {name} fixture: {url}")
    return paths


@pytest.fixture
def embedder_paths(tmp_path, monkeypatch):
    faces = tmp_path / "data" / "faces"
    faces.mkdir(parents=True)
    monkeypatch.setattr(emb, "EMBEDDINGS_PATH", str(tmp_path / "data" / "embeddings.json"))
    monkeypatch.setattr(emb, "FACES_DIR", str(faces))
    return emb


def face_of(p):
    from core.detector import detect_faces

    img = cv2.imread(p)
    faces = detect_faces(img)
    assert faces is not None and len(faces) >= 1, f"no face detected in {p}"
    return img, faces[0]