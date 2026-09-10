import cv2
import numpy as np
import pytest

from core.embedder import align_face, cosine_similarity, embed
from core.ktp import dewarp_card, extract_face_region

from conftest import models_available

pytestmark = pytest.mark.skipif(not models_available(), reason="ONNX models not downloaded (run python setup_models.py)")


@pytest.fixture(scope="module")
def lena_emb(face_images):
    from conftest import face_of

    img, face = face_of(face_images["lena"])
    return embed(align_face(img, face))


@pytest.fixture
def ktp_img(face_images):
    img = cv2.imread(face_images["lena"])
    face_w = 280
    fh = int(img.shape[0] * face_w / img.shape[1])
    face = cv2.resize(img, (face_w, fh))
    canvas = np.full((950, 1400, 3), 128, np.uint8)
    canvas[120:120 + 630, 150:150 + 1000] = 235
    py = 120 + int(630 * 0.18)
    px = 150 + int(1000 * 0.60)
    canvas[py:py + fh, px:px + face_w] = face
    return canvas


def test_dewarp_finds_face(ktp_img):
    warped, dewarped = dewarp_card(ktp_img)
    assert warped is not None
    assert isinstance(dewarped, bool)


def test_face_region_matches_registered(ktp_img, lena_emb):
    warped, _ = dewarp_card(ktp_img)
    face = extract_face_region(warped)
    assert face is not None
    emb_from_ktp = embed(align_face(warped, face))
    assert cosine_similarity(emb_from_ktp, lena_emb) > 0.5
