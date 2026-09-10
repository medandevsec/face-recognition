import numpy as np
import pytest

from core.embedder import align_face, cosine_similarity, embed

from conftest import models_available

pytestmark = pytest.mark.skipif(not models_available(), reason="ONNX models not downloaded (run python setup_models.py)")


@pytest.fixture(scope="module")
def lena_emb(face_images):
    from conftest import face_of

    img, face = face_of(face_images["lena"])
    return embed(align_face(img, face))


@pytest.fixture(scope="module")
def messi_emb(face_images):
    from conftest import face_of

    img, face = face_of(face_images["messi"])
    return embed(align_face(img, face))


def test_embed_shape_and_norm(lena_emb):
    assert lena_emb.shape == (128,)
    assert np.abs(np.linalg.norm(lena_emb) - 1.0) < 1e-3


def test_same_person_high(lena_emb):
    sim = cosine_similarity(lena_emb, lena_emb)
    assert sim > 0.9


def test_different_person_low(lena_emb, messi_emb):
    sim = cosine_similarity(lena_emb, messi_emb)
    assert sim < 0.35
