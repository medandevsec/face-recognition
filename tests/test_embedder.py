import numpy as np
import pytest

from core.embedder import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    align_face,
    cosine_similarity,
    embed,
    embedding_compatible,
    load_embeddings,
    save_embeddings,
)

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
    assert lena_emb.shape == (512,)
    assert np.abs(np.linalg.norm(lena_emb) - 1.0) < 1e-3


def test_same_person_high(lena_emb):
    sim = cosine_similarity(lena_emb, lena_emb)
    assert sim > 0.9


def test_different_person_low(lena_emb, messi_emb):
    sim = cosine_similarity(lena_emb, messi_emb)
    assert sim < 0.35


def test_roundtrip_preserves_metadata(embedder_paths):
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    save_embeddings({"1234567890123456": {"embedding": vec, "samples": 3, "name": "BUDI SANTOSO"}})
    loaded = load_embeddings()["1234567890123456"]
    assert loaded["samples"] == 3
    assert loaded["name"] == "BUDI SANTOSO"
    assert loaded["model"] == EMBEDDING_MODEL
    assert loaded["dim"] == EMBEDDING_DIM
    assert np.allclose(loaded["embedding"], vec)
    assert embedding_compatible(loaded)

    save_embeddings({"1234567890123456": {"embedding": vec, "samples": 1}})
    assert "name" not in load_embeddings()["1234567890123456"]


def test_embedding_compatible_rejects_old_model():
    old = {"embedding": np.zeros(128, dtype=np.float32), "samples": 1, "model": "sface", "dim": 128}
    assert not embedding_compatible(old)
    missing = {"embedding": np.zeros(512, dtype=np.float32), "samples": 1}
    assert not embedding_compatible(missing)
