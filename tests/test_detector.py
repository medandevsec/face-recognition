import cv2
import pytest

from core.detector import detect_faces
from conftest import models_available

pytestmark = pytest.mark.skipif(
    not models_available(),
    reason="ONNX models not downloaded (run python setup_models.py)")


def test_detect_lena(face_images):
    faces = detect_faces(cv2.imread(face_images["lena"]))
    assert faces is not None
    assert len(faces) >= 1
    assert len(faces[0]) == 15


def test_detect_messi(face_images):
    faces = detect_faces(cv2.imread(face_images["messi"]))
    assert faces is not None
    assert len(faces) >= 1