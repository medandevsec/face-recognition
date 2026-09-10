import cv2
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")

_detector = None

def detect_faces(frame):
    if not os.path.exists(YUNET_PATH):
        raise FileNotFoundError("YuNet model missing. Run: python setup_models.py")
    global _detector
    h, w = frame.shape[:2]
    if _detector is None:
        _detector = cv2.FaceDetectorYN_create(YUNET_PATH, "", (w, h))
    else:
        _detector.setInputSize((w, h))
    ok, faces = _detector.detect(frame)
    if not ok or faces is None:
        return None
    return faces