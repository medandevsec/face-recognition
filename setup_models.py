import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

def download(url, dest):
    if os.path.exists(dest):
        print(f"exists: {os.path.basename(dest)}")
        return
    print(f"downloading {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"download failed: {e}")
        sys.exit(1)
    print(f"done: {os.path.basename(dest)}")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for fname, url in MODELS.items():
        download(url, os.path.join(MODELS_DIR, fname))
    print("models ready")

if __name__ == "__main__":
    main()