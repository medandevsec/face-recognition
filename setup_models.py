import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "adaface_ir_101.onnx": "https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_101.onnx",
}

def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"exists: {os.path.basename(dest)}")
        return
    print(f"downloading {os.path.basename(dest)} ...")
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            got = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
        if total and got != total:
            raise IOError(f"size mismatch: got {got}, expected {total}")
        if got == 0:
            raise IOError("empty download")
        os.replace(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"download failed: {e}")
        sys.exit(1)
    print(f"done: {os.path.basename(dest)} ({got // 1024} KB)")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for fname, url in MODELS.items():
        download(url, os.path.join(MODELS_DIR, fname))
    print("models ready")

if __name__ == "__main__":
    main()