import argparse
import math
import os
import time

import cv2

from core.detector import detect_faces
from core.embedder import align_face, embed, load_embeddings, cosine_similarity
from core.ui import draw_box, draw_corners, draw_mesh

MATCH_THRESHOLD = 0.45
LIVENESS_MIN_TRAVEL = 25.0   # cumulative nose movement (px) to count as live
VERIFY_FRAMES = 60           # consecutive matched frames to confirm (≈2s at 30fps)
WINDOW_NAME = "KTP Verify"
CYAN = (255, 255, 0)

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")

def mask_nik(nik):
    return f"{nik[:2]}xx-xxxx-{nik[-4:]}"

class Verifier:
    def __init__(self, nik, meta, threshold):
        self.nik = nik
        self.meta = meta
        self.threshold = threshold
        self.travel = 0.0
        self.nose = None
        self.matched_frames = 0
        self.face_frames = 0
        self.total_frames = 0
        self.verified = False
        self.verified_since = None

    def step(self, frame):
        self.total_frames += 1
        faces = detect_faces(frame)
        if faces is None or len(faces) == 0:
            self.nose = None
            self.matched_frames = 0
            self.travel = 0.0
            return None, "NO FACE: show your face to the camera", 0.0, False

        self.face_frames += 1
        face = faces[0]
        aligned = align_face(frame, face)
        sim = cosine_similarity(embed(aligned), self.meta["embedding"])
        matched = sim >= self.threshold

        nx, ny = float(face[8]), float(face[9])
        if self.nose is not None:
            self.travel += math.hypot(nx - self.nose[0], ny - self.nose[1])
        self.nose = (nx, ny)

        if matched:
            self.matched_frames += 1
            if self.matched_frames >= VERIFY_FRAMES and self.travel >= LIVENESS_MIN_TRAVEL:
                self.verified = True
                if self.verified_since is None:
                    self.verified_since = time.time()
            status = f"MATCH {sim*100:.0f}% | hold still / move a little | frame {self.matched_frames}/{VERIFY_FRAMES} | travel {self.travel:.0f}px"
        else:
            self.matched_frames = 0
            self.verified = False
            status = f"NO MATCH (sim {sim*100:.0f}%) | target {mask_nik(self.nik)} | threshold {self.threshold:.2f}"
        return face, status, sim, matched

def draw(frame, verifier, face, sim, matched, status):
    if face is not None:
        x, y, w, h = [int(v) for v in face[:4]]
        pad_x, pad_top, pad_bottom = int(w * 0.12), int(h * 0.30), int(h * 0.12)
        bx, by = max(0, x - pad_x), max(0, y - pad_top)
        bw, bh = min(frame.shape[1] - bx, w + pad_x * 2), min(frame.shape[0] - by, h + pad_top + pad_bottom)
        label = f"MATCH {sim*100:.0f}%" if matched else "UNKNOWN"
        draw_mesh(frame, bx, by, bw, bh)
        draw_corners(frame, bx, by, bw, bh)
        draw_box(frame, bx, by, bw, bh, label, sim * 100 if matched else 0.0)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
    if verifier.verified:
        cv2.rectangle(frame, (0, 44), (frame.shape[1], 100), (0, 128, 0), cv2.FILLED)
        cv2.putText(frame, "VERIFIED KTP HOLDER", (12, 86), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)

def summarize(verifier):
    if not verifier.face_frames:
        return "NOT VERIFIED: no face was ever detected"
    if not verifier.verified:
        if verifier.travel < LIVENESS_MIN_TRAVEL:
            return "NOT VERIFIED: identity matched but no movement seen (liveness failed)"
        return "NOT VERIFIED: identity did not match the registered NIK"
    return "VERIFIED OK"

def main():
    parser = argparse.ArgumentParser(description="1:1 live verification against a registered NIK")
    parser.add_argument("nik", help="16-digit NIK registered via ktp_register.py")
    parser.add_argument("--source", default="0", help="0 for webcam, or path to image/video file")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help="cosine similarity threshold")
    parser.add_argument("--no-show", action="store_true", help="run without the preview window (headless)")
    args = parser.parse_args()

    nik = args.nik
    embeddings = load_embeddings()
    meta = embeddings.get(nik)
    if meta is None:
        print(f"NIK {mask_nik(nik)} is not registered. Run: python ktp_register.py {nik} <ktp_image>")
        return

    verifier = Verifier(nik, meta, args.threshold)
    source = args.source
    ext = os.path.splitext(str(source))[1].lower()

    if ext in IMAGE_EXT:
        frame = cv2.imread(source)
        if frame is None:
            print("could not read image")
            return
        face, status, sim, matched = verifier.step(frame)
        if not args.no_show:
            draw(frame, verifier, face, sim, matched, status)
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(status)
        print(summarize(verifier))
        return

    if ext in VIDEO_EXT or source == "0":
        cap = cv2.VideoCapture(int(source) if source == "0" else source)
        if not cap.isOpened():
            print("cannot open source")
            return
        if not args.no_show:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 800, 600)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            face, status, sim, matched = verifier.step(frame)
            if not args.no_show:
                draw(frame, verifier, face, sim, matched, status)
                cv2.imshow(WINDOW_NAME, frame)
            if verifier.verified and verifier.verified_since and time.time() - verifier.verified_since > 1.5:
                break
            if not args.no_show and cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        print(status)
        print(summarize(verifier))
        return

    print("unsupported source")

if __name__ == "__main__":
    main()