import argparse
import collections
import math
import os
import time

import cv2
import numpy as np

from core.detector import detect_faces
from core.embedder import align_face, embed, load_embeddings, cosine_similarity
from core.ui import draw_box, draw_corners, draw_mesh, padded_box

MATCH_THRESHOLD = 0.45
LIVENESS_MIN_TRAVEL = 25.0    # cumulative nose movement (px) when using --motion-only
VERIFY_SECONDS = 3.0          # consecutive matched time (adapted to source fps) to confirm
STREAK_DECAY = 2              # points lost per brief gap frame (blinks/interruptions)
GAP_RESET_FRAMES = 10         # continuous gap frames before the streak is fully reset
HARD_RESET_MARGIN = 0.20      # sim much lower than threshold => different person, hard reset
BLINK_DROP = 8.0              # eye-brightness drop (0..255) required to register a blink
WINDOW_NAME = "KTP Verify"
CYAN = (255, 255, 0)

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def mask_nik(nik):
    return f"{nik[:2]}xx-xxxx-{nik[-4:]}"


class BlinkDetector:
    """Heuristic blink counter from the two eye areas of the face bbox.

    Uses bbox-proportional eye bands (not the per-frame landmarks, which can
    jump when the eye region darkens) and only analyses frames where the face
    is roughly stable, so movement is not mistaken for a blink.
    """

    EYE_Y = 0.38          # height fraction of the eye band centre
    EYE_H = 0.14          # band height fraction (of face height)
    EYE_W = 0.32          # band width fraction (of face width)

    def __init__(self, drop=BLINK_DROP):
        self.drop = drop
        self.history = collections.deque(maxlen=15)
        self.blinks = 0
        self._state = "open"
        self._dip_len = 0
        self._unstable = 0

    def reset(self):
        self.history.clear()
        self._state = "open"
        self._dip_len = 0
        self._unstable = 0

    def update(self, gray, face, stable):
        if not stable:
            self._unstable += 1
            if self._unstable > 4:
                self._state = "open"
                self._dip_len = 0
            return
        self._unstable = 0
        x, y, w, h = [int(v) for v in face[:4]]
        if w < 20 or h < 20:
            return
        values = []
        cy = y + int(h * self.EYE_Y)
        for cx in (x + int(w * 0.28), x + int(w * 0.72)):
            ww, wh = int(w * self.EYE_W), int(h * self.EYE_H)
            x0 = max(0, cx - ww // 2)
            y0 = max(0, cy - wh // 2)
            x1 = min(gray.shape[1], x0 + ww)
            y1 = min(gray.shape[0], y0 + wh)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            roi = cv2.resize(gray[y0:y1, x0:x1], (32, 16))
            values.append(float(roi.mean()))
        if len(values) < 2:
            return
        value = float(np.mean(values))
        self.history.append(value)
        if len(self.history) < 12:
            return

        baseline = float(np.median(self.history))
        below = (baseline - value) > self.drop and value < baseline
        if below:
            if self._state == "open":
                self._state = "dip"
                self._dip_len = 1
            else:
                self._dip_len += 1
                if self._dip_len > 12:
                    self._state = "open"
                    self._dip_len = 0
        else:
            if self._state == "dip":
                self.blinks += 1
            self._state = "open"
            self._dip_len = 0


class Verifier:
    def __init__(self, nik, meta, threshold, motion_only=False, verify_frames=60):
        self.nik = nik
        self.meta = meta
        self.threshold = threshold
        self.motion_only = motion_only
        self.verify_frames = verify_frames
        self.travel = 0.0
        self.nose = None
        self.streak = 0
        self.gap = 0
        self.face_frames = 0
        self.total_frames = 0
        self.verified = False
        self.verified_since = None
        self.blink = BlinkDetector()

    def _decay(self, hard):
        self.streak = 0 if hard else max(0, self.streak - STREAK_DECAY)

    def step(self, frame):
        self.total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(frame)
        if faces is None or len(faces) == 0:
            self.nose = None
            self.travel = 0.0
            self.gap += 1
            self._decay(self.gap >= GAP_RESET_FRAMES)
            self.blink.reset()
            return None, "NO FACE: show your face to the camera", 0.0, False

        self.face_frames += 1
        face = faces[0]
        x, y, w, h = [int(v) for v in face[:4]]
        aligned = align_face(frame, face)
        sim = cosine_similarity(embed(aligned), self.meta["embedding"])
        matched = sim >= self.threshold

        nx, ny = float(face[8]), float(face[9])
        if self.nose is not None:
            self.travel += math.hypot(nx - self.nose[0], ny - self.nose[1])
        stable = self.nose is None or math.hypot(nx - self.nose[0], ny - self.nose[1]) < w * 0.03
        self.nose = (nx, ny)
        self.blink.update(gray, face, stable)

        if matched:
            self.gap = 0
            self.streak += 1
            live = self.travel >= LIVENESS_MIN_TRAVEL if self.motion_only else self.blink.blinks >= 1
            if self.streak >= self.verify_frames and live:
                self.verified = True
                if self.verified_since is None:
                    self.verified_since = time.time()
            signal = f"motion {self.travel:.0f}px" if self.motion_only else f"blinks {self.blink.blinks}"
            status = f"MATCH {sim*100:.0f}% | {signal} | streak {self.streak}/{self.verify_frames}"
            if not self.motion_only:
                status += " | blink now" if self.blink.blinks < 1 else " | live"
        else:
            self.gap += 1
            hard = sim <= self.threshold - HARD_RESET_MARGIN or self.gap >= GAP_RESET_FRAMES
            self._decay(hard)
            self.verified = False
            status = f"NO MATCH (sim {sim*100:.0f}%) | target {mask_nik(self.nik)} | threshold {self.threshold:.2f}"
        return face, status, sim, matched


def draw(frame, verifier, face, sim, matched, status):
    if face is not None:
        x, y, w, h = [int(v) for v in face[:4]]
        bx, by, bw, bh = padded_box(x, y, w, h, frame.shape[0], frame.shape[1])
        label = f"MATCH {sim*100:.0f}%" if matched else "Unknown"
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
    if verifier.verified:
        return "VERIFIED OK"
    if verifier.streak <= 0:
        return "NOT VERIFIED: identity did not match the registered NIK"
    if verifier.streak < verifier.verify_frames:
        return f"NOT VERIFIED: identity matched but for only {verifier.streak}/{verifier.verify_frames} frames"
    if verifier.motion_only:
        return "NOT VERIFIED: movement below the liveness threshold"
    return "NOT VERIFIED: no blink detected (liveness failed)"


def main():
    parser = argparse.ArgumentParser(description="1:1 live verification against a registered NIK")
    parser.add_argument("nik", help="16-digit NIK registered via ktp_register.py")
    parser.add_argument("--source", default="0", help="0 for webcam, or path to image/video file")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help="cosine similarity threshold")
    parser.add_argument("--motion-only", action="store_true",
                        help="use facial movement instead of blink detection for liveness")
    parser.add_argument("--no-show", action="store_true", help="run without the preview window (headless)")
    args = parser.parse_args()

    nik = args.nik
    embeddings = load_embeddings()
    meta = embeddings.get(nik)
    if meta is None:
        print(f"NIK {mask_nik(nik)} is not registered. Run: python ktp_register.py {nik} <ktp_image>")
        return

    verifier = Verifier(nik, meta, args.threshold, motion_only=args.motion_only)
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

    if ext in VIDEO_EXT or source.isdigit():
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            print("cannot open source")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        verify_frames = max(30, int(VERIFY_SECONDS * fps))
        verifier = Verifier(nik, meta, args.threshold, motion_only=args.motion_only,
                            verify_frames=verify_frames)
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