"""Scan a CCTV video (file/live/RTSP) for faces registered from KTP photos.

Each detected face is tracked across frames; only sufficiently large faces are
considered, and for every appearance the sharpest matching frame is picked.
When a track's best cosine similarity clears the threshold, the identity is
reported and evidence snapshots (face, aligned crop, annotated frame) are saved
under --evidence-dir.

Usage:
    python tools/scan_cctv.py --source cctv.mp4
    python tools/scan_cctv.py --source rtsp://USER:PASS@CAM/stream
    python tools/scan_cctv.py --source 0            # live webcam
    python tools/scan_cctv.py --source ktps/A.jpg   # single image quick-check
    python tools/scan_cctv.py --source cctv.mp4 --threshold 0.40 --debug
"""

import argparse
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main as app
from core.detector import detect_faces
from core.embedder import align_face, embed, load_embeddings

MIN_FACE_WIDTH = 60          # px; smaller faces are skipped (low confidence)
TRACK_LOST_FRAMES = 15       # frames before a disappeared face is closed out
MATCH_REQUIRED_FRAMES = 2    # min tracked frames before a detection counts
MAX_TRACKS = 16


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


class Track:
    __slots__ = ("box", "lost", "frames", "best_pct", "best_name", "best_nik",
                 "best_pts", "best_frame", "best_box", "best_aligned", "best_orig",
                 "best_ev_sharp", "ev_pct", "ev_name", "ev_nik", "ev_pts",
                 "ev_box", "ev_aligned", "ev_orig")

    def __init__(self, box):
        self.box = tuple(int(v) for v in box[:4])
        self.lost = 0
        self.frames = 0
        self.best_pct = 0.0
        self.best_name = None
        self.best_nik = None
        self.best_pts = 0
        self.best_frame = 0
        self.best_box = None
        self.best_aligned = None
        self.best_orig = None
        self.best_ev_sharp = -1.0
        self.ev_pct = 0.0
        self.ev_name = None
        self.ev_nik = None
        self.ev_pts = 0
        self.ev_box = None
        self.ev_aligned = None
        self.ev_orig = None

    def update(self, face, aligned, name, nik, pct, orig, frame_idx, pts, threshold):
        self.box = tuple(int(v) for v in face[:4])
        self.lost = 0
        self.frames += 1
        if pct > self.best_pct:
            self.best_pct = pct
            self.best_name = name
            self.best_nik = nik
            self.best_pts = pts
            self.best_frame = frame_idx
            self.best_box = face
            self.best_aligned = aligned
            self.best_orig = orig
        if pct >= threshold:
            sharp = cv2.Laplacian(aligned, cv2.CV_64F).var()
            if sharp > self.best_ev_sharp:
                self.best_ev_sharp = sharp
                self.ev_pct = pct
                self.ev_name = name
                self.ev_nik = nik
                self.ev_pts = pts
                self.ev_box = face
                self.ev_aligned = aligned
                self.ev_orig = orig

    def result(self, threshold, min_frames=MATCH_REQUIRED_FRAMES):
        if (self.best_pct >= threshold and self.frames >= min_frames
                and self.best_nik is not None):
            return {"nik": self.best_nik, "name": self.best_name,
                    "pct": self.best_pct, "pts": self.best_pts,
                    "frame": self.best_frame, "box": self.best_box,
                    "aligned": self.best_aligned, "orig": self.best_orig}
        return None


def _match_existing(tracks, face):
    best, best_score = None, 0.3
    for t in tracks:
        score = _iou(t.box, face)
        if score > best_score:
            best, best_score = t, score
    return best


def _save_evidence(evidence_dir, nik, pts, result):
    folder = os.path.join(evidence_dir, nik)
    os.makedirs(folder, exist_ok=True)
    secs = pts // 1000
    base = f"t{secs // 60:02d}m{secs % 60:02d}s_f{result['frame']:05d}_{result['pct']:.0f}pct"
    x, y, w, h = [int(v) for v in result["box"][:4]]
    face_patch = result["orig"][y:y + h, x:x + w]
    cv2.imwrite(os.path.join(folder, base + "_face.jpg"), face_patch)
    cv2.imwrite(os.path.join(folder, base + "_aligned.jpg"), result["aligned"])
    annotated = result["orig"].copy()
    bx, by, bw, bh = app.padded_box(x, y, w, h, annotated.shape[0], annotated.shape[1])
    app.draw_box(annotated, bx, by, bw, bh, result["name"], result["pct"], tag=result["name"])
    cv2.imwrite(os.path.join(folder, base + "_frame.jpg"), annotated)
    return os.path.join(folder, base)


def format_pts(pts):
    secs = pts // 1000
    return f"{secs // 60:02d}:{secs % 60:02d}.{pts % 1000:03d}"


def scan_source(cap, embeddings, threshold, evidence_dir, min_face=MIN_FACE_WIDTH,
                debug=False):
    tracks = []
    summary = {}
    frame_idx = 0
    single_image = False

    def close(track):
        res = track.result(threshold, min_frames=1 if single_image else MATCH_REQUIRED_FRAMES)
        if res is None:
            return
        path = (_save_evidence(evidence_dir, res["nik"], res["pts"], res)
                if evidence_dir else None)
        prev = summary.get(res["nik"])
        if prev is None or res["pct"] > prev[0]:
            summary[res["nik"]] = (res["pct"], res["name"], path, res["pts"])
        print(f"MATCH {res['name']} ({app.mask_nik(res['nik'])}) "
              f"{res['pct']:.0f}% @ {format_pts(res['pts'])} frame {res['frame']}"
              + (f" -> {path}" if path else ""))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pts = int(cap.get(cv2.CAP_PROP_POS_MSEC) or 0)
        if frame_idx == 0:
            single_image = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
        farr = detect_faces(frame)
        faces = [] if farr is None else list(farr)
        matched = set()
        for face in faces:
            x, y, w, h = [int(v) for v in face[:4]]
            if w < min_face:
                continue
            aligned = align_face(frame, face)
            name, nik, pct = app.recognize(embed(aligned), embeddings, threshold)
            if debug and pct > 0:
                print(f"  frame {frame_idx}: {name} ({nik}) {pct:.1f}%")
            tr = _match_existing(tracks, tuple(int(v) for v in face[:4]))
            if tr is None:
                tr = Track(face)
                tracks.append(tr)
            tr.update(face, aligned, name, nik, pct, frame, frame_idx, pts, threshold)
            matched.add(id(tr))
        for tr in list(tracks):
            if id(tr) in matched:
                continue
            tr.lost += 1
            if tr.lost > TRACK_LOST_FRAMES:
                close(tr)
                tracks.remove(tr)
        frame_idx += 1

    for tr in list(tracks):
        close(tr)

    print("\n--- ringkasan ---")
    if not summary:
        print("tidak ada wajah terdaftar yang terdeteksi")
    for nik, (pct, name, path, pts) in sorted(
            summary.items(), key=lambda kv: -kv[1][0]):
        print(f"{app.mask_nik(nik)} {name}: {pct:.0f}% @ {format_pts(pts)}"
              + (f" (bukti: {path})" if path else ""))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Scan CCTV for faces registered from KTP photos")
    ap.add_argument("--source", default="0",
                    help="0 for webcam, a file path (mp4/avi/mov/mkv/jpg), or an RTSP URL")
    ap.add_argument("--threshold", type=float, default=app.COSINE_THRESHOLD,
                    help="cosine threshold (per-embedding stored threshold overrides)")
    ap.add_argument("--min-face", type=int, default=MIN_FACE_WIDTH,
                    help="minimum face width in px to consider (default 60)")
    ap.add_argument("--evidence-dir", default=None,
                    help="folder where match snapshots are saved (default none)")
    ap.add_argument("--debug", action="store_true", help="print per-frame similarities")
    args = ap.parse_args()

    embeddings = load_embeddings()
    if not embeddings:
        print("no registered faces. register KTPs first with ktp_register.py")
        return
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"cannot open source: {args.source}")
        return
    print(f"scanning {args.source} / {len(embeddings)} NIK terdaftar "
          f"(threshold {args.threshold:.2f})")
    if args.evidence_dir:
        os.makedirs(args.evidence_dir, exist_ok=True)
    scan_source(cap, embeddings, args.threshold, args.evidence_dir,
                min_face=args.min_face, debug=args.debug)
    cap.release()


if __name__ == "__main__":
    main()