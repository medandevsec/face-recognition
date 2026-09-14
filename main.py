import cv2
import argparse
import csv
import os
import time
import numpy as np
from collections import deque
from core.detector import detect_faces
from core.embedder import (align_face, embed, load_embeddings, cosine_similarity,
                           embedding_compatible, EMBEDDING_DIM)
from core.ui import draw_box, draw_corners, draw_mesh, draw_hud, draw_side_panel, padded_box

# KTP-photo templates match live webcam faces at lower similarity than
# webcam-trained templates; 0.40 keeps strangers out while admitting genuine
# KTP->webcam matches. Per-embedding threshold in embeddings.json overrides this.
COSINE_THRESHOLD = 0.40
STABILITY_FRAMES = 30   # rolling window over which a NIK must persist before shown
STABILITY_REQUIRE = 3   # min appearances in that window
PROBE_UPSCALE_MIN = 120  # px; smaller faces are aligned from a 2x-upscaled frame
CAM_WIDTH, CAM_HEIGHT = 1280, 720
WINDOW_NAME = "Face Recognition"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(ROOT_DIR, "data", "master_ktp.csv")
PERSONAL_CSV = os.path.join(ROOT_DIR, "data", "personal_info.csv")  # local-only overlay info (gitignored)

def mask_nik(nik):
    return f"{nik[:2]}xx-xxxx-{nik[-4:]}"

def load_master(csv_path=MASTER_CSV):
    """nik -> master row (nama, alamat, ...) for the camera overlay.

    The committed master CSV only holds synthetic example people. A local,
    git-ignored data/personal_info.csv is merged on top so real people (e.g. an
    employee testing on their own webcam) still get their name/address shown
    without leaking that data into the repo.
    """
    master = {}
    for path in (csv_path, PERSONAL_CSV):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    nik = (row.get("nik") or "").strip()
                    if nik:
                        master[nik] = row
        except Exception as e:
            print(f"warning: failed to read {path}: {e}")
    return master

def recognize(face_vec, embeddings, threshold):
    best_name, best_nik, best_sim = "Unknown", None, 0.0
    for name, meta in embeddings.items():
        if not embedding_compatible(meta) or len(meta["embedding"]) != EMBEDDING_DIM:
            continue
        sim = cosine_similarity(face_vec, meta["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_name = meta.get("name") or name
            best_nik = name
    # a per-embedding threshold (set at registration) overrides the global one
    eff = threshold
    if best_nik is not None:
        stored = embeddings.get(best_nik, {}).get("threshold")
        if stored is not None:
            eff = stored
    if best_sim >= eff:
        return best_name, best_nik, best_sim * 100
    return "Unknown", None, 0.0

def _aligned_probe(frame, face):
    """Align a face to 112x112. Distant/small faces are aligned from a 2x
    upscaled frame so the crop keeps more pixels (better embedding)."""
    if float(face[2]) < PROBE_UPSCALE_MIN:
        big = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        scaled = np.asarray(face, dtype=np.float32).copy()
        scaled[0:14] *= 2.0
        return align_face(big, scaled)
    return align_face(frame, face)


def process_frame(frame, embeddings, threshold=COSINE_THRESHOLD, master=None,
                  history=None, require=STABILITY_REQUIRE, debug=False):
    faces = detect_faces(frame)
    face_count = 0 if faces is None else len(faces)
    if faces is None or not embeddings:
        if history is not None:
            history.append(None)
        return frame, 0

    best_panel, best_match, best_nik = None, 0.0, None
    for face in faces:
        x, y, w, h = [int(v) for v in face[:4]]
        bx, by, bw, bh = padded_box(x, y, w, h, frame.shape[0], frame.shape[1])

        aligned = _aligned_probe(frame, face)
        name, nik, match_pct = recognize(embed(aligned), embeddings, threshold)
        if debug and match_pct > 0:
            print(f"  sim {name} ({nik}): {match_pct:.1f}%")
        if match_pct > best_match and nik and master:
            row = master.get(nik)
            if row:
                best_match = match_pct
                best_nik = nik
                best_panel = [
                    ("NIK", mask_nik(nik)),
                    ("NAMA", (row.get("nama") or "").strip() or None),
                    ("ALAMAT", (row.get("alamat") or "").strip() or None),
                    ("MATCH", f"{match_pct:.0f}%"),
                ]

        draw_mesh(frame, bx, by, bw, bh)
        draw_corners(frame, bx, by, bw, bh)
        draw_box(frame, bx, by, bw, bh, name, match_pct, tag=name)

    if history is not None:
        history.append(best_nik)
        if best_nik is None or history.count(best_nik) < require:
            best_panel = None
    if best_panel:
        draw_side_panel(frame, best_panel)

    return frame, face_count

def run_image(path, embeddings, threshold, output, no_show=False, master=None, debug=False):
    frame = cv2.imread(path)
    if frame is None:
        print("could not read image")
        return
    frame, _ = process_frame(frame, embeddings, threshold, master, debug=debug)
    out_path = output or "output.jpg"
    cv2.imwrite(out_path, frame)
    print(f"saved to {out_path}")
    if no_show:
        return
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_video(source, embeddings, threshold, output, master=None, debug=False):
    cam = cv2.VideoCapture(source)
    if not cam.isOpened():
        print("cannot open source")
        return
    if str(source).isdigit():
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    writer = None
    if output:
        fps = cam.get(cv2.CAP_PROP_FPS) or 20
        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)
    print("press q to quit")
    history = deque(maxlen=STABILITY_FRAMES)

    prev_time = time.time()
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame, face_count = process_frame(frame, embeddings, threshold, master,
                                          history=history, debug=debug)

        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now
        draw_hud(frame, fps, face_count)

        if writer:
            writer.write(frame)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    if writer:
        writer.release()
        print(f"saved to {output}")
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 for webcam, or path to image/video file")
    parser.add_argument("--output", default=None, help="path to save result (image or video)")
    parser.add_argument("--threshold", type=float, default=COSINE_THRESHOLD,
                        help="cosine similarity threshold (0..1), higher = stricter")
    parser.add_argument("--no-show", action="store_true",
                        help="headless: do not open a window or wait for input")
    parser.add_argument("--debug", action="store_true",
                        help="print the live cosine similarity of every detected face")
    args = parser.parse_args()

    embeddings = load_embeddings()
    if not embeddings:
        print("no registered faces yet. register one first: python register_face.py <name> <image>")
    master = load_master()

    source = args.source
    ext = os.path.splitext(source)[1].lower()

    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        run_image(source, embeddings, args.threshold, args.output, args.no_show, master, debug=args.debug)
    elif ext in (".mp4", ".avi", ".mov", ".mkv"):
        run_video(source, embeddings, args.threshold, args.output, master, debug=args.debug)
    else:
        run_video(int(source), embeddings, args.threshold, args.output, master, debug=args.debug)

if __name__ == "__main__":
    main()