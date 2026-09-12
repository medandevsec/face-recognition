import cv2
import argparse
import csv
import os
import time
from core.detector import detect_faces
from core.embedder import (align_face, embed, load_embeddings, cosine_similarity,
                           embedding_compatible, EMBEDDING_DIM)
from core.ui import draw_box, draw_corners, draw_mesh, draw_hud, padded_box

COSINE_THRESHOLD = 0.45  # higher = stricter match (cosine similarity, 0..1); 0.45 aligns with verify.py/validator
WINDOW_NAME = "Face Recognition"
MASTER_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "master_ktp.csv")

def load_master(csv_path=MASTER_CSV):
    """nik -> master row (nama, alamat, ...) for the camera overlay."""
    master = {}
    if not os.path.isfile(csv_path):
        return master
    try:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                nik = (row.get("nik") or "").strip()
                if nik:
                    master[nik] = row
    except Exception as e:
        print(f"warning: failed to read master csv {csv_path}: {e}")
    return master

def recognize(face_vec, embeddings, threshold):
    best_name, best_nik, best_sim = "Unknown", None, 0.0
    for name, meta in embeddings.items():
        if not embedding_compatible(meta) or len(meta["embedding"]) != EMBEDDING_DIM:
            continue
        sim = cosine_similarity(face_vec, meta["embedding"])
        if sim > best_sim:
            best_sim, best_name, best_nik = sim, name, name
    if best_sim >= threshold:
        return best_name, best_nik, best_sim * 100
    return "Unknown", None, 0.0

def process_frame(frame, embeddings, threshold=COSINE_THRESHOLD, master=None):
    faces = detect_faces(frame)
    face_count = 0 if faces is None else len(faces)
    if faces is None:
        return frame, 0

    for face in faces:
        x, y, w, h = [int(v) for v in face[:4]]
        bx, by, bw, bh = padded_box(x, y, w, h, frame.shape[0], frame.shape[1])

        name, match_pct = "Unknown", 0.0
        info = None
        if embeddings:
            aligned = align_face(frame, face)
            name, nik, match_pct = recognize(embed(aligned), embeddings, threshold)
            if nik and master:
                row = master.get(nik)
                if row:
                    addr = (row.get("alamat") or "").strip()
                    info = [f"NIK {nik}"]
                    if addr:
                        info.append(addr)

        draw_mesh(frame, bx, by, bw, bh)
        draw_corners(frame, bx, by, bw, bh)
        draw_box(frame, bx, by, bw, bh, name, match_pct, info)

    return frame, face_count

def run_image(path, embeddings, threshold, output, no_show=False, master=None):
    frame = cv2.imread(path)
    if frame is None:
        print("could not read image")
        return
    frame, _ = process_frame(frame, embeddings, threshold, master)
    out_path = output or "output.jpg"
    cv2.imwrite(out_path, frame)
    print(f"saved to {out_path}")
    if no_show:
        return
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_video(source, embeddings, threshold, output, master=None):
    cam = cv2.VideoCapture(source)
    if not cam.isOpened():
        print("cannot open source")
        return
    writer = None
    if output:
        fps = cam.get(cv2.CAP_PROP_FPS) or 20
        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)
    print("press q to quit")

    prev_time = time.time()
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame, face_count = process_frame(frame, embeddings, threshold, master)

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
    args = parser.parse_args()

    embeddings = load_embeddings()
    if not embeddings:
        print("no registered faces yet. register one first: python register_face.py <name> <image>")
    master = load_master()

    source = args.source
    ext = os.path.splitext(source)[1].lower()

    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        run_image(source, embeddings, args.threshold, args.output, args.no_show, master)
    elif ext in (".mp4", ".avi", ".mov", ".mkv"):
        run_video(source, embeddings, args.threshold, args.output, master)
    else:
        run_video(int(source), embeddings, args.threshold, args.output, master)

if __name__ == "__main__":
    main()