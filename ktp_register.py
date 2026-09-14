import argparse
import cv2
import os
import re
import sys
import time
from core.embedder import align_face, mean_embedding, load_embeddings, save_embeddings, FACES_DIR
from core.detector import detect_faces
from core.ktp import dewarp_card, extract_face_region
from core.ktp_ocr import extract_fields, extract_text, find_tesseract

MIN_CROP_SHARPNESS = 30  # Laplacian variance; blurrier aligned crops are dropped

def mask_nik(nik):
    return f"{nik[:2]}xx-xxxx-{nik[-4:]}"

def _finalize_registration(nik, name, threshold=None):
    """Recompute the mean embedding from the NIK's photo folder and save it."""
    person_dir = os.path.join(FACES_DIR, nik)
    embedding = mean_embedding(person_dir)
    if embedding is None:
        print("could not compute embedding")
        return False
    embeddings = load_embeddings()
    meta = {"embedding": embedding, "samples": len(os.listdir(person_dir))}
    if name:
        meta["name"] = name
    if threshold is not None:
        meta["threshold"] = threshold
    embeddings[nik] = meta
    save_embeddings(embeddings)
    print(f"registered NIK {mask_nik(nik)} ({len(os.listdir(person_dir))} sample(s))"
          + (f", name: {name}" if name else ""))
    return True

def _ktp_view_crops(image, face):
    """Several aligned 112x112 views of the KTP face from ONE photo so the mean
    embedding is less sensitive to pose/skew: base alignment, small rotation and
    scale jitter, plus an alignment from a 2x-upscaled frame (more source pixels)."""
    base = align_face(image, face)
    crops = [base]
    for deg, scale in ((-3, 1.0), (3, 1.0), (0, 0.9), (0, 1.1)):
        matrix = cv2.getRotationMatrix2D((56.0, 56.0), deg, scale)
        crops.append(cv2.warpAffine(base, matrix, (112, 112), borderValue=0.0))
    big = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    f2 = [v * 2.0 for v in face]
    crops.append(align_face(big, f2))
    return crops

def register_from_ktp(nik, image_path, name=None, threshold=None):
    if not re.fullmatch(r"\d{16}", nik):
        print("invalid NIK: must be exactly 16 digits, e.g. 3273012501900001")
        return False
    img = cv2.imread(image_path)
    if img is None:
        print("could not read image")
        return False

    warped, dewarped = dewarp_card(img)
    candidates = []
    for source, label in ((warped, "dewarped card"), (img, "original photo")):
        face = extract_face_region(source, dewarped=label == "dewarped card")
        if face is not None:
            candidates.append((source, face, label))
    if not candidates:
        print("no face found in KTP image, try a clearer scan/photo")
        return False

    crops = []
    for source, face, label in candidates:
        for crop in _ktp_view_crops(source, face):
            sharpness = cv2.Laplacian(crop, cv2.CV_64F).var()
            if sharpness >= MIN_CROP_SHARPNESS:
                crops.append((crop, sharpness, label))
    if not crops:
        print("no usable (sharp) face crop extracted from the KTP image")
        return False

    person_dir = os.path.join(FACES_DIR, nik)
    os.makedirs(person_dir, exist_ok=True)
    for fname in os.listdir(person_dir):
        os.remove(os.path.join(person_dir, fname))
    for i, (crop, sharpness, label) in enumerate(sorted(crops, key=lambda c: -c[1])):
        cv2.imwrite(os.path.join(person_dir, f"{i}.jpg"), crop)

    print(f"registered NIK {mask_nik(nik)} from KTP ({len(crops)} views;"
          f" dewarped: {dewarped}, best sharpness {max(s for _, s, _ in crops):.0f})"
          + (f", name: {name}" if name else ""))
    return _finalize_registration(nik, name, threshold=threshold)

def register_from_camera(nik, n, source=0, name=None, interval=2.0, threshold=None):
    """Capture n live webcam frames of the holder and register them under the NIK."""
    if not re.fullmatch(r"\d{16}", nik):
        print("invalid NIK: must be exactly 16 digits, e.g. 3273012501900001")
        return False
    person_dir = os.path.join(FACES_DIR, nik)
    os.makedirs(person_dir, exist_ok=True)
    start = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])

    cam = cv2.VideoCapture(source)
    if not cam.isOpened():
        print(f"cannot open webcam source {source}")
        return False

    print(f"capturing {n} live samples for NIK {mask_nik(nik)}. look straight at the camera...")
    saved, t0 = 0, time.time()
    while saved < n:
        ok, frame = cam.read()
        if not ok:
            break
        faces = detect_faces(frame)
        good = False
        if faces is not None and len(faces) == 1:
            x, y, w, h = [int(v) for v in faces[0][:4]]
            cx = (x + w / 2) / frame.shape[1]
            cy = (y + h / 2) / frame.shape[0]
            if 0.25 < cx < 0.75 and 0.2 < cy < 0.7 and w > 60:
                good = True
        if good and time.time() - t0 >= interval:
            aligned = align_face(frame, faces[0])
            sharp = cv2.Laplacian(aligned, cv2.CV_64F).var()
            if sharp >= 30:
                cv2.imwrite(os.path.join(person_dir, f"{start + saved}.jpg"), aligned)
                saved += 1
                print(f"  captured {saved}/{n} (sharpness {sharp:.0f})")
            t0 = time.time()

        disp = frame.copy()
        if good:
            x, y, w, h = [int(v) for v in faces[0][:4]]
            cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(disp, f"LOOK STRAIGHT  samples {saved}/{n}  (q=quit)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Register from camera", disp)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cam.release()
    cv2.destroyAllWindows()

    if saved == 0:
        print("no usable samples captured")
        return False
    return _finalize_registration(nik, name, threshold=threshold)

def main():
    parser = argparse.ArgumentParser(description="Register a person's face from a KTP photo, keyed by NIK")
    parser.add_argument("nik", nargs="?", default=None, help="16-digit NIK (omit when using --ocr)")
    parser.add_argument("ktp_image", nargs="?", default=None, help="path to a KTP photo/scan (not needed with --capture)")
    parser.add_argument("--ocr", action="store_true",
                        help="read NIK (and name) automatically from the KTP photo via OCR")
    parser.add_argument("--tesseract-cmd", default=None, help="path to tesseract (auto-detected if omitted)")
    parser.add_argument("--tessdata-dir", default=None, help="folder containing *.traineddata (auto-detected)")
    parser.add_argument("--lang", default=None, help="OCR language (auto: ind if available, else eng)")
    parser.add_argument("--capture", type=int, default=0, metavar="N",
                        help="register from the webcam instead of an image: capture N live samples")
    parser.add_argument("--source", type=int, default=0, help="webcam index for --capture (default 0)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="seconds between captured samples (default 2)")
    parser.add_argument("--name", default=None, help="display name stored with the registration")
    parser.add_argument("--threshold", type=float, default=None,
                        help="optional per-embedding cosine threshold (0..1); "
                             "used instead of the global default in main.py")
    args = parser.parse_args()

    nik = args.nik
    name = args.name or None
    if args.ocr:
        tess = args.tesseract_cmd or find_tesseract()
        if not tess:
            print("tesseract not found; pass --tesseract-cmd")
            sys.exit(1)
        tessdata = args.tessdata_dir or find_tessdata(tess, args.tessdata_dir)
        img = cv2.imread(args.ktp_image)
        text = extract_text(img, tesseract_cmd=tess, tessdata_dir=tessdata, lang=args.lang)
        fields = extract_fields(text, image=img, tesseract_cmd=tess, tessdata_dir=tessdata, lang=args.lang)
        nik = fields["nik"]
        name = fields["name"]
        if not nik:
            print("OCR could not find a 16-digit NIK; run ktp_ocr.py to inspect the raw text")
            sys.exit(1)
        print(f"OCR >> NIK {mask_nik(nik)}" + (f", name: {name}" if name else ""))
    if not nik or (args.capture <= 0 and not args.ktp_image):
        print("usage: python ktp_register.py <NIK 16-digit> <ktp_image>    or    --ocr <ktp_image>    or    <NIK> --capture <N>")
        sys.exit(1)

    if args.capture > 0:
        register_from_camera(nik, args.capture, source=args.source, name=name,
                             interval=args.interval, threshold=args.threshold)
    else:
        register_from_ktp(nik, args.ktp_image, name=name, threshold=args.threshold)

if __name__ == "__main__":
    main()