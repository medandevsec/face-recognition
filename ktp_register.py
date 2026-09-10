import argparse
import cv2
import os
import re
import sys
from core.embedder import align_face, mean_embedding, load_embeddings, save_embeddings, FACES_DIR
from core.ktp import dewarp_card, extract_face_region
from core.ktp_ocr import extract_fields, extract_text, find_tessdata, find_tesseract

def mask_nik(nik):
    return f"{nik[:2]}xx-xxxx-{nik[-4:]}"

def register_from_ktp(nik, image_path, name=None):
    if not re.fullmatch(r"\d{16}", nik):
        print("invalid NIK: must be exactly 16 digits, e.g. 3273012501900001")
        return False
    img = cv2.imread(image_path)
    if img is None:
        print("could not read image")
        return False

    warped, dewarped = dewarp_card(img)
    face = extract_face_region(warped)
    if face is None:
        print("no face found in KTP image, try a clearer scan/photo")
        return False

    aligned = align_face(warped, face)
    person_dir = os.path.join(FACES_DIR, nik)
    os.makedirs(person_dir, exist_ok=True)
    count = len(os.listdir(person_dir))
    cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), aligned)

    embedding = mean_embedding(person_dir)
    if embedding is None:
        print("could not compute embedding")
        return False

    embeddings = load_embeddings()
    meta = {"embedding": embedding, "samples": len(os.listdir(person_dir))}
    if name:
        meta["name"] = name
    embeddings[nik] = meta
    save_embeddings(embeddings)
    print(f"registered NIK {mask_nik(nik)} from KTP ({len(os.listdir(person_dir))} sample(s), card dewarped: {dewarped})"
          + (f", name: {name}" if name else ""))
    return True

def main():
    parser = argparse.ArgumentParser(description="Register a person's face from a KTP photo, keyed by NIK")
    parser.add_argument("nik", nargs="?", default=None, help="16-digit NIK (omit when using --ocr)")
    parser.add_argument("ktp_image", help="path to a KTP photo/scan")
    parser.add_argument("--ocr", action="store_true",
                        help="read NIK (and name) automatically from the KTP photo via OCR")
    parser.add_argument("--tesseract-cmd", default=None, help="path to tesseract (auto-detected if omitted)")
    parser.add_argument("--tessdata-dir", default=None, help="folder containing *.traineddata (auto-detected)")
    parser.add_argument("--lang", default=None, help="OCR language (auto: ind if available, else eng)")
    args = parser.parse_args()

    nik = args.nik
    name = None
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
    if not nik:
        print("usage: python ktp_register.py <NIK 16-digit> <ktp_image>   or   --ocr <ktp_image>")
        sys.exit(1)

    register_from_ktp(nik, args.ktp_image, name=name)

if __name__ == "__main__":
    main()