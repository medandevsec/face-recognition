import argparse
import os

import cv2

from core.ktp import dewarp_card
from core.ktp_ocr import extract_fields, extract_text, find_tessdata, find_tesseract


def main():
    parser = argparse.ArgumentParser(description="Read NIK and name from a KTP photo via OCR")
    parser.add_argument("ktp_image", help="path to a KTP photo/scan")
    parser.add_argument("--tesseract-cmd", default=None,
                        help="path to tesseract executable (auto-detected if omitted)")
    parser.add_argument("--tessdata-dir", default=None,
                        help="folder containing *.traineddata (auto-detected if omitted)")
    parser.add_argument("--lang", default=None, help="OCR language (auto: ind if available, else eng)")
    parser.add_argument("--no-dewarp", action="store_true", help="OCR the image as-is, no card dewarp")
    args = parser.parse_args()

    img = cv2.imread(args.ktp_image)
    if img is None:
        print("could not read image")
        return

    tess = args.tesseract_cmd or find_tesseract()
    if not tess:
        print("tesseract not found; pass --tesseract-cmd")
        return
    tessdata = args.tessdata_dir or find_tessdata(tess, args.tessdata_dir)
    if tessdata is None:
        print("warning: no tessdata found next to tesseract; using its default search path")

    source = dewarp_card(img)[0] if not args.no_dewarp else img
    text = extract_text(source, tesseract_cmd=tess, tessdata_dir=tessdata, lang=args.lang)
    fields = extract_fields(text, image=source, tesseract_cmd=tess, tessdata_dir=tessdata, lang=args.lang)
    print(f"NIK : {fields['nik'] or 'not found'}")
    print(f"Name: {fields['name'] or 'not found'}")
    print("--- raw text (trimmed) ---")
    print(text.strip()[:800])


if __name__ == "__main__":
    main()