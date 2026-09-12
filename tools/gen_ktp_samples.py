import argparse
import csv
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CARD_W, CARD_H = 1000, 630  # ID-1 ratio (85.60 x 53.98 mm)

FACE_URLS = {
    "lena": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "messi": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
}

FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
FONT_REG = r"C:\Windows\Fonts\arial.ttf"


def sample_faces_dir():
    d = os.path.join(ROOT, "data", "sample_faces")
    os.makedirs(d, exist_ok=True)
    return d


def ensure_sample_faces():
    import urllib.request

    d = sample_faces_dir()
    for name, url in FACE_URLS.items():
        target = os.path.join(d, name + ".jpg")
        if not os.path.isfile(target) or cv2.imread(target) is None:
            urllib.request.urlretrieve(url, target)
            if cv2.imread(target) is None:
                raise RuntimeError(f"could not download sample face {url}")
    return d


def load_font(path, size, fallback=None):
    for candidate in (path, fallback) if fallback else (path,):
        if candidate and os.path.isfile(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_ktp(row, out_path, face_path):
    bg = np.full((CARD_H + 80, CARD_W + 80, 3), 105, np.uint8)
    rng = np.random.default_rng(42)
    noise = rng.integers(80, 130, size=(CARD_H + 80, CARD_W + 80, 1), dtype=np.uint8)
    bg = (bg.astype(np.int16) * 0.85 + noise * 0.15).astype(np.uint8)

    card = Image.new("RGB", (CARD_W, CARD_H), "white")
    d = ImageDraw.Draw(card)

    d.rectangle([0, 0, CARD_W, 70], fill=(213, 42, 36))
    d.rectangle([0, 70, CARD_W, 76], fill=(190, 30, 24))
    for i in range(2, 21, 2):
        x0 = CARD_W - i * 22
        d.rectangle([x0, 0, x0 + 14, 70], fill=(255, 255, 255))

    font_sm = load_font(FONT_REG, 22)
    font_sm_b = load_font(FONT_BOLD, 22)
    font_big = load_font(FONT_BOLD, 30)
    field_font = load_font(FONT_REG, 26)

    d.text((24, 10), "PROVINSI SUMATERA UTARA", font=font_sm_b, fill="white")
    d.text((24, 38), "KABUPATEN CONTOH SELATAN", font=font_sm, fill="white")

    y = 100
    nik = row["nik"]
    fields = [
        ("NIK", nik, 18),
        ("NAMA", row["nama"].title(), 0),
        ("TEMPAT/TGL LAHIR", f"{row['tempat_lahir']}, {row['tanggal_lahir']}", 0),
        ("JENIS KELAMIN", row["jenis_kelamin"].title(), 0),
        ("ALAMAT", row["alamat"].upper(), 0),
    ]
    d.text((30, y), "NIK", font=font_sm_b, fill=(40, 40, 40))
    d.text((190, y), ": " + nik, font=field_font, fill=(0, 0, 0))
    y += 44
    for label, value, _offset in fields[1:]:
        d.text((30, y), label, font=font_sm_b, fill=(40, 40, 40))
        if label == "ALAMAT":
            lines = wrap_text(d, ": " + value, field_font, 560)
            for li, line in enumerate(lines[:2]):
                d.text((190, y + li * 34), line, font=field_font, fill=(0, 0, 0))
            y += 44 + 34 * (min(len(lines), 2) - 1)
        else:
            d.text((190, y), ": " + value, font=field_font, fill=(0, 0, 0))
            y += 44

    face = cv2.imread(face_path)
    if face is None:
        raise RuntimeError(f"cannot read face sample {face_path}")
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    fw, fh = 300, 350
    face = cv2.resize(face, (fw, fh))
    card.paste(Image.fromarray(face), (CARD_W - fw - 26, 120))

    bg[CARD_H // 2 - CARD_H // 2 + 40:40 + CARD_H, 40:40 + CARD_W] = np.array(card)
    cv2.imwrite(out_path, cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))


def synthetic_rows(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("keterangan", "").strip().lower() != "contoh":
                continue
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "master_ktp.csv"))
    ap.add_argument("--ktps", default=os.path.join(ROOT, "ktps"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    faces = ensure_sample_faces()
    face_list = [os.path.join(faces, "lena.jpg"), os.path.join(faces, "messi.jpg")]
    os.makedirs(args.ktps, exist_ok=True)

    made = skipped = 0
    for i, row in enumerate(synthetic_rows(args.csv)):
        out = os.path.join(args.ktps, row["filename"])
        if not args.force and os.path.isfile(out):
            skipped += 1
            continue
        render_ktp(row, out, face_list[i % len(face_list)])
        made += 1
        print(f"generated {row['filename']} (NIK {row['nik']}, face {os.path.basename(face_list[i % 2])})")
    print(f"done: {made} generated, {skipped} skipped/found")


if __name__ == "__main__":
    main()