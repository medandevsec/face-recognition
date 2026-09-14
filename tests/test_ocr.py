import os

import cv2
import numpy as np
import pytest

from core.ktp_ocr import (extract_fields, extract_nik, extract_text, find_tessdata,
                          find_tesseract, read_ktp_fields)

try:
    import pytesseract  # noqa: F401
    _has_pytesseract = True
except ImportError:
    _has_pytesseract = False

TESS = os.environ.get("KTP_OCR_TESSERACT") or find_tesseract()
TESSDATA = find_tessdata(TESS) if TESS else None
has_tesseract = TESS is not None and _has_pytesseract

pytestmark = pytest.mark.skipif(not has_tesseract, reason="Tesseract not installed")


def build_text_image():
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1300, 300), "white")
    d = ImageDraw.Draw(img)
    font = None
    for candidate in (r"C:\Windows\Fonts\arialbd.ttf",
                      r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.isfile(candidate):
            try:
                font = ImageFont.truetype(candidate, 64)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
    d.text((40, 30), "NIK : 1234567890123456", fill="black", font=font)
    d.text((40, 150), "NAMA : BUDI SANTOSO", fill="black", font=font)
    return np.array(img)


def test_extract_nik_regex():
    assert extract_nik("NIK : 1234567890123456") == "1234567890123456"
    assert extract_nik("http://x/1234567890123456z") == "1234567890123456"
    assert extract_nik("1234567890123456") == "1234567890123456"
    assert extract_nik("12345678901234567") is None
    assert extract_nik("12 34567890123456 7") is None


def test_ocr_reads_nik_and_name():
    image = build_text_image()
    text = extract_text(image, tesseract_cmd=TESS, tessdata_dir=TESSDATA)
    fields = extract_fields(text, image=image, tesseract_cmd=TESS, tessdata_dir=TESSDATA)
    assert fields["nik"] == "1234567890123456"
    assert fields["name"] == "BUDI SANTOSO"


def test_ocr_without_whitelist_fallback():
    image = build_text_image()
    fields = extract_fields("NIK : 1234567890123456", image=image)
    assert fields["nik"] == "1234567890123456"


def test_ocr_real_ktp_photo_reads_identity():
    """A real (crumpled/rotated) KTP photo: NIK and name must auto-extract from
    the image alone, reconstructing digits OCR mangles (b->6 etc.)."""
    path = os.path.join(os.path.dirname(__file__), "..", "ktps", "KTPALEX.jpg")
    img = cv2.imread(path)
    assert img is not None
    fields = read_ktp_fields(img, tesseract_cmd=TESS, tessdata_dir=TESSDATA,
                             no_dewarp=True)
    assert fields["name"] and "ALEX" in fields["name"].upper()
    assert fields["nik"]
    assert len(fields["nik"]) == 16
    assert fields["nik"].startswith("12")
    assert fields["nik"].endswith("0003")