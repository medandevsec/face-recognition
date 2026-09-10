import os
import re
import shutil

import cv2

NIK_RE = re.compile(r"(?<!\d)(\d{16})(?!\d)")
NAME_LABEL_RE = re.compile(r"^\s*(?:nama|name)\s*:?\s*(.+)$", re.IGNORECASE)
SKIP_LABELS = {"nik", "nama", "name", "nif"}


def find_tesseract():
    path = shutil.which("tesseract")
    if path:
        return path
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"/usr/bin/tesseract",
        r"/opt/homebrew/bin/tesseract",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def find_tessdata(tesseract_path, override=None):
    if override and os.path.isdir(override):
        return override
    env = os.environ.get("TESSDATA_PREFIX")
    if env and os.path.isfile(os.path.join(env, "eng.traineddata")):
        return env
    base = os.path.dirname(tesseract_path)
    for root in (base, os.path.join(base, "tessdata")):
        if os.path.isfile(os.path.join(root, "eng.traineddata")):
            return root
    return None


def available_langs(tessdata_dir):
    if not tessdata_dir:
        return set()
    return {
        name[: -len(".traineddata")].lower()
        for name in os.listdir(tessdata_dir)
        if name.endswith(".traineddata")
    }


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray


def _apply_tessdata_env(tessdata_dir, tesseract_cmd):
    if tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = tessdata_dir


def extract_text(image, tesseract_cmd=None, tessdata_dir=None, lang=None, psm=6):
    import pytesseract

    if tesseract_cmd is None:
        tesseract_cmd = find_tesseract()
    if not tesseract_cmd:
        raise RuntimeError("Tesseract not found; install it and pass --tesseract-cmd")
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    _apply_tessdata_env(tessdata_dir, tesseract_cmd)
    if lang is None:
        langs = available_langs(tessdata_dir or find_tessdata(tesseract_cmd))
        lang = "ind" if "ind" in langs else "eng"
    config = f"--psm {psm} -l {lang}"
    return pytesseract.image_to_string(preprocess(image), config=config)


def _clean_digits(text):
    return "".join(c for c in text if c.isdigit())


def extract_nik(text):
    match = NIK_RE.search(text)
    return match.group(1) if match else None


def ocr_nik_from_image(image, tesseract_cmd=None, tessdata_dir=None, lang=None):
    """Find the NIK label line with tesseract word boxes, re-OCR it with a
    digits-only whitelist so 6/8 and other digit confusions resolve correctly."""
    import pytesseract
    from pytesseract import Output

    if tesseract_cmd is None:
        tesseract_cmd = find_tesseract()
    if not tesseract_cmd:
        return extract_nik(extract_text(image, lang=lang))
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    _apply_tessdata_env(tessdata_dir, tesseract_cmd)
    if lang is None:
        langs = available_langs(tessdata_dir or find_tessdata(tesseract_cmd))
        lang = "ind" if "ind" in langs else "eng"
    base = f"-l {lang}"

    prep = preprocess(image)
    data = pytesseract.image_to_data(prep, config=f"{base} --psm 6", output_type=Output.DICT)
    words = []
    for i, txt in enumerate(data["text"]):
        if not txt.strip():
            continue
        words.append({"text": txt.strip(), "b": data["left"][i], "t": data["top"][i],
                      "w": data["width"][i], "h": data["height"][i],
                      "line": data["line_num"][i], "block": data["block_num"][i]})
    if not words:
        return None
    label_idx = next((i for i, w in enumerate(words)
                      if w["text"].lower().lstrip(":").strip() in SKIP_LABELS), None)
    if label_idx is None:
        return extract_nik(" ".join(w["text"] for w in words))

    picked = []
    for w in words[label_idx + 1:]:
        digits = _clean_digits(w["text"])
        if digits and len(digits) >= 12:
            picked.append(w)
        elif any(c.isdigit() for c in w["text"]):
            picked.append(w)
        elif len(picked) == 0:
            continue
        else:
            break
    if not picked:
        return extract_nik(" ".join(w["text"] for w in words))

    x0 = min(w["b"] for w in picked) - 10
    y0 = min(w["t"] for w in picked) - 10
    x1 = max(w["b"] + w["w"] for w in picked) + 10
    y1 = max(w["t"] + w["h"] for w in picked) + 10
    x0, y0 = max(0, x0), max(0, y0)
    crop = prep[y0:y1, x0:x1]
    res = pytesseract.image_to_string(
        crop, config=f"{base} --psm 7 -c tessedit_char_whitelist=0123456789")
    return extract_nik(res)


def extract_name(text):
    for line in text.splitlines():
        match = NAME_LABEL_RE.match(line)
        if match:
            value = match.group(1).strip()
            if 3 <= len(value) <= 40 and value.replace(" ", "").isalpha():
                return value
    for line in text.splitlines():
        line = line.strip()
        if not 3 <= len(line) <= 40:
            continue
        if not all(c.isalpha() or c in " '.-" for c in line):
            continue
        words = line.split()
        if any(c.islower() for c in line):
            continue
        if any(len(w) < 2 for w in words):
            continue
        if line.lower().lstrip(":").strip() in SKIP_LABELS:
            continue
        return line
    return None


def extract_fields(text, image=None, tesseract_cmd=None, tessdata_dir=None, lang=None):
    nik = extract_nik(text)
    if nik is None and image is not None:
        nik = ocr_nik_from_image(image, tesseract_cmd=tesseract_cmd,
                                 tessdata_dir=tessdata_dir, lang=lang) or nik
    return {"nik": nik, "name": extract_name(text)}