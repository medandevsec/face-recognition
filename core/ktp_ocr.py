import os
import re
import shutil

import cv2
import numpy as np

NIK_RE = re.compile(r"(?<!\d)(\d{16})(?!\d)")
# label line Nama/Name may be mangled by OCR (e.g. "Nema"), and the value may
# carry punctuation such as "ALEX SIREGAR,S.KOM"
NAME_LABEL_RE = re.compile(r"^\s*n[aeiou]{1,2}m[a-z]{0,2}\s*[:.\-]?\s+(.+)$",
                           re.IGNORECASE)
SKIP_LABELS = {"nik", "nama", "name", "nif", "nema"}
FIELD_TOKENS = {
    "alamat", "provinsi", "kabupaten", "kecamatan", "kel", "desa", "rt/rw", "rt",
    "rw", "agama", "status", "perkawinan", "pekerjaan", "kewarganegaraan", "berlaku",
    "tempat", "tgl", "ttl", "jenis", "kelamin", "gol", "darah", "nik", "nama",
    "name", "masa", "golongan", "keadaan",
}


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


def _apply_tessdata_env(tessdata_dir):
    if tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = tessdata_dir


def extract_text(image, tesseract_cmd=None, tessdata_dir=None, lang=None, psm=6):
    import pytesseract

    if tesseract_cmd is None:
        tesseract_cmd = find_tesseract()
    if not tesseract_cmd:
        raise RuntimeError("Tesseract not found; install it and pass --tesseract-cmd")
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    _apply_tessdata_env(tessdata_dir)
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


def _confusion_digits(text):
    """Best-effort digit reconstruction for OCR letter/digit confusions
    (b->6, O->0, l->1, ...). Unknown characters are dropped."""
    table = {"o": "0", "O": "0", "b": "6", "g": "9", "q": "9", "l": "1", "I": "1",
             "i": "1", "|": "1", "s": "5", "S": "5", "z": "2", "Z": "2", "B": "8",
             "a": "4", "e": "2", "L": "1"}
    out = []
    for ch in text:
        if ch.isdigit():
            out.append(ch)
        else:
            mapped = table.get(ch)
            if mapped:
                out.append(mapped)
    return "".join(out)


def _birth_hints(text):
    """Birth-date substrings expected inside a NIK (positions 6..12), derived
    from a date like 06-07-1986 on the same card (male day->+40 accepted)."""
    m = re.search(r"\b(\d{2})[-/.](\d{2})[-/.](\d{2,4})\b", text or "")
    if not m:
        return set()
    dd, mm, yy = m.group(1), m.group(2), m.group(3)[-2:]
    hints = {dd + mm + yy}
    try:
        if 1 <= int(dd) <= 31:
            hints.add(f"{int(dd) + 40:02d}{mm}{yy}")
    except ValueError:
        pass
    return hints


def _pick_nik(candidates, hints=None):
    """Choose the most plausible 16-digit NIK. Prefer: (1) exactly 16 digits,
    (2) a birth-date match from the same card, (3) digit reads of different
    scales agreeing, (4) a '12' prefix (North Sumatra)."""
    from collections import Counter

    hints = hints or set()
    counts = Counter(candidates)
    best, best_score = None, None
    for c in sorted(set(candidates)):
        score = counts[c]
        if len(c) != 16:
            score -= 1000
        if c.startswith("12"):
            score += 100
        if any(c[6:12] == h for h in hints):
            score += 500
        if best_score is None or score > best_score:
            best, best_score = c, score
    return best if best and len(best) == 16 else None


def ocr_nik_from_image(image, tesseract_cmd=None, tessdata_dir=None, lang=None):
    """Find the NIK label line with tesseract word boxes, re-OCR the trailing
    number region with a digits-only whitelist so 6/8 and other digit
    confusions resolve correctly. Falls back to confusion-mapping when the
    whitelist reads are short."""
    import pytesseract
    from pytesseract import Output

    if tesseract_cmd is None:
        tesseract_cmd = find_tesseract()
    if not tesseract_cmd:
        return extract_nik(extract_text(image, lang=lang))
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    _apply_tessdata_env(tessdata_dir)
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
    sep = " ".join(w["text"] for w in words)
    if not words:
        return extract_nik(extract_text(image, lang=lang))

    label_idx = next((i for i, w in enumerate(words)
                      if w["text"].lower().lstrip(":").strip() in SKIP_LABELS), None)
    if label_idx is None:
        return extract_nik(sep)

    label = words[label_idx]
    ymid = label["t"] + label["h"] / 2
    tol = max(label["h"] * 0.6, 6)
    line_right = [w for i, w in enumerate(words)
                  if i != label_idx
                  and abs((w["t"] + w["h"] / 2) - ymid) <= tol
                  and w["b"] >= label["b"] - 20]
    if not line_right:
        return extract_nik(sep)

    # crop from just past the label to the end of the number region, so digits
    # tesseract failed to segment (silent gaps) are still captured
    x0 = max(0, label["b"] + label["w"] + 10)
    y0 = max(0, min(w["t"] for w in line_right) - 10)
    x1 = min(prep.shape[1], max(w["b"] + w["w"] for w in line_right) + 20)
    y1 = min(prep.shape[0], max(w["t"] + w["h"] for w in line_right) + 10)
    if x1 - x0 < 30 or y1 - y0 < 10:
        hints = _birth_hints(sep)
        return _pick_nik([_confusion_digits(" ".join(w["text"] for w in line_right))],
                         hints) or extract_nik(sep)

    # re-OCR the region from the ORIGINAL (unfiltered) image at several
    # magnifications: the bilateral-filtered plus 2x crop of preprocess()
    # sometimes merges the leading digits and drops them
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ox0, oy0 = max(0, x0 // 2 - 10), max(0, y0 // 2 - 10)
    ox1, oy1 = min(gray.shape[1], x1 // 2 + 10), min(gray.shape[0], y1 // 2 + 10)
    if ox1 - ox0 < 30 or oy1 - oy0 < 10:
        return extract_nik(sep)
    body = gray[oy0:oy1, ox0:ox1]

    hints = _birth_hints(sep)
    candidates = set()
    for mag in (4, 6, 2):
        up = cv2.resize(body, None, fx=mag, fy=mag, interpolation=cv2.INTER_CUBIC)
        for view in (up, cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]):
            for psm in ("13", "7", "6"):
                try:
                    res = pytesseract.image_to_string(
                        view, config=f"{base} --psm {psm} -c tessedit_char_whitelist=0123456789")
                except pytesseract.TesseractError:
                    continue
                digits = _clean_digits(res)
                if not digits:
                    continue
                candidates.add(digits)
                # early return: confident 16-digit read agreeing with the card date
                if len(digits) == 16 and any(digits[6:12] == h for h in hints):
                    return digits

    fixed = _confusion_digits(" ".join(w["text"] for w in line_right))
    if fixed:
        candidates.add(fixed)

    return _pick_nik(candidates, hints) or extract_nik(sep)


def _valid_name_value(value):
    value = (value or "").strip()
    value = re.sub(r"[^A-Za-z0-9]+$", "", value).strip()
    if not 3 <= len(value) <= 40:
        return None
    if not all(c.isalpha() or c in " '.,-'+" for c in value):
        return None
    words = [w for w in value.replace(".", " ").replace(",", " ").replace("-", " ").split() if w]
    multi = [w for w in words if len(w) >= 2]
    if not words or len(multi) < 2:
        return None
    if not any(w.isalpha() for w in words):
        return None
    if sum(1 for c in value if c.isalpha()) < 3:
        return None
    if sum(1 for c in value if c.islower()) > 2:
        return None
    if value.lower().lstrip(":").strip() in SKIP_LABELS:
        return None
    return value


def _looks_like_field(line):
    head = line.split()[0].lower().rstrip(":")
    return head in FIELD_TOKENS


def extract_name(text):
    for line in text.splitlines():
        match = NAME_LABEL_RE.match(line)
        if match:
            value = _valid_name_value(match.group(1))
            if value:
                return value
    for line in text.splitlines():
        value = _valid_name_value(line)
        if value and not _looks_like_field(line):
            return value
    return None


def extract_fields(text, image=None, tesseract_cmd=None, tessdata_dir=None, lang=None):
    nik = extract_nik(text)
    if nik is None and image is not None:
        nik = ocr_nik_from_image(image, tesseract_cmd=tesseract_cmd,
                                 tessdata_dir=tessdata_dir, lang=lang) or nik
    return {"nik": nik, "name": extract_name(text)}


def read_ktp_fields(image, tesseract_cmd=None, tessdata_dir=None, lang=None,
                    no_dewarp=False):
    """Best-effort NIK/name extraction: tries the (dewarped) card first, then
    falls back to the raw photo and the 90/180/270 rotations until both fields
    are found."""
    sources = []
    if not no_dewarp:
        try:
            from core.ktp import dewarp_card
            sources.append(dewarp_card(image)[0])
        except Exception:
            pass
    sources.append(image)

    best, best_score = {"nik": None, "name": None}, -1
    for src in sources:
        for rot in range(4):
            probe = np.rot90(src, rot) if rot else src
            try:
                text = extract_text(probe, tesseract_cmd=tesseract_cmd,
                                    tessdata_dir=tessdata_dir, lang=lang)
            except Exception:
                continue
            fields = extract_fields(text, image=probe, tesseract_cmd=tesseract_cmd,
                                    tessdata_dir=tessdata_dir, lang=lang)
            score = bool(fields["nik"]) + bool(fields["name"])
            if score == 2:
                return fields
            if score > best_score:
                best_score, best = score, fields
    return best