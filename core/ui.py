import cv2

CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)

def padded_box(x, y, w, h, height, width):
    pad_x, pad_top, pad_bottom = int(w * 0.12), int(h * 0.30), int(h * 0.12)
    bx, by = max(0, x - pad_x), max(0, y - pad_top)
    bw = min(width - bx, w + pad_x * 2)
    bh = min(height - by, h + pad_top + pad_bottom)
    return bx, by, bw, bh

def draw_box(frame, x, y, w, h, name, confidence, info=None):
    color = CYAN if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    lines = [name.upper()]
    if name != "Unknown":
        if info:
            lines.extend(info)
        lines.append(f"{confidence:.0f}% MATCH")

    fonts = []
    max_w = 0
    for i, line in enumerate(lines):
        scale = 1.5 if i == 0 else 0.7
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, scale,
                                      3 if i == 0 else 1)
        fonts.append((line, scale, tw, th))
        max_w = max(max_w, tw)

    row_h = 30
    panel_w = max_w + 24
    panel_h = len(lines) * row_h + 16
    px = x - 8
    py = y - panel_h - 10
    if py < 4:
        py = y + h + 10

    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), color, 1)

    ty = py + row_h
    for i, (line, scale, tw, th) in enumerate(fonts):
        if i == 0:
            text_color = color
        elif line.endswith("MATCH"):
            text_color = color
        else:
            text_color = (255, 255, 255)
        cv2.putText(frame, line, (px + 12, ty), cv2.FONT_HERSHEY_DUPLEX, scale,
                    text_color, 3 if i == 0 else 1)
        ty += row_h

def draw_corners(frame, x, y, w, h, size=28):
    color = YELLOW
    t = 3
    for px, py, dx, dy in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * size, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy * size), color, t)

# fuller face mesh: jawline outline + temples + brows + eyes + nose + mouth
MESH_POINTS = [
    (0.10, 0.45), (0.13, 0.60), (0.20, 0.75), (0.32, 0.88), (0.50, 0.95),
    (0.68, 0.88), (0.80, 0.75), (0.87, 0.60), (0.90, 0.45),          # 0-8 jawline
    (0.12, 0.22), (0.88, 0.22),                                     # 9-10 temples
    (0.28, 0.30), (0.40, 0.24), (0.60, 0.24), (0.72, 0.30),         # 11-14 brows
    (0.30, 0.42), (0.42, 0.40), (0.58, 0.40), (0.70, 0.42),         # 15-18 eyes
    (0.50, 0.46), (0.44, 0.60), (0.56, 0.60), (0.50, 0.63),         # 19-22 nose
    (0.34, 0.73), (0.50, 0.70), (0.66, 0.73),                       # 23-25 mouth top
    (0.40, 0.82), (0.50, 0.84), (0.60, 0.82),                       # 26-28 mouth bottom
]

MESH_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (9, 0), (10, 8), (9, 11), (10, 14),
    (11, 12), (12, 13), (13, 14),
    (11, 15), (12, 16), (13, 17), (14, 18),
    (15, 16), (16, 19), (17, 19), (17, 18),
    (19, 20), (19, 21), (20, 22), (21, 22),
    (16, 23), (17, 25), (22, 24),
    (23, 24), (24, 25), (23, 26), (25, 28), (26, 27), (27, 28),
    (2, 23), (6, 25), (3, 26), (5, 28),
]

def draw_mesh(frame, x, y, w, h):
    overlay = frame.copy()
    pts = [(int(x + rx * w), int(y + ry * h)) for rx, ry in MESH_POINTS]
    for a, b in MESH_LINES:
        cv2.line(overlay, pts[a], pts[b], CYAN, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(overlay, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

def wrap_text_cv(text, font, scale, thickness, max_w):
    words = str(text).split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for word in words[1:]:
        test = cur + " " + word
        (tw, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if tw <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def draw_side_panel(frame, pairs, title="IDENTITAS", color=CYAN, width=420):
    """Right-edge panel with label/value rows (e.g. NIK / NAMA / ALAMAT)."""
    h, w = frame.shape[:2]
    if w <= width + 24:
        width = int(w * 0.55)

    font = cv2.FONT_HERSHEY_DUPLEX
    label_scale, val_scale = 0.55, 0.7
    label_th, val_th = 1, 2
    pad = 14
    title_h = 30
    row_h = 26
    label_gap = 10
    val_max = width - pad * 2 - 60 - label_gap - 6

    layout = []
    rows = 0
    for label, value in pairs:
        if value is None:
            continue
        lines = wrap_text_cv(value, font, val_scale, val_th, val_max)
        layout.append((label, lines))
        rows += len(lines)
    if not layout:
        return 0

    panel_h = title_h + rows * row_h + (len(layout) - 1) * 6 + pad
    x0 = w - width - 12
    y0 = 12

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + panel_h), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + panel_h), color, 1)
    cv2.putText(frame, title, (x0 + pad, y0 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)

    ty = y0 + title_h + row_h - 4
    for label, lines in layout:
        (lw, lh), _ = cv2.getTextSize(label, font, label_scale, label_th)
        cv2.putText(frame, label, (x0 + pad, ty), font, label_scale, color, label_th)
        lx = x0 + pad + lw + label_gap
        for line in lines:
            cv2.putText(frame, line, (lx, ty), font, val_scale, (255, 255, 255), val_th)
            ty += row_h
        ty += 6
    return 1


def draw_hud(frame, fps, face_count):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 55), (220, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1)
    cv2.putText(frame, f"FACES: {face_count}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1)
    cv2.rectangle(frame, (w - 100, 0), (w, 35), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, "LIVE", (w - 85, 24), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)