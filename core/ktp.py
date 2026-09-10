import cv2
import numpy as np
from core.detector import detect_faces

WARP_W, WARP_H = 1000, 630  # ID-1 ratio (85.60 x 53.98 mm)

def _order_points(pts):
    pts = np.array(pts, dtype="float32").reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def find_card_corners(image):
    h, w = image.shape[:2]
    if max(h, w) > 1600:
        scale = 1600.0 / max(h, w)
        work = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0
        work = image
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return _order_points(approx) / scale
    return None

def dewarp_card(image):
    corners = find_card_corners(image)
    if corners is None:
        return image, False
    dst = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, matrix, (WARP_W, WARP_H))
    return warped, True

def extract_face_region(warped):
    faces = detect_faces(warped)
    if faces is None or len(faces) == 0:
        return None
    h, w = warped.shape[:2]
    best_face, best_score = None, -1.0
    for face in faces:
        x, y, fw, fh = [int(v) for v in face[:4]]
        cx = x + fw / 2.0
        cy = y + fh / 2.0
        score = fw * fh * 1e-6
        if cx > w * 0.52:
            score += 1.0
        if h * 0.12 < cy < h * 0.88:
            score += 0.5
        if score > best_score:
            best_score = score
            best_face = face
    return best_face