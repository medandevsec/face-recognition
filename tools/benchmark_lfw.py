"""Benchmark the face-recognition pipeline (YuNet + AdaFace) on a real face dataset.

Expects an LFW-style layout: <data>/<Identity>/*.jpg.
Computes genuine vs impostor cosine distributions, then FAR/FRR at a given
threshold and an approximate EER.

Example:
    python tools/benchmark_lfw.py --data ./lfw-deepfunneled
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from core.detector import detect_faces
from core.embedder import align_face, embed


def collect(base, max_per_id):
    items = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        pid = os.path.basename(d)
        files = sorted(glob.glob(os.path.join(d, "*.jpg")))[:max_per_id]
        for f in files:
            items.append((pid, f))
    return items


def stats(a):
    return dict(n=len(a), mn=float(a.min()), q1=float(np.percentile(a, 25)),
                med=float(np.median(a)), q3=float(np.percentile(a, 75)),
                mx=float(a.max()), mean=float(a.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with <Identity>/*.jpg")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--max-per-id", type=int, default=8)
    args = ap.parse_args()

    items = collect(args.data, args.max_per_id)
    print(f"images selected: {len(items)} from "
          f"{len(set(p for p, _ in items))} identities\n")

    labels, vecs = [], []
    failed = 0
    for pid, f in items:
        img = cv2.imread(f)
        faces = detect_faces(img)
        if faces is None or len(faces) == 0:
            failed += 1
            print(f"  no face: {f}")
            continue
        try:
            vecs.append(embed(align_face(img, faces[0])))
            labels.append(pid)
        except Exception as exc:
            failed += 1
            print(f"  error {f}: {exc}")
    n = len(labels)
    if n < 4:
        print("too few images to benchmark")
        return

    mat = np.array(vecs)
    sim = mat @ mat.T
    idmap = {pid: idx for idx, pid in enumerate(sorted(set(labels)))}
    lab = np.array([idmap[p] for p in labels])

    genuine, impostor = [], []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            (genuine if lab[i] == lab[j] else impostor).append(s)
    genuine, impostor = np.array(genuine), np.array(impostor)

    gs, is_ = stats(genuine), stats(impostor)
    print(f"embeddings computed: {n}   detection failures: {failed}")
    print(f"\nGENUINE  pairs: {gs['n']:>5}  min={gs['mn']:.3f} q1={gs['q1']:.3f} "
          f"med={gs['med']:.3f} q3={gs['q3']:.3f} max={gs['mx']:.3f}  "
          f"mean={gs['mean']:.3f}")
    print(f"IMPOSTOR pairs: {is_['n']:>5}  min={is_['mn']:.3f} q1={is_['q1']:.3f} "
          f"med={is_['med']:.3f} q3={is_['q3']:.3f} max={is_['mx']:.3f}  "
          f"mean={is_['mean']:.3f}")
    print(f"separation (genuine q1 minus impostor q3): "
          f"{gs['q1'] - is_['q3']:+.3f}")

    def far_frr(t):
        fa = int(np.count_nonzero(impostor >= t))
        fr = int(np.count_nonzero(genuine < t))
        return fa / len(impostor), fr / len(genuine), fa, fr

    def report(t):
        f, r, fa, fr = far_frr(t)
        print(f"  threshold {t:.2f}: FAR={f:.4f} ({fa}/{len(impostor)})  "
              f"FRR={r:.4f} ({fr}/{len(genuine)})")

    print(f"\nevaluation at threshold {args.threshold:.2f}:")
    report(args.threshold)

    lo = min(gs["q1"], is_["q3"])
    hi = max(gs["mn"], is_["mx"])
    best_t, best_d = args.threshold, 1.0
    for t in np.linspace(lo, hi, 2001):
        f, r, _, _ = far_frr(t)
        if abs(f - r) < best_d:
            best_d, best_t = abs(f - r), t
    f, r, _, _ = far_frr(best_t)
    print(f"\nEER (approx): threshold={best_t:.3f} FAR={f:.4f} FRR={r:.4f} "
          f"EER={100 * (f + r) / 2:.2f}%")


if __name__ == "__main__":
    main()