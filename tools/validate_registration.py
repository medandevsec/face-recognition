import argparse
import csv
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.embedder import (EMBEDDING_DIM, align_face, cosine_similarity, embed,
                           embedding_compatible, load_embeddings)
from core.ktp import dewarp_card, extract_face_region


def main():
    ap = argparse.ArgumentParser(
        description="Simulate recognition matching for every KTP photo in master CSV")
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "master_ktp.csv"))
    ap.add_argument("--ktps", default=os.path.join(ROOT, "ktps"))
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="cosine threshold above which a match is reported")
    args, _ = ap.parse_known_args()

    emb = load_embeddings()
    rows = []
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            rows.append(row)

    correct = 0
    total = 0
    for row in rows:
        nik = row["nik"].strip()
        path = os.path.join(args.ktps, row["filename"].strip())
        img = cv2.imread(path)
        if img is None:
            print(f"[{'NOIMG':5}] {row['filename']}: cannot read image")
            continue
        warped, dewarped = dewarp_card(img)
        face = extract_face_region(warped, dewarped=dewarped)
        if face is None:
            print(f"[{'NOFACE':5}] {row['filename']}: no face detected")
            continue
        q = embed(align_face(warped, face))
        ranked = sorted(
            ((cosine_similarity(q, m["embedding"]), k, m.get("name"))
             for k, m in emb.items()
             if embedding_compatible(m) and len(m["embedding"]) == EMBEDDING_DIM),
            key=lambda t: t[0], reverse=True)
        best_cos, best_nik, best_name = ranked[0]
        own_cos = next(c for c, k, _ in ranked if k == nik)
        total += 1
        ok = best_nik == nik and best_cos >= args.threshold
        correct += int(ok)
        mark = "OK " if ok else "MISS"
        print(f"[{mark}] {row['filename']:14} own={own_cos:.3f} best={best_cos:.3f}"
              f" -> {best_name or best_nik} ({best_nik})")

    print(f"\nmatched correctly: {correct}/{total} (threshold {args.threshold})")


if __name__ == "__main__":
    main()