import cv2
import sys
import os
import numpy as np
from core.detector import detect_faces
from core.embedder import align_face, embed, load_embeddings, save_embeddings, FACES_DIR

def mean_embedding(person_dir):
    vectors = []
    for fname in sorted(os.listdir(person_dir)):
        crop = cv2.imread(os.path.join(person_dir, fname))
        if crop is None:
            continue
        vectors.append(embed(crop))
    if not vectors:
        return None
    return np.mean(np.array(vectors), axis=0)

def register(name, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("could not read image")
        return
    faces = detect_faces(img)
    if faces is None or len(faces) == 0:
        print("no face found in image, try another photo")
        return

    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    count = len(os.listdir(person_dir))
    aligned = align_face(img, faces[0])
    cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), aligned)
    print(f"saved photo for {name}, updating embeddings...")

    embedding = mean_embedding(person_dir)
    if embedding is None:
        print("could not compute embedding")
        return

    embeddings = load_embeddings()
    embeddings[name] = {"embedding": embedding, "samples": len(os.listdir(person_dir))}
    save_embeddings(embeddings)
    print(f"registered {name} ({len(os.listdir(person_dir))} photo(s))")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python register_face.py <name> <image_path>")
        sys.exit(1)
    register(sys.argv[1], sys.argv[2])