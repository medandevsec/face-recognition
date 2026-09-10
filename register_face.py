import cv2
import re
import sys
import os
from core.detector import detect_faces
from core.embedder import align_face, embed, mean_embedding, load_embeddings, save_embeddings, FACES_DIR

def sanitize_name(name):
    name = name.strip().replace("/", "_").replace("\\", "_").replace("\x00", "_")
    if not name or ".." in name:
        return None
    return name[:80]

def register(name, image_path):
    name = sanitize_name(name)
    if name is None:
        print("invalid name: avoid path separators or '..'")
        return False
    img = cv2.imread(image_path)
    if img is None:
        print("could not read image")
        return False
    faces = detect_faces(img)
    if faces is None or len(faces) == 0:
        print("no face found in image, try another photo")
        return False

    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    count = len(os.listdir(person_dir))
    aligned = align_face(img, faces[0])
    cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), aligned)
    print(f"saved photo for {name}, updating embeddings...")

    embedding = mean_embedding(person_dir)
    if embedding is None:
        print("could not compute embedding")
        return False

    embeddings = load_embeddings()
    embeddings[name] = {"embedding": embedding, "samples": len(os.listdir(person_dir))}
    save_embeddings(embeddings)
    print(f"registered {name} ({len(os.listdir(person_dir))} photo(s))")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python register_face.py <name> <image_path>")
        sys.exit(1)
    register(sys.argv[1], sys.argv[2])