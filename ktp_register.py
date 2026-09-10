import cv2
import os
import re
import sys
from core.embedder import align_face, mean_embedding, load_embeddings, save_embeddings, FACES_DIR
from core.ktp import dewarp_card, extract_face_region

def mask_nik(nik):
    return f"{nik[:2]}xx-xxxx-{nik[-4:]}"

def register_from_ktp(nik, image_path):
    if not re.fullmatch(r"\d{16}", nik):
        print("invalid NIK: must be exactly 16 digits, e.g. 3273012501900001")
        return False
    img = cv2.imread(image_path)
    if img is None:
        print("could not read image")
        return False

    warped, dewarped = dewarp_card(img)
    face = extract_face_region(warped)
    if face is None:
        print("no face found in KTP image, try a clearer scan/photo")
        return False

    aligned = align_face(warped, face)
    person_dir = os.path.join(FACES_DIR, nik)
    os.makedirs(person_dir, exist_ok=True)
    count = len(os.listdir(person_dir))
    cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), aligned)

    embedding = mean_embedding(person_dir)
    if embedding is None:
        print("could not compute embedding")
        return False

    embeddings = load_embeddings()
    embeddings[nik] = {"embedding": embedding, "samples": len(os.listdir(person_dir))}
    save_embeddings(embeddings)
    print(f"registered NIK {mask_nik(nik)} from KTP ({len(os.listdir(person_dir))} sample(s), card dewarped: {dewarped})")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python ktp_register.py <NIK 16-digit> <ktp_image>")
        sys.exit(1)
    register_from_ktp(sys.argv[1], sys.argv[2])