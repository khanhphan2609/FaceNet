import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

# IMPORT CONFIG
from config import DATA_DIR

# ===== Device =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Factory using device: {device}")

# ===== PATH =====
RAW_DIR = DATA_DIR["original"]
CLEAN_DIR = DATA_DIR["faces"]

# ===== Detector =====
detector = MTCNN(
    image_size=160,
    margin=30,
    min_face_size=40,
    thresholds=[0.7, 0.8, 0.8],
    device=device,
    post_process=False
)

# ==========================================
# ALIGN FACE
# ==========================================
def align_face(img, landmarks):
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    eye_center = (
        int((left_eye[0] + right_eye[0]) // 2),
        int((left_eye[1] + right_eye[1]) // 2)
    )

    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    h, w = img.shape[:2]

    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

# ==========================================
# ENHANCE LIGHT
# ==========================================
def enhance_illumination(face_img):
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

# ==========================================
# QUALITY CHECK
# ==========================================
def check_quality(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > 40

# ==========================================
# PROCESS IMAGE
# ==========================================
def process_single_image(img_rgb):
    boxes, probs, landmarks = detector.detect(img_rgb, landmarks=True)

    if boxes is None:
        return None

    idx = probs.argmax()
    if probs[idx] < 0.9:
        return None

    # ALIGN
    rotated_img = align_face(img_rgb, landmarks[idx])

    # DETECT AGAIN (more accurate)
    box_new, _ = detector.detect(rotated_img)
    if box_new is None:
        return None

    x1, y1, x2, y2 = box_new[0].astype(int)
    h, w = rotated_img.shape[:2]

    face = rotated_img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    if face.size == 0:
        return None

    face = cv2.resize(face, (160, 160))

    # QUALITY FILTER
    if not check_quality(face):
        return None

    # LIGHT NORMALIZATION
    face = enhance_illumination(face)

    return face

# ==========================================
# MAIN FACTORY
# ==========================================
def start_factory(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    people = [
        d for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ]

    for person in people:
        in_dir = os.path.join(input_path, person)
        out_dir = os.path.join(output_path, person)
        os.makedirs(out_dir, exist_ok=True)

        images = [
            f for f in os.listdir(in_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        success = 0

        print(f"\n📂 Processing: {person}")

        for img_name in tqdm(images):
            save_path = os.path.join(out_dir, img_name)

            if os.path.exists(save_path):
                continue

            img = cv2.imread(os.path.join(in_dir, img_name))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                face = process_single_image(img_rgb)

                if face is not None:
                    cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                    success += 1

            except Exception:
                continue

        print(f"✅ {person}: {success}/{len(images)} ảnh đạt chuẩn")

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(RAW_DIR):
        print("❌ RAW dataset not found")
        exit()

    start_factory(RAW_DIR, CLEAN_DIR)