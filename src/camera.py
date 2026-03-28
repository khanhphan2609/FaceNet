import cv2
import mediapipe as mp
import os
import time
import unicodedata
import re
import sys

# Import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR, CAMERA

# ==========================================
# 1. Chuẩn hóa tên (Xóa dấu, thay khoảng trắng bằng gạch dưới)
# ==========================================
def sanitize_name(name):
    nfkd_form = unicodedata.normalize('NFKD', name)
    name_no_accent = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    name_clean = name_no_accent.lower().replace(" ", "_")
    name_clean = re.sub(r'[^a-z0-9_]', '', name_clean)
    return name_clean

# ==========================================
# 2. Khởi tạo MediaPipe
# ==========================================
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.4
)

# ==========================================
# 3. Thu thập ảnh khuôn mặt
# ==========================================
def collect_face_data(raw_name, base_dir=None, max_samples=None):
    # Lấy từ config nếu không truyền vào
    base_dir    = base_dir    or DATA_DIR["original"]
    max_samples = max_samples or CAMERA["max_samples"]

    folder_name = sanitize_name(raw_name)
    save_path   = os.path.join(base_dir, folder_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"✅ Đã tạo thư mục: {save_path}")

    cap   = cv2.VideoCapture(0)
    count = 0

    print(f"🚀 Thu thập dữ liệu cho: {raw_name} (Thư mục: {folder_name})")
    print("Mẹo: Xoay nhẹ đầu, thay đổi biểu cảm. Nhấn 'q' để dừng.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image   = cv2.flip(image, 1)
        h, w, _ = image.shape
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ix = max(0, int(bbox.xmin * w))
                iy = max(0, int(bbox.ymin * h))
                iw = int(bbox.width * w)
                ih = int(bbox.height * h)

                x2 = min(w, ix + iw)
                y2 = min(h, iy + ih)

                padding   = 30
                face_crop = image[
                    max(0, iy - padding):min(h, y2 + padding),
                    max(0, ix - padding):min(w, x2 + padding)
                ]

                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (160, 160))

                    count     += 1
                    file_name  = os.path.join(save_path, f"{folder_name}_{count}.jpg")
                    cv2.imwrite(file_name, face_resized)

                    cv2.rectangle(image, (ix, iy), (ix + iw, iy + ih), (255, 0, 0), 2)
                    cv2.putText(
                        image, f"Captured: {count}/{max_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
                    )

        cv2.imshow('Face Data Collector', image)

        if count >= max_samples or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🎉 Hoàn thành! Đã thu thập {count} ảnh cho '{raw_name}'.")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    name_input = input("Nhập tên: ").strip()

    if not name_input:
        print("❌ Tên không được để trống!")
        exit()

    while True:
        cmd = input("👉 Gõ 'start' để bắt đầu thu thập (hoặc 'q' để thoát): ").strip().lower()

        if cmd == "start":
            collect_face_data(name_input)
            break
        elif cmd == "q":
            print("👋 Đã thoát.")
            break
        else:
            print("⚠️ Lệnh không hợp lệ, hãy nhập 'start' hoặc 'q'")