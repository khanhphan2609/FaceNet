import cv2
import os
import sys
import csv
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_DIR

# ==========================================
# CONFIG
# ==========================================
VIDEO_PATH  = "video.mp4"                  # Đường dẫn video đầu vào
OUTPUT_DIR  = DATA_DIR["original"]         # Lưu vào data/original/
CSV_PATH    = DATA_DIR["csv"]              # data/Dataset.csv
PERSON_NAME = "person_name"                # Tên người (tên folder)
FRAME_STEP  = 5                            # Cứ mỗi 5 frame thì lấy 1 ảnh
IMG_SIZE    = (160, 160)                   # Resize ảnh đầu ra

# ==========================================
# UPDATE CSV
# ==========================================
def update_csv(csv_path, person_name, saved_count):
    """Thêm các ảnh mới vào Dataset.csv, tránh trùng lặp."""
    existing_ids = set()

    # Đọc các id đã có trong CSV
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["id"])

    # Chuẩn bị các dòng mới
    new_rows = []
    for i in range(1, saved_count + 1):
        file_id = f"{person_name}_{i}.jpg"
        if file_id not in existing_ids:
            new_rows.append({"id": file_id, "label": person_name})

    if not new_rows:
        print("⚠️  Không có dòng mới để thêm (tất cả đã tồn tại trong CSV).")
        return

    # Ghi thêm vào CSV (tạo mới nếu chưa có)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"📝 Đã cập nhật CSV: +{len(new_rows)} dòng → {csv_path}")


# ==========================================
# EXTRACT FRAMES
# ==========================================
def extract_frames(
    video_path  = VIDEO_PATH,
    output_dir  = OUTPUT_DIR,
    csv_path    = CSV_PATH,
    person_name = PERSON_NAME,
    frame_step  = FRAME_STEP,
    img_size    = IMG_SIZE,
):
    # Tạo thư mục lưu ảnh theo tên người
    save_dir = os.path.join(output_dir, person_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Không mở được video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps if fps > 0 else 0

    print(f"📹 Video     : {video_path}")
    print(f"🎞️  Tổng frame: {total_frames}")
    print(f"⏱️  FPS       : {fps:.1f} | Thời lượng: {duration:.1f}s")
    print(f"📂 Lưu vào   : {save_dir}")
    print(f"🔢 Frame step: lấy 1/{frame_step} frame\n")

    frame_idx = 0
    saved     = 0

    with tqdm(total=total_frames, desc="Extracting", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Chỉ lấy frame theo bước nhảy
            if frame_idx % frame_step == 0:
                resized   = cv2.resize(frame, img_size)
                file_name = os.path.join(save_dir, f"{person_name}_{saved + 1}.jpg")
                cv2.imwrite(file_name, resized)
                saved += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"\n🎉 Hoàn thành! Đã trích xuất {saved} ảnh → {save_dir}")

    # Cập nhật CSV nếu có ảnh được lưu
    if saved > 0:
        update_csv(csv_path, person_name, saved)


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    person = input("Nhập tên người trong video: ").strip()
    video  = input("Nhập đường dẫn video (.mp4): ").strip()
    step   = input("Lấy 1 frame mỗi bao nhiêu frame? (mặc định 5): ").strip()

    step = int(step) if step.isdigit() else FRAME_STEP

    if person and video:
        extract_frames(
            video_path  = video,
            person_name = person,
            frame_step  = step,
        )
    else:
        print("❌ Tên và đường dẫn video không được để trống!")