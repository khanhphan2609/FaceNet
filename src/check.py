import os
import cv2
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
import csv
from datetime import datetime

# IMPORT CONFIG
from config import DATA_DIR, WEIGHTS_DIR, RECOGNITION

# ==========================================
# CONFIG
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_DIR = DATA_DIR["faces"]
WEIGHTS_PATH = WEIGHTS_DIR["model"]
CACHE_FILE = WEIGHTS_DIR["database"]
CSV_FILE = WEIGHTS_DIR["attendance"]

SIMILARITY_THRESHOLD = RECOGNITION["similarity_threshold"]
VOTE_THRESHOLD = RECOGNITION["vote_threshold"]

print(f"🚀 Running on: {DEVICE}")

# ==========================================
# MODEL
# ==========================================
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)

model = InceptionResnetV1(pretrained=None, classify=False).to(DEVICE)

try:
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE), strict=False)
    print("✅ Loaded trained weights")
except:
    model = InceptionResnetV1(pretrained='vggface2').to(DEVICE)
    print("⚠️ Using pretrained model")

model.eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ==========================================
# EMBEDDING
# ==========================================
def get_embedding(img_pil):
    face_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(face_tensor)
        emb = F.normalize(emb, p=2, dim=1)
    return emb

# ==========================================
# BUILD DATABASE
# ==========================================
def build_person_embedding(person_dir):
    embeddings = []

    for img_name in os.listdir(person_dir):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                img = Image.open(os.path.join(person_dir, img_name)).convert('RGB')
                emb = get_embedding(img)
                embeddings.append(emb)
            except:
                continue

    if not embeddings:
        return None

    centroid = torch.stack(embeddings).mean(dim=0)
    return F.normalize(centroid, p=2, dim=1)


def load_database():
    if os.path.exists(CACHE_FILE):
        print("⚡ Loading database...")
        return torch.load(CACHE_FILE, map_location=DEVICE)
    return {}


def update_database(database):
    print("🔄 Checking for new people...")

    existing = set(database.keys())
    current = set(os.listdir(DATASET_DIR))

    new_people = current - existing

    if not new_people:
        print("✅ No new people found")
        return database

    print(f"🆕 Found {len(new_people)} new people")

    for person in new_people:
        person_dir = os.path.join(DATASET_DIR, person)

        if not os.path.isdir(person_dir):
            continue

        print(f"➡️ Processing {person}...")

        centroid = build_person_embedding(person_dir)

        if centroid is not None:
            database[person] = centroid
            print(f"✅ Added {person}")

    torch.save(database, CACHE_FILE)
    print("💾 Database updated")

    return database

# ==========================================
# ATTENDANCE
# ==========================================
def log_attendance(name):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Name", "Time", "Date"])

        now = datetime.now()
        writer.writerow([name, now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d")])

    print(f"📝 {name} attended")

# ==========================================
# RECOGNITION
# ==========================================
def recognize(database):
    cap = cv2.VideoCapture(0)

    frame_count = {name: 0 for name in database}
    attended = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, _ = mtcnn.detect(img_pil)

        if boxes is not None:

            # ==========================================
            # 👉 CHỌN FACE LỚN NHẤT (NGƯỜI GẦN NHẤT)
            # ==========================================
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            max_idx = areas.index(max(areas))
            box = boxes[max_idx]

            x1, y1, x2, y2 = map(int, box)

            # Optional: bỏ qua mặt quá nhỏ
            if (x2 - x1) < 80:
                cv2.imshow("FaceNet System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            try:
                face = img_pil.crop((x1, y1, x2, y2))
                emb_unknown = get_embedding(face)
            except:
                continue

            best_name = "Unknown"
            best_score = -1

            for name, emb_db in database.items():
                sim = F.cosine_similarity(emb_unknown, emb_db).item()

                if sim > best_score:
                    best_score = sim
                    best_name = name

            if best_score > SIMILARITY_THRESHOLD:
                if best_name in attended:
                    color = (255, 150, 0)
                    text = f"{best_name} (Done)"
                else:
                    frame_count[best_name] += 1
                    count = frame_count[best_name]

                    color = (0, 255, 255)
                    text = f"{best_name}: {count}/{VOTE_THRESHOLD}"

                    if count >= VOTE_THRESHOLD:
                        log_attendance(best_name)
                        attended.add(best_name)
            else:
                text = "Unknown"
                color = (0, 0, 255)

                for k in frame_count:
                    frame_count[k] = 0

            # Vẽ duy nhất 1 face
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("FaceNet System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print("❌ Dataset not found")
        exit()

    db = load_database()
    db = update_database(db)

    if not db:
        print("❌ Database empty")
    else:
        recognize(db)