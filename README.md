# 🎯 Face Recognition Attendance System (FaceNet)

## 📌 Giới thiệu

Đây là hệ thống **nhận diện khuôn mặt + điểm danh tự động** sử dụng FaceNet và Triplet Loss.

Pipeline đầy đủ bao gồm:

* Thu thập dữ liệu khuôn mặt
* Tiền xử lý (lọc + chuẩn hóa ảnh)
* Huấn luyện model
* Nhận diện realtime qua webcam
* Lưu log điểm danh

---

## 🧠 Công nghệ sử dụng

* Python
* PyTorch
* facenet-pytorch (MTCNN + InceptionResnetV1)
* OpenCV
* NumPy

---

## 📁 Cấu trúc project

```
FaceNet/
│
├── data/
│   ├── original/        # Ảnh gốc
│   ├── faces/           # Ảnh đã xử lý
│   └── Dataset.csv
│
├── model/
│   ├── dataset.py       # Triplet dataset
│   └── model.py         # Model + loss
│
├── src/
│   ├── camera.py        # Thu thập ảnh
│   ├── data_prepare.py  # Xử lý dữ liệu
│   ├── train.py         # Train model
│   └── check.py         # Nhận diện realtime
│
├── weights/
│   ├── *.pth            # Model weights
│   ├── face_database.pt
│   └── attendance.csv
│
├── config.py
└── requirements.txt
```

---

## ⚙️ Cài đặt

### 1. Clone project

```bash
git clone <repo_url>
cd FaceNet
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

---

## 🚀 Cách sử dụng

### 🔹 Bước 1: Thu thập dữ liệu

```bash
python -m src.camera
```

* Chụp ảnh khuôn mặt cho từng người
* Mỗi người nên có 30–50 ảnh

---

### 🔹 Bước 2: Tiền xử lý dữ liệu

```bash
python -m src.data_prepare
```

Bao gồm:

* Detect mặt
* Align khuôn mặt
* Lọc ảnh mờ
* Chuẩn hóa ánh sáng (CLAHE)

---

### 🔹 Bước 3: Train model

```bash

python -m src.train
```

* Sử dụng Triplet Loss
* Lưu weights vào thư mục `weights/`

---

### 🔹 Bước 4: Nhận diện

```bash

python -m src.check
```

* Mở webcam
* Nhận diện khuôn mặt realtime
* Ghi log vào `attendance.csv`

---

## 📊 Cấu hình

Chỉnh trong `config.py`

```python
TRAIN = {
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 0.0001,
    "margin": 1.0,
}

RECOGNITION = {
    "similarity_threshold": 0.30,
    "vote_threshold": 10,
}
```

---

## 📈 Pipeline hệ thống

```
Raw Images
   ↓
Face Detection (MTCNN)
   ↓
Face Alignment
   ↓
Quality Filter (Blur)
   ↓
Lighting Normalize (CLAHE)
   ↓
Train (Triplet Loss)
   ↓
Embedding Database
   ↓
Realtime Recognition
   ↓
Attendance Logging
```

---

## ✅ Tính năng

* Nhận diện khuôn mặt realtime
* Chống nhiễu bằng voting
* Tự động cập nhật database
* Lọc ảnh chất lượng thấp
* Chuẩn hóa ánh sáng

---

## 🔥 Hướng phát triển

* API với FastAPI
* Web dashboard điểm danh
* Anti-spoofing (chống ảnh giả)
* Deploy Docker

---

## 👨‍💻 Tác giả

Khánh Phan x Duy Anh x Tuấn Anh

---

## 📄 License

MIT License
