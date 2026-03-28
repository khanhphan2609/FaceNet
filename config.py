import os

# Root của project (tự động tính từ vị trí file config.py)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# PATHS
# ==========================================
DATA_DIR = {
    "original": os.path.join(ROOT_DIR, "data", "original"),
    "faces":    os.path.join(ROOT_DIR, "data", "faces"),
    "csv":      os.path.join(ROOT_DIR, "data", "Dataset.csv"),
}

WEIGHTS_DIR = {
    "root":      os.path.join(ROOT_DIR, "weights"),
    "model":     os.path.join(ROOT_DIR, "weights", "facenet_epoch_10.pth"),
    "database":  os.path.join(ROOT_DIR, "weights", "face_database.pt"),
    "attendance":os.path.join(ROOT_DIR, "weights", "attendance.csv"),
}

# ==========================================
# TRAIN HYPERPARAMETERS
# ==========================================
TRAIN = {
    "batch_size":    16,
    "epochs":        30,
    "learning_rate": 0.0001,
    "margin":        1.0,
}

# ==========================================
# RECOGNITION
# ==========================================
RECOGNITION = {
    "similarity_threshold": 0.30,
    "vote_threshold":       10,
}

# ==========================================
# DATA COLLECTION
# ==========================================
CAMERA = {
    "max_samples": 150,
}