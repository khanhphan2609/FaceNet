import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# IMPORT MODULE
from model.dataset import TripletFaceDataset
from model.model import get_facenet_model, get_triplet_loss

# IMPORT CONFIG
from config import DATA_DIR, TRAIN, WEIGHTS_DIR

# ==========================================
# CONFIG
# ==========================================
DATASET_DIR = DATA_DIR["faces"]

BATCH_SIZE = TRAIN["batch_size"]
EPOCHS = TRAIN["epochs"]
LEARNING_RATE = TRAIN["learning_rate"]
MARGIN = TRAIN["margin"]

SAVE_DIR = WEIGHTS_DIR["root"]

# ==========================================
# TRAIN
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Training on: {device}")

    # DATASET
    dataset = TripletFaceDataset(root_dir=DATASET_DIR)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )

    # MODEL
    model = get_facenet_model(device=device, freeze_features=False)
    model.train()

    criterion = get_triplet_loss(margin=MARGIN)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==========================================
    # LOOP
    # ==========================================
    for epoch in range(EPOCHS):
        total_loss = 0.0

        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

        for _, (anchor, positive, negative) in progress_bar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # FORWARD
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)

            # BACKWARD
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}\n")

        # SAVE MODEL
        save_path = os.path.join(SAVE_DIR, f"facenet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

    print("🎉 Training complete!")

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print("❌ Dataset not found")
        exit()

    train()