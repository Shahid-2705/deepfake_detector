import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ================= CONFIG =================
DATA_DIR = "data/face_crops"
BATCH_SIZE = 32
EPOCHS = 20
LR = 2e-4
IMG_SIZE = 224
MODEL_NAME = "face_deepfake_b3"
# ==========================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- STRONG FORENSIC AUGMENTATION --------
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------- DATASET --------
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -------- CLASS IMBALANCE --------
    targets = [full_dataset.targets[i] for i in train_dataset.indices]
    real_count = targets.count(0)
    fake_count = targets.count(1)

    pos_weight = torch.tensor([real_count / fake_count]).to(device)

    print(f"Real: {real_count}, Fake: {fake_count}")
    print(f"Using pos_weight: {pos_weight.item():.4f}")

    # -------- MODEL --------
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs("models", exist_ok=True)

    best_val_acc = 0.0

    print("\n🔥 Starting Forensic Deepfake Training...\n")

    for epoch in range(EPOCHS):
        # -------- TRAIN --------
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                logits = model(imgs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        # -------- SAVE CHECKPOINT --------
        torch.save(model.state_dict(), f"models/{MODEL_NAME}_epoch_{epoch+1}.pth")

        # -------- SAVE BEST MODEL --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"models/{MODEL_NAME}_best.pth")
            print("✅ Best model updated.")

    print("\n🎯 Training Completed.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best model saved as: models/{MODEL_NAME}_best.pth")


if __name__ == "__main__":
    main()
