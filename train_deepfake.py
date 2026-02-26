import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from collections import Counter
from tqdm import tqdm
import os

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    DATASET_PATH = "real_vs_fake/real-vs-fake/train"

    # ---------------- TRANSFORMS ----------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageFolder(DATASET_PATH, transform=transform)

    print("Classes:", dataset.classes)

    # ---------------- TRAIN / VAL SPLIT ----------------
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # ---------------- CLASS WEIGHTS ----------------
    targets = dataset.targets
    class_count = Counter(targets)

    print("Class distribution:", class_count)

    weights = torch.tensor([
        1.0 / class_count[0],
        1.0 / class_count[1]
    ], dtype=torch.float).to(device)

    # ---------------- DATALOADERS ----------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ---------------- MODEL ----------------
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Train only final layer
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # ---------------- LOSS & OPTIMIZER ----------------
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0003)

    EPOCHS = 10
    best_acc = 0

    # ================= TRAIN LOOP =================
    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # ---------------- VALIDATION ----------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        print(f"\nEpoch {epoch+1}")
        print("Train Loss:", round(running_loss, 4))
        print("Validation Accuracy:", round(val_acc, 2), "%")

        # ---------------- SAVE BEST MODEL ----------------
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("deepfake_model", exist_ok=True)
            torch.save(model.state_dict(), "deepfake_model/model.pth")
            print("✅ Best model saved")

    print("\n🏁 Training Complete")
    print("🔥 Best Validation Accuracy:", round(best_acc, 2), "%")


if __name__ == "__main__":
    main()