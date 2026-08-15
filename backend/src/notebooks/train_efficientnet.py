import os 
from pathlib import Path
import torch 
import torch.nn as nn
from torch.optim  import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from tqdm import tqdm

DATASET = Path("data/indian_food_yolo")
BATCH_SIZE = 8
EPOCHS = 30
LR=1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PATH = "weights/efficientnetv2_best.pth"


train_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    DATASET/"train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATASET/"val",
    transform=val_transform
)

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

num_classes = len(train_dataset.classes)

model = timm.create_model(
    "tf_efficientnetv2_s",
    pretrained=True,
    num_classes=num_classes
)

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LR
)

best_acc = 0


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    pbar = tqdm(train_loader)

    for images, labels in pbar:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

        pbar.set_description(
            f"Epoch {epoch+1}/{EPOCHS}"
        )

    train_acc = 100*correct/total

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    val_acc = 100*correct/total

    print()

    print(f"Train Accuracy : {train_acc:.2f}")

    print(f"Validation Accuracy : {val_acc:.2f}")

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save({

            "epoch": epoch+1,
            "model":model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes":train_dataset.classes,
            "best_value_acc": best_acc
        },SAVE_PATH)

        print("Saved Best Model")

print("finished")


