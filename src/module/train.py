import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------------
# 1. إعداد الجهاز
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📌 الجهاز المستخدم:", device)

# -----------------------------
# 2. التحويلات للصور
# -----------------------------
train_transform = transforms.Compose(
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -----------------------------
# 3. تحميل البيانات
# -----------------------------
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("dataset/val", transform=val_test_transform)
test_dataset  = datasets.ImageFolder("dataset/test", transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print("📂 الفئات:", train_dataset.classes)

# -----------------------------
# 4. تعريف النموذج
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # فئتين فقط
model = model.to(device)

# -----------------------------
# 5. الخسارة والمُحسّن
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# -----------------------------
# 6. التدريب
# -----------------------------
num_epochs = 8
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\n📌 Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    
    # ---------------- تقييم على val ----------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    print(f"Loss: {train_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
    
    # حفظ أفضل نموذج
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_bottle_binary.pth")
        print("✅ تم حفظ أفضل نموذج")
    
    scheduler.step()

print(f"\n🎯 التدريب انتهى. أفضل دقة على Val = {best_acc*100:.2f}%")

# -----------------------------
# 7. التقييم على test
# -----------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"📊 دقة النموذج على Test = {100*correct/total:.2f}%")
