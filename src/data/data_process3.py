import os
import shutil
import random
from PIL import Image
from torchvision import transforms

# -----------------------------
# 1. إعداد المجلدات
# -----------------------------
input_dir = "train"  # المجلد الأصلي الذي يحتوي على with_label و without_label
output_dir = "dataset"  # المجلد النهائي

classes = ["with_label", "without_label"]
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# إنشاء المجلدات
for folder in ["train", "val", "test", "train_aug"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, folder, cls), exist_ok=True)

# -----------------------------
# 2. تقسيم الصور train/val/test
# -----------------------------
for cls in classes:
    imgs = os.listdir(os.path.join(input_dir, cls))
    random.shuffle(imgs)

    total = len(imgs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_imgs = imgs[:train_end]
    val_imgs = imgs[train_end:val_end]
    test_imgs = imgs[val_end:]

    # نسخ الصور للتدريب
    for img in train_imgs:
        shutil.copy(os.path.join(input_dir, cls, img),
                    os.path.join(output_dir, "train", cls, img))

    # نسخ الصور ل validation
    for img in val_imgs:
        shutil.copy(os.path.join(input_dir, cls, img),
                    os.path.join(output_dir, "val", cls, img))

    # نسخ الصور للاختبار
    for img in test_imgs:
        shutil.copy(os.path.join(input_dir, cls, img),
                    os.path.join(output_dir, "test", cls, img))

print("✅ تم تقسيم الصور إلى train/val/test بنسبة 70/10/20")

# -----------------------------
# 3. Augmentation للصور
# -----------------------------
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
])

for cls in classes:
    train_folder = os.path.join(output_dir, "train", cls)
    aug_folder = os.path.join(output_dir, "train_aug", cls)
    images = os.listdir(train_folder)

    count = 0
    target_num = max(len(os.listdir(os.path.join(output_dir, "train", c))) for c in classes)
    # نكرر الصور مع Augmentation حتى نصل لعدد مماثل للفئة الأكبر
    while count < target_num:
        for img_name in images:
            img_path = os.path.join(train_folder, img_name)
            img = Image.open(img_path).convert("RGB")
            img_aug = augment(img)
            save_path = os.path.join(aug_folder, f"aug_{count}_{img_name}")
            img_aug.save(save_path)
            count += 1
            if count >= target_num:
                break

print("✅ تم عمل Augmentation لجميع الفئات")
