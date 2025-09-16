import cv2
import os

# -----------------------------
# 1. مسارات المجلدات
# -----------------------------
input_dir = "captured_frames1"   # مجلد الصور الأصلية
output_dir = "captured_frames2"   # مجلد حفظ الصور المقصوصة
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 2. اختيار صورة أولية لتحديد ROI
# -----------------------------
sample_image_path = os.path.join(input_dir, os.listdir(input_dir)[0])
sample_img = cv2.imread(sample_image_path)

# تقليل حجم العرض فقط (مثلاً العرض = 800 بكسل)
scale_width = 300
h, w = sample_img.shape[:2]
scale = scale_width / w
resized_img = cv2.resize(sample_img, (scale_width, int(h * scale)))

# اختيار ROI من الصورة المصغّرة
roi_resized = cv2.selectROI("حدد المربع بالماوس ثم اضغط ENTER", resized_img, False, False)
cv2.destroyAllWindows()

# إرجاع الإحداثيات إلى الحجم الأصلي
x, y, w, h = roi_resized
x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

# -----------------------------
# 3. قص نفس المربع لجميع الصور
# -----------------------------
count = 1000
for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    cropped = img[y:y+h, x:x+w]
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, cropped)
    count += 1

print(f"✅ تم قص {count} صورة وحفظها في {output_dir}")
