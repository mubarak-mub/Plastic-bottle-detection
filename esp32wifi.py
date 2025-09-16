import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import cv2
import socket
import time

# -----------------------------
# 1. إعداد الجهاز
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("الجهاز المستخدم:", device)

# -----------------------------
# 2. تحميل نموذج YOLO
# -----------------------------
yolo_model = YOLO("yolov8n.pt")  # YOLO v8 Nano

# -----------------------------
# 3. تحميل نموذج التصنيف
# -----------------------------
classes = ['with_label', 'without_label']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

classifier = models.resnet18(pretrained=False)
classifier.fc = nn.Linear(classifier.fc.in_features, 2)
classifier.load_state_dict(torch.load("best_bottle_binary.pth", map_location=device))

classifier = classifier.to(device)
classifier.eval()

# -----------------------------
# 4. دالة التنبؤ
# -----------------------------
def predict_image(img, model):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_t = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]

# -----------------------------
# 5. فتح الاتصال مع ESP32 عبر WiFi
# -----------------------------
esp_ip = "192.168.226.102"   # ⚠️ غيّر هذا حسب الـ IP الذي يطبعه ESP32 عند الاتصال بالواي فاي
esp_port = 1234

esp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
esp.connect((esp_ip, esp_port))
time.sleep(1)
print("تم الاتصال مع ESP32 عبر WiFi")

# -----------------------------
# 6. تشغيل الكاميرا
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# 7. منع التكرار في الإرسال
# -----------------------------
last_state = None  # 'on' أو 'off'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, verbose=False)
    label = None  # نتيجة التصنيف الحالية

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label_name = yolo_model.names[cls_id]

            if label_name == "bottle":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]

                if roi.size != 0:
                    label = predict_image(roi, classifier)
                    color = (0, 255, 0) if label == "with_label" else (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # -----------------------------
    # 8. إرسال الأمر إلى ESP32 عبر WiFi
    # -----------------------------
    if label:
        command = "on" if label == "with_label" else "off"

        if command != last_state:
            try:
                esp.send((command + "\n").encode())
                print("تم الإرسال إلى ESP32:", command)
                last_state = command
            except Exception as e:
                print("خطأ في الإرسال إلى ESP32:", e)

    cv2.imshow("AI Bottle Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 9. إغلاق الموارد
# -----------------------------
cap.release()
esp.close()
cv2.destroyAllWindows()
