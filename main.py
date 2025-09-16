
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from ultralytics import YOLO
# from PIL import Image
# import cv2

# # -----------------------------
# # 1. الجهاز
# # -----------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("الجهاز المستخدم:", device)

# # -----------------------------
# # 2. كلاس YOLO لكشف الزجاجة
# # -----------------------------
# yolo_model = YOLO("yolov8n.pt")  # نموذج جاهز يعرف الكائنات ومنها bottle

# # -----------------------------
# # 3. نموذج التصنيف (ملصق / بدون ملصق)
# # -----------------------------
# classes = ['with_label', 'without_label']

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# classifier = models.resnet18(pretrained=False)
# classifier.fc = nn.Linear(classifier.fc.in_features, 2)
# classifier.load_state_dict(torch.load("best_bottle_binary.pth", map_location=device))
# classifier = classifier.to(device)
# classifier.eval()

# # -----------------------------
# # 4. دالة التنبؤ على ROI
# # -----------------------------
# def predict_image(img, model):
#     img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     img_t = transform(img_pil).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(img_t)
#         _, pred = torch.max(output, 1)
#     return classes[pred.item()]

# # -----------------------------
# # 5. تشغيل الكاميرا
# # -----------------------------
# cap = cv2.VideoCapture(1)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # كشف باستخدام YOLO
#     results = yolo_model(frame, verbose=False)

#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls[0])  # كلاس YOLO
#             label_name = yolo_model.names[cls_id]

#             # نتأكد إنه Bottle فقط
#             if label_name == "bottle":
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # إحداثيات البوكس
#                 roi = frame[y1:y2, x1:x2]

#                 if roi.size != 0:
#                     # تصنيف (مع ملصق / بدون ملصق)
#                     label = predict_image(roi, classifier)

#                     # اللون حسب النتيجة
#                     color = (0, 255, 0) if label == "with_label" else (0, 0, 255)

#                     # رسم البوكس والنص
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, label, (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     cv2.imshow("AI Bottle Detector", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()












import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import cv2

# -----------------------------
# 1. الجهاز
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("الجهاز المستخدم:", device)

# -----------------------------
# 2. كلاس YOLO لكشف الزجاجة
# -----------------------------
yolo_model = YOLO("yolov8n.pt")  # نموذج جاهز يعرف الكائنات ومنها bottle

# -----------------------------
# 3. نموذج التصنيف (ملصق / بدون ملصق)
# -----------------------------
classes = ['with_label', 'without_label']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

classifier = model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
classifier.fc = nn.Linear(classifier.fc.in_features, 2)

# تحميل الوزن مع map_location للجهاز الصحيح
classifier.load_state_dict(torch.load("best_bottle_binary.pth", map_location=device))
classifier = classifier.to(device)
classifier.eval()

# -----------------------------
# 4. دالة التنبؤ على ROI
# -----------------------------
def predict_image(img, model):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_t = transform(img_pil).unsqueeze(0).to(device)  # إرسال Tensor إلى GPU
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]

# -----------------------------
# 5. تشغيل الكاميرا
# -----------------------------
def main():
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # كشف باستخدام YOLO
        results = yolo_model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])  # كلاس YOLO
                label_name = yolo_model.names[cls_id]

                # نتأكد إنه Bottle فقط
                if label_name == "bottle":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # إحداثيات البوكس
                    roi = frame[y1:y2, x1:x2]

                    if roi.size != 0:
                        # تصنيف (مع ملصق / بدون ملصق)
                        label = predict_image(roi, classifier)

                        # اللون حسب النتيجة
                        color = (0, 255, 0) if label == "with_label" else (0, 0, 255)

                        # رسم البوكس والنص
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("AI Bottle Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# 6. حماية multiprocessing في Windows
# -----------------------------
if __name__ == "__main__":
    torch.cuda.empty_cache()  # تنظيف أي ذاكرة GPU قديمة
    main()

name = "plastic-bottle-detection"