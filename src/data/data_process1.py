import cv2
import os

# مسار الفيديو
video_path = "7.mp4"

# مجلد حفظ الصور
output_dir = "captured_frames1"
os.makedirs(output_dir, exist_ok=True)

# فتح الفيديو
cap = cv2.VideoCapture(video_path)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # نهاية الفيديو

    # حفظ الصورة
    frame_filename = os.path.join(output_dir, f"fram_{count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    print(f"تم حفظ: {frame_filename}")

    count += 1

cap.release()
print("✅ تم استخراج كل الصور من الفيديو.")

