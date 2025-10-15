from ultralytics import YOLO
import cv2
import numpy as np

# ============================================================
#  REAL-TIME RAIL DETECTION (Hanya Mask & Box untuk "Rail")
# ============================================================

# Load model hasil training terbaik
model_path = r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\rail_training\yolov8_rail_seg2\weights\best.pt"
model = YOLO(model_path)

# Inisialisasi kamera laptop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ Kamera tidak terdeteksi.")
    exit()

print("📷 Tekan 'q' untuk keluar.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Jalankan deteksi YOLOv8
    results = model(frame)
    result = results[0]

    # Salin frame untuk anotasi
    annotated_frame = frame.copy()

    # Ambil koordinat bounding box, kelas, dan mask
    boxes = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()

    if hasattr(result.masks, "data"):
        masks = result.masks.data.cpu().numpy()
    else:
        masks = []

    # Overlay mask dan bounding box hanya untuk kelas rel (0 dan 1)
    for i, box in enumerate(boxes):
        class_id = int(cls[i])
        if class_id in [0, 1]:  # hanya untuk rail
            # Ambil mask
            if len(masks) > i:
                mask = masks[i]
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                color = (0, 255, 0)  # warna hijau
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask > 0.5] = color

                # Overlay mask ke frame
                annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

            # Gambar bounding box
            x1, y1, x2, y2 = map(int, box)
            label = f"Rail ({conf[i]:.2f})"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Tampilkan hasil akhir
    cv2.imshow("YOLOv8 Rail Detection (Mask + Box Only for Rail)", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
