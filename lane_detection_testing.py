from ultralytics import YOLO
import cv2

# ============================================================
#  REAL-TIME TESTING YOLOv8 INSTANCE SEGMENTATION
# ============================================================

# Load model hasil training (gunakan model terbaik hasil training)
model_path = r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\rail_training\yolov8_rail_seg2\weights\best.pt"
model = YOLO(model_path)

# Inisialisasi kamera laptop (0 untuk kamera utama)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("⚠️ Kamera tidak terdeteksi. Pastikan kamera aktif atau gunakan indeks kamera lain.")
    exit()

print("📷 Tekan 'q' untuk keluar dari jendela deteksi.\n")

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Jalankan deteksi menggunakan model YOLOv8-seg
    results = model(frame)

    # Ambil hasil deteksi (dalam format visualisasi)
    annotated_frame = results[0].plot()  # tampilkan hasil deteksi dan mask

    # Tampilkan hasil di jendela OpenCV
    cv2.imshow("YOLOv8 Rail Detection - Real Time", annotated_frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
