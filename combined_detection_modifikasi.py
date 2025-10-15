from ultralytics import YOLO
import cv2
import numpy as np

# ============================================================
#  COMBINED DETECTION: CANNY EDGE + YOLOv8 RAIL SEGMENTATION
# ============================================================

def canny_edge_detection(image_path):
    """Canny Edge Detection dengan parameter low=50, high=150, kernel=5"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)
    cv2.imwrite("canny_result_low50_high150_k5.jpg", edges)
    return "canny_result_low50_high150_k5.jpg"


def combined_detection(image_path):
    """Menggabungkan Canny Edge Detection dengan hasil segmentasi YOLOv8 hanya untuk rel"""
    # Load model hasil training terbaik
    model_path = r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\rail_training\yolov8_rail_seg2\weights\best.pt"
    model = YOLO(model_path)

    # Jalankan Canny Edge Detection
    canny_result_path = canny_edge_detection(image_path)
    edges = cv2.imread(canny_result_path, cv2.IMREAD_GRAYSCALE)

    # Jalankan deteksi YOLOv8 pada gambar input
    results = model(image_path)
    result = results[0]
    original_img = cv2.imread(image_path)

    # Buat salinan gambar asli
    annotated_img = original_img.copy()

    # Ambil mask hasil segmentasi
    if hasattr(result.masks, "data"):
        masks = result.masks.data.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()
    else:
        masks = []
        cls = []

    # Overlay hanya untuk kelas rel (0 dan 1)
    for i, class_id in enumerate(cls):
        if int(class_id) in [0, 1]:
            mask = masks[i]
            mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
            color = (0, 255, 0)  # warna hijau untuk rel
            colored_mask = np.zeros_like(original_img, dtype=np.uint8)
            colored_mask[mask > 0.5] = color
            annotated_img = cv2.addWeighted(annotated_img, 1, colored_mask, 0.5, 0)

    # Gabungkan hasil YOLO (mask rel) dengan hasil Canny Edge
    combined = cv2.addWeighted(annotated_img, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # Simpan dan tampilkan hasil akhir
    cv2.imshow("Combined Detection (Canny + Rail Segmentation)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("combined_result_final.jpg", combined)
    print("✅ Hasil kombinasi disimpan sebagai 'combined_result_final.jpg'")


# ============================================================
#  CONTOH PENGGUNAAN
# ============================================================

image_path = r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\dataset\rail_segmentation\test\images\1000195092_0011-0_jpeg.rf.1eae7f637435330f6d28c42aa8052bbf.jpg"
combined_detection(image_path)
