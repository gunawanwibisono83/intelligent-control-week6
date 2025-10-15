from ultralytics import YOLO
import cv2
import numpy as np

# ============================================================
#  CANNY EDGE DETECTION
# ============================================================
def canny_edge_detection(image_path):
    """Mendeteksi tepi menggunakan metode Canny Edge Detection (parameter asli)"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)
    cv2.imwrite("canny_result.jpg", edges)
    return "canny_result.jpg"


# ============================================================
#  COMBINED DETECTION: CANNY + YOLOv8 SEGMENTATION (NO BOX, NO LABEL)
# ============================================================
def combined_detection(image_path):
    """Menggabungkan hasil Canny Edge Detection dengan hasil segmentasi YOLOv8 (tanpa box dan label)"""

    # Load model hasil training terbaik
    model_path = r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\rail_training\yolov8_rail_seg2\weights\best.pt"
    model = YOLO(model_path)

    # Jalankan Canny Edge Detection
    canny_result_path = canny_edge_detection(image_path)
    edges = cv2.imread(canny_result_path, cv2.IMREAD_GRAYSCALE)

    # Jalankan deteksi YOLOv8 untuk segmentasi
    results = model(image_path)
    result = results[0]

    # Buat salinan gambar asli
    base_img = cv2.imread(image_path)
    overlay = base_img.copy()

    # Ambil mask hasil segmentasi
    if hasattr(result, "masks") and result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, "cls") else np.zeros(len(masks))

        # Warna berbeda untuk tiap kelas (2 kelas: Branch Rail dan Main Rail)
        colors = {
            0: (139, 0, 0),    # biru tua (navy) → Branch Rail
            1: (255, 255, 0)   # biru aqua → Main Rail
        }

        for i, mask in enumerate(masks):
            mask_resized = cv2.resize(mask, (base_img.shape[1], base_img.shape[0]))
            color = colors.get(int(cls[i]), (255, 255, 255))
            colored_mask = np.zeros_like(base_img, dtype=np.uint8)
            colored_mask[mask_resized > 0.5] = color

            # Overlay dengan transparansi 0.5
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

    # Overlay hasil Canny Edge di atas segmentasi
    combined = cv2.addWeighted(overlay, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # Tampilkan hasil akhir
    cv2.imshow("Combined Detection (Mask Only + Canny)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Simpan hasil akhir
    cv2.imwrite("combined_result_mask_only.jpg", combined)
    print("✅ Hasil kombinasi disimpan sebagai 'combined_result_mask_only.jpg'")


# ============================================================
#  CONTOH PENGGUNAAN
# ============================================================
if __name__ == "__main__":
    image_path = r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\dataset\rail_segmentation\test\images\1000195092_0011-0_jpeg.rf.1eae7f637435330f6d28c42aa8052bbf.jpg"
    combined_detection(image_path)
