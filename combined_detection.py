from ultralytics import YOLO
import cv2
import numpy as np
from canny_edge import canny_edge_detection

# Load YOLOv8 Instance Segmentation model
model = YOLO("yolov8n-seg.pt")
def combined_detection(image_path):
    """Menggabungkan Canny Edge Detection dengan Lane Detection"""
 
# Jalankan Canny Edge Detection
canny_result = canny_edge_detection(r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\canny_result.jpg")

# Jalankan Lane Detection dengan YOLOv8-seg
results = model(r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\lane_detection_result.jpg")
lane_img = results[0].plot()

# Baca hasil Canny Edge Detection
edges = cv2.imread(canny_result, cv2.IMREAD_GRAYSCALE)

# Overlay hasil Lane Detection dengan Canny Edge
combined = cv2.addWeighted(lane_img, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)

# Simpan dan tampilkan hasil
cv2.imshow("Combined Detection", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("combined_result.jpg", combined)
 
# Contoh penggunaan
combined_detection(r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\dataset\rail_segmentation\test\images\1000195092_0011-0_jpeg.rf.1eae7f637435330f6d28c42aa8052bbf.jpg")