import cv2
import numpy as np
 
def canny_edge_detection(image_path):
    """Mendeteksi tepi menggunakan metode Canny Edge Detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)
    
    cv2.imshow("Canny Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("canny_result.jpg", edges)
    
    return "canny_result.jpg"

# Contoh penggunaan
canny_edge_detection(r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\dataset\rail_segmentation\test\images\1000195092_0011-0_jpeg.rf.1eae7f637435330f6d28c42aa8052bbf.jpg")