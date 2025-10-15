import cv2
import os

def canny_experiment(image_path):
    """
    Eksperimen perubahan parameter pada Canny Edge Detection.
    - Mengubah nilai ambang bawah dan atas (low/high threshold)
    - Mengubah ukuran kernel Gaussian (blur)
    - Menyimpan semua hasil deteksi untuk dibandingkan
    """

    # Baca gambar dalam format grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Gambar tidak ditemukan. Periksa kembali path gambar.")

    # Daftar kombinasi parameter yang akan diuji:
    # (low_threshold, high_threshold, ukuran_kernel)
    parameter_set = [
        (25, 75, 3),
        (50, 150, 5),
        (75, 200, 7),
        (100, 250, 7)
    ]

    # Buat folder output untuk menyimpan hasil
    os.makedirs("output_canny", exist_ok=True)

    # Lakukan deteksi dengan setiap kombinasi parameter
    for low, high, k in parameter_set:
        # Tahap 1: Penghalusan citra menggunakan Gaussian Blur
        blurred = cv2.GaussianBlur(img, (k, k), 0)

        # Tahap 2: Deteksi tepi menggunakan Canny
        edges = cv2.Canny(blurred, low, high)

        # Tahap 3: Simpan hasil deteksi
        filename = f"output_canny/canny_low{low}_high{high}_k{k}.jpg"
        cv2.imwrite(filename, edges)
        print(f"Hasil tersimpan: {filename}")

    print("\nEksperimen selesai! Cek folder 'output_canny' untuk melihat hasilnya.")


if __name__ == "__main__":
    # Path ke gambar rel kereta (ganti sesuai file kamu)
    image_path = r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M6\dataset\rail_segmentation\test\images\1000195092_0011-0_jpeg.rf.1eae7f637435330f6d28c42aa8052bbf.jpg"
    
    # Jalankan eksperimen
    canny_experiment(image_path)
