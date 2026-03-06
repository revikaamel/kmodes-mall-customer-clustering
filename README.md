# 🛍️ Clustering Mall Customer Menggunakan K-Modes

## 📌 Deskripsi Project
Project ini bertujuan untuk melakukan **segmentasi pelanggan mall** menggunakan metode **clustering K-Modes**. Berbeda dengan K-Means yang digunakan untuk data numerik, **K-Modes** dirancang khusus untuk menangani **data kategorikal**.

Dengan melakukan clustering, pelanggan dapat dikelompokkan berdasarkan karakteristik tertentu sehingga membantu dalam **analisis perilaku pelanggan dan strategi pemasaran**.

Project ini dibuat sebagai bagian dari **praktikum Machine Learning / Data Mining**.

---

## 📊 Dataset

Dataset yang digunakan adalah **Mall Customer Dataset** yang berisi informasi mengenai karakteristik pelanggan mall.

Beberapa fitur yang digunakan antara lain:

- `CustomerID` → ID unik pelanggan  
- `Gender` → jenis kelamin pelanggan  
- `Age` → umur pelanggan  
- `Annual Income (k$)` → pendapatan tahunan pelanggan  
- `Spending Score (1-100)` → skor pengeluaran pelanggan  

Pada project ini, fitur yang digunakan akan **dikategorikan atau diproses menjadi data kategorikal** agar dapat digunakan oleh algoritma **K-Modes**.

---

## 🧠 Metode yang Digunakan

### K-Modes Clustering
K-Modes merupakan algoritma clustering yang merupakan pengembangan dari **K-Means**, tetapi dirancang untuk **data kategorikal**.

Perbedaan utama dengan K-Means:

| K-Means | K-Modes |
|------|------|
| Data numerik | Data kategorikal |
| Mean sebagai centroid | Mode sebagai centroid |
| Euclidean distance | Matching dissimilarity |

K-Modes bekerja dengan cara:

1. Menentukan jumlah cluster **K**
2. Memilih centroid awal
3. Menghitung kesamaan kategori antar data
4. Memperbarui centroid berdasarkan **mode**
5. Mengulangi proses hingga cluster stabil

---

## ⚙️ Tahapan Proses

1. Import library Python
2. Load dataset Mall Customer
3. Data preprocessing
4. Konversi fitur menjadi data kategorikal
5. Menentukan jumlah cluster
6. Melatih model **K-Modes**
7. Menghasilkan hasil clustering
8. Analisis hasil segmentasi pelanggan
