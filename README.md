# Neural Network Web Application

Aplikasi web ini adalah implementasi sederhana dari Jaringan Saraf Berulang (RNN) untuk klasifikasi gambar, yang kemungkinan besar dirancang untuk pengenalan digit tulisan tangan (seperti dataset MNIST). Aplikasi ini menyediakan antarmuka web di mana pengguna dapat mengunggah gambar, dan model RNN akan memprediksi kelas gambar tersebut.

## Fitur

*   **Klasifikasi Gambar**: Menggunakan Simple RNN untuk memprediksi kelas gambar yang diunggah.
*   **Antarmuka Web Interaktif**: Antarmuka pengguna berbasis web untuk interaksi yang mudah.
*   **Pelatihan Model**: Kemampuan untuk melatih model RNN menggunakan data yang disediakan.
*   **Penyimpanan/Pemuatan Model**: Model dapat disimpan dan dimuat untuk penggunaan di masa mendatang.

## Teknologi yang Digunakan

*   **Backend**: Python (Flask, NumPy, OpenCV, PIL)
*   **Frontend**: HTML, CSS, JavaScript
*   **Model**: Simple Recurrent Neural Network (RNN)

## Struktur Repository

...

## Cara Mengatur dan Menjalankan

1.  **Kloning Repositori**:
    ```bash
    git clone https://github.com/FaizunKarim/Neural_Network.git
    cd Neural_Network
    ```

2.  **Buat dan Aktifkan Lingkungan Virtual** (Opsional tapi disarankan):
    ```bash
    python -m venv venv
    # Di Windows
    .\venv\Scripts\activate
    # Di macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal Dependensi**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Melatih Model** (Opsional, jika Anda ingin melatih model baru):
    Buka `train.ipynb` di lingkungan Jupyter Anda dan jalankan sel-selnya untuk melatih model dan menyimpannya. Model yang sudah terlatih (`rnn_model.pkl`) diharapkan ada di direktori `api/`.

5.  **Jalankan Aplikasi Flask**:
    ```bash
    python api/index.py
    ```
    Aplikasi akan berjalan di `http://127.0.0.1:5000/` secara default.

## Penggunaan

Setelah aplikasi berjalan, buka browser web Anda dan navigasikan ke `http://127.0.0.1:5000/`. Anda akan melihat antarmuka di mana Anda dapat menggambar atau mengunggah gambar, dan aplikasi akan menampilkan prediksinya.

