document.addEventListener('DOMContentLoaded', () => {
    // Pengaturan elemen dari DOM
    const canvas = document.getElementById('canvas');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const predictionSpan = document.getElementById('prediction');
    
    // Konteks untuk menggambar di kanvas
    const ctx = canvas.getContext('2d');
    let drawing = false;

    // Pengaturan kuas
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';

    // Fungsi untuk memulai gambar
    function startDrawing(e) {
        drawing = true;
        draw(e);
    }

    // Fungsi untuk berhenti gambar
    function stopDrawing() {
        drawing = false;
        ctx.beginPath();
    }

    // Fungsi utama untuk menggambar
    function draw(e) {
        if (!drawing) return;
        
        // Dapatkan posisi mouse yang benar relatif terhadap kanvas
        const rect = canvas.getBoundingClientRect();
        let x, y;
        if (e.touches) { // Cek untuk perangkat sentuh (mobile)
            x = e.touches[0].clientX - rect.left;
            y = e.touches[0].clientY - rect.top;
        } else { // Untuk mouse
            x = e.clientX - rect.left;
            y = e.clientY - rect.top;
        }

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    // Fungsi untuk membersihkan kanvas
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        predictionSpan.textContent = '-';
    }

    // Fungsi untuk mengirim gambar ke backend dan mendapatkan prediksi
    function predictDigit() {
        const imageData = canvas.toDataURL('image/png');
        
        // Tampilkan 'loading' sementara menunggu
        predictionSpan.textContent = '...';

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData }),
        })
        .then(response => response.json())
        .then(data => {
            // Tampilkan hasil prediksi
            predictionSpan.textContent = data.prediction;
        })
        .catch(error => {
            console.error('Error:', error);
            predictionSpan.textContent = 'X'; // Tanda error
        });
    }

    // Menambahkan event listeners untuk mouse
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseout', stopDrawing);

    // Menambahkan event listeners untuk sentuhan (mobile)
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchend', stopDrawing);
    canvas.addEventListener('touchmove', draw);

    // Menambahkan event listeners untuk tombol
    predictBtn.addEventListener('click', predictDigit);
    clearBtn.addEventListener('click', clearCanvas);
});