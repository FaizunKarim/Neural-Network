from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import base64
import re
import os
from PIL import Image
from io import BytesIO
from network import SimpleRNN

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Memuat model dari root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'rnn_model.pkl')
net = SimpleRNN.load(model_path)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img_gray = img.convert('L')
    img_np = np.array(img_gray)
    
    img_resized = cv2.resize(img_np, (28, 28), interpolation=cv2.INTER_AREA)
    processed_image = img_resized.astype('float32') / 255.0

    logits = net.feedforward(processed_image)
    probabilities = softmax(logits)
    prediction = int(np.argmax(probabilities))

    return jsonify({
        'prediction': prediction,
        'probabilities': probabilities.flatten().tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)