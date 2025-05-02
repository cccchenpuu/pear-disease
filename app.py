from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO("best.pt")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400

    f = request.files['file']
    path = os.path.join("temp", f.filename)
    f.save(path)

    results = model(path)
    name = results[0].names[int(results[0].probs.top1)]
    conf = float(results[0].probs.top1conf)

    return jsonify({"class": name, "confidence": conf})
