from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import base64
import numpy as np
import cv2
from face_detection import DetectEmotion, detect_emotion_from_image
from database import db, EmotionRecord

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///emotion.db')
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    records = EmotionRecord.query.order_by(EmotionRecord.timestamp.desc()).limit(10).all()
    return render_template('index.html', records=records)

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form.get('name')
    file = request.files.get('file')
    if file and name:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        emotion = detect_emotion_from_image(filepath)
        record = EmotionRecord(name=name, image_path=filepath, emotion=emotion)
        db.session.add(record)
        db.session.commit()
    return redirect(url_for('index'))

@app.route('/detect_webcam', methods=['POST'])
def detect_webcam():
    data = request.get_json()
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'error': 'No image data'}), 400
    
    # Decode base64 image
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save temporarily
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_webcam.jpg')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    cv2.imwrite(temp_path, frame)
    
    # Detect emotion
    emotion = detect_emotion_from_image(temp_path)
    
    return jsonify({'emotion': emotion})

def gen(camera):
    while True:
        frame = camera.get_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(DetectEmotion()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)