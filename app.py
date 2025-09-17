import os
import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from flask import Flask, request, render_template, url_for, redirect
from werkzeug.utils import secure_filename

# --- INITIALIZATION ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# --- LOAD THE TRAINED MODEL AND FACE DETECTOR ---
# Load them once at the start to save time on each request.
print("Loading deepfake detection model...")
model = tf.keras.models.load_model('deepfake_detector_model.h5')
print("Model loaded.")

print("Loading MTCNN face detector...")
detector = MTCNN()
print("Face detector loaded.")

IMAGE_SIZE = 224  # The image size your model expects


def analyze_video(video_path):
    """
    Analyzes a video to detect deepfakes.
    Returns the verdict and confidence score.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error", "Could not open video file."

    predictions = []
    frame_count = 0
    frame_interval = 5  # Analyze 1 frame every 5 frames to speed things up

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Detect face in the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(frame_rgb)

            if results:
                # Use the first detected face
                bounding_box = results[0]['box']
                x, y, w, h = bounding_box
                face = frame[max(0, y):y + h, max(0, x):x + w]

                if face.size > 0:
                    # Preprocess the face for the model
                    face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
                    face = face.astype('float32') / 255.0
                    face = np.expand_dims(face, axis=0)  # Add batch dimension

                    # Make a prediction
                    pred = model.predict(face)[0][0]
                    predictions.append(pred)

        frame_count += 1

    cap.release()

    if not predictions:
        return "Inconclusive", "No faces were detected in the video."

    # Calculate the average prediction
    avg_prediction = np.mean(predictions)

    if avg_prediction > 0.5:
        verdict = "Real"
        confidence = avg_prediction * 100
    else:
        verdict = "Fake"
        confidence = (1 - avg_prediction) * 100

    return verdict, f"{confidence:.2f}"


# --- FLASK ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded file securely
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # Analyze the video
            verdict, confidence = analyze_video(video_path)

            # Clean up the uploaded file
            os.remove(video_path)

            # Show the result
            return render_template('result.html', prediction=verdict, confidence=confidence)

    # Show the upload form if it's a GET request
    return render_template('index.html')


if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Run the Flask app
    app.run(debug=True)