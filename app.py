import cv2
import numpy as np
import pickle
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import threading
import time
import os
from playsound import playsound

# Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load the classifier model
classifier_model = load_model('classifier_model.keras')

# Load the face embedding model
face_embedding_model = load_model('face_embedding_model.keras')

# Initialize Mediapipe face detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Create directory for strangers if not exists
os.makedirs("strangers", exist_ok=True)

# Function to handle unknown face detection
def handle_unknown_face(frame, timestamp):
    filename = f"strangers/{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved unknown face image: {filename}")

# Function to be called if no known face detected within 15 seconds
def send_alert():
    print("No known face detected within 15 seconds.")
    try:
        playsound('alert.mp3')
    except:
        print("Failed to play sound alert")

# Thread to monitor the detection of known faces
def monitor_known_faces():
    global last_unknown_time, known_face_detected
    while True:
        if last_unknown_time is not None and not known_face_detected:
            if time.time() - last_unknown_time > 15:
                send_alert()
                break
        time.sleep(1)

last_unknown_time = None
known_face_detected = False

video_capture = cv2.VideoCapture(1)  # Default webcam

# Start the monitoring thread
monitor_thread = threading.Thread(target=monitor_known_faces)
monitor_thread.start()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Resize the face image to the input size of the model
        face_image = cv2.resize(rgb_frame, (224, 224))
        face_image = np.expand_dims(face_image, axis=0)
        face_image = preprocess_input(face_image)

        # Get face embeddings
        face_embedding = face_embedding_model.predict(face_image).flatten()

        # Predict the face
        face_embedding_exp = np.expand_dims(face_embedding, axis=0)
        prediction = classifier_model.predict(face_embedding_exp)
        pred_label = np.argmax(prediction, axis=1)
        confidence = np.max(prediction)  # Prediction confidence
        name = "Unknown"

        # Confidence threshold check
        if confidence > 0.90:  # Adjusted threshold for better accuracy
            name = le.inverse_transform(pred_label)[0]
            known_face_detected = True
        else:
            known_face_detected = False

        h, w, _ = frame.shape
        cx_min = int(min([landmark.x for landmark in face_landmarks.landmark]) * w)
        cy_min = int(min([landmark.y for landmark in face_landmarks.landmark]) * h)
        cx_max = int(max([landmark.x for landmark in face_landmarks.landmark]) * w)
        cy_max = int(max([landmark.y for landmark in face_landmarks.landmark]) * h)

        cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (0, 0, 255), 2)
        cv2.rectangle(frame, (cx_min, cy_max - 35), (cx_max, cy_max), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name} ({confidence:.2f})", (cx_min + 6, cy_max - 6), font, 1.0, (255, 255, 255), 1)

        if name == "Unknown":
            last_unknown_time = time.time()
            timestamp = int(time.time())
            #handle_unknown_face(frame, timestamp)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
