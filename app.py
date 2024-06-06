import cv2
import mediapipe as mp
import numpy as np
import pickle
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from datetime import datetime

def send_email(subject, body, to_email, from_email, password, image_path):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, 'rb') as img_file:
        img = MIMEImage(img_file.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        msg.attach(img)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(from_email, password)
        server.send_message(msg)

def get_current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def capture_image(image_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Kamera açılamadı!")
    ret, frame = cap.read()
    if not ret:
        raise Exception("Kamera görüntüsü alınamadı!")
    cv2.imwrite(image_path, frame)
    cap.release()

with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        face_encoding = np.array([[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark])

        name = "Bilinmeyen"
        min_distance = float("inf")
        for known_face_encoding, known_face_name in zip(known_face_encodings, known_face_names):
            distance = np.linalg.norm(known_face_encoding - face_encoding)
            if distance < min_distance:
                min_distance = distance
                name = known_face_name if distance < 0.6 else "Bilinmeyen"

        if name == "Bilinmeyen":
            image_path = 'captured_image.jpg'
            capture_image(image_path)

            subject = "Bilinmeyen Ziyaretçi Tespit Edildi"
            body = f"Bilinmeyen Ziyaretçi. Gönderilme tarihi ve saati: {get_current_time()}"
            to_email = "xxx@gmail.com"
            from_email = "xxx@gmail.com"
            password = os.getenv('EMAIL_PASSWORD')  # Ensure this is set correctly

            try:
                send_email(subject, body, to_email, from_email, password, image_path)
            except Exception as e:
                print(f"E-posta gönderme hatası: {e}")

            os.remove(image_path)

        h, w, _ = frame.shape
        cx_min = int(min([landmark.x for landmark in face_landmarks.landmark]) * w)
        cy_min = int(min([landmark.y for landmark in face_landmarks.landmark]) * h)
        cx_max = int(max([landmark.x for landmark in face_landmarks.landmark]) * w)
        cy_max = int(max([landmark.y for landmark in face_landmarks.landmark]) * h)

        cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (0, 0, 255), 2)
        cv2.rectangle(frame, (cx_min, cy_max - 35), (cx_max, cy_max), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (cx_min + 6, cy_max - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
