import dlib
import cv2
import os
import pickle
import numpy as np
import threading
import time
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from playsound import playsound

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

known_face_encodings = []
known_face_names = []

with open('face_encodings.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

EMAIL_ADDRESS = "SENDER_MAIL@gmail.com"
EMAIL_PASSWORD = "SPECIAL_PASSWORD_SMTP"
EMAIL_TO = ["RECEIVER_MAIL@gmail.com"]

lock = threading.Lock()
unknown_counter = 0
MAX_UNKNOWN_TIME = 5

def send_warning(image_path):
    msg = MIMEMultipart()
    msg['Subject'] = 'UYARI: Odaya yabancı birisi girdi'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ", ".join(EMAIL_TO)

    text = MIMEText(f"Odaya yabancı birisi girdi. Saat: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    msg.attach(text)

    with open(image_path, 'rb') as f:
        img_data = f.read()
    image = MIMEImage(img_data, name=os.path.basename(image_path))
    msg.attach(image)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, EMAIL_TO, msg.as_string())

def unknown_face_timer():
    global unknown_counter
    while True:
        time.sleep(1)
        with lock:
            if unknown_counter > 0:
                unknown_counter -= 1
                if unknown_counter == 0:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    image_path = f"unknown_{timestamp}.jpg"
                    cv2.imwrite(image_path, last_frame)
                    send_warning(image_path)
                    playsound("alert.mp3")

timer_thread = threading.Thread(target=unknown_face_timer)
timer_thread.daemon = True
timer_thread.start()

def get_face_encodings(image):
    dets = detector(image)
    encodings = []
    for det in dets:
        shape = sp(image, det)
        face_encoding = np.array(facerec.compute_face_descriptor(image, shape))
        encodings.append(face_encoding)
    return encodings

def compare_faces(known_encodings, face_encoding, tolerance=0.5):
    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    min_distance = np.min(distances)
    return min_distance <= tolerance, min_distance

video_capture = cv2.VideoCapture(1)
last_frame = None

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    last_frame = frame.copy()

    face_encodings = get_face_encodings(rgb_frame)

    any_known_face = False
    for face_encoding in face_encodings:
        match_found, distance = compare_faces(np.array(known_face_encodings), face_encoding)

        if match_found:
            matched_idx = np.argmin(np.linalg.norm(np.array(known_face_encodings) - face_encoding, axis=1))
            name = known_face_names[matched_idx]
            label = f"{name} ({distance:.2f})"
            any_known_face = True
        else:
            label = f"Yabancı ({distance:.2f})"

        for (i, det) in enumerate(detector(rgb_frame)):
            left, top, right, bottom = (det.left(), det.top(), det.right(), det.bottom())
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if any_known_face:
        with lock:
            unknown_counter = 0
    else:
        with lock:
            if unknown_counter == 0:
                unknown_counter = MAX_UNKNOWN_TIME

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
