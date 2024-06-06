import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

known_face_encodings = []
known_face_names = []

image_folder = "images"
for person_name in os.listdir(image_folder):
    person_folder = os.path.join(image_folder, person_name)
    if os.path.isdir(person_folder):
        for image_name in os.listdir(person_folder):
            if image_name.endswith(".jpg"):
                image_path = os.path.join(person_folder, image_name)
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    face_encoding = np.array(
                        [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark])
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)

print("Yüzler başarıyla yüklendi ve kodlandı.")

data = {"encodings": known_face_encodings, "names": known_face_names}
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Model başarıyla kaydedildi.")
