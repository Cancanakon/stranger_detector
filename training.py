import dlib
import cv2
import os
import pickle
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

known_face_encodings = []
known_face_names = []


def load_face_encodings(images_path):
    image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg") or f.endswith(".png")]
    for image_name in image_files:
        image_path = os.path.join(images_path, image_name)
        img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector(rgb_img)

        for det in dets:
            shape = sp(rgb_img, det)
            face_encoding = np.array(facerec.compute_face_descriptor(rgb_img, shape))
            known_face_encodings.append(face_encoding)
            name = image_name.split('_')[0]  # Dosya adını "_" karakterine göre böl ve ilk kısmı al
            known_face_names.append(name)
            print(f"{name} yüz kodlaması oluşturuldu.")

    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print("Yüz kodlamaları dosyaya kaydedildi.")


images_path = "images"
load_face_encodings(images_path)
