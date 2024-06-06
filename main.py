import cv2
import mediapipe as mp

# MediaPipe'in gerekli bileşenlerini başlatın
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
t_f_d = 0

# Web kamerasını başlatın
cap = cv2.VideoCapture(1)

with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kameradan görüntü alınamıyor.")
            continue

        image = cv2.resize(image, (1280, 1024))

        # Görüntüyü BGR'den RGB'ye çevirin
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Görüntü üzerinde işlem yaparken performansı artırmak için bu işlemi yapıyoruz
        image.flags.writeable = False

        # Yüz algılama işlemini gerçekleştirin
        results = face_detection.process(image)

        # Görüntüyü tekrar BGR formatına dönüştürün
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Yüz algılandığında değeri 1 olarak ayarla

        face_detected = 0
        if results.detections:
            face_detected = 1
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Sonucu gösterin
        cv2.imshow('MediaPipe Face Detection', image)

        # Yüz algılandığını gösteren değeri yazdırın
        if t_f_d != face_detected:
            print(face_detected)

        t_f_d = face_detected

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
