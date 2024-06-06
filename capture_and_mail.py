import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from datetime import datetime
import cv2


# Bilgisayarın kamerasından fotoğraf çekme
def capture_image(image_path):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise Exception("Kamera açılamadı!")

    ret, frame = cap.read()
    if not ret:
        raise Exception("Kamera görüntüsü alınamadı!")

    cv2.imwrite(image_path, frame)
    cap.release()


# E-posta gönderme
def send_email(subject, body, to_email, from_email, password, image_path):
    # E-posta mesajı oluşturma
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    # E-posta body ekleme
    msg.attach(MIMEText(body, 'plain'))

    # Fotoğraf ekleme
    with open(image_path, 'rb') as img_file:
        img = MIMEImage(img_file.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        msg.attach(img)

    # E-posta gönderme
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(from_email, password)
        server.send_message(msg)


# Anlık tarih ve saat bilgisi alma
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Kullanıcı bilgileri ve e-posta içeriği
subject = "Başlık"
body = f"Bilinmeyen Ziyaretçi. Gönderilme tarihi ve saati: {current_time}"
to_email = "omerbirinci3734@gmail.com"
from_email = "negzelrs@gmail.com"

# Daha önce Komut İstemicisinden Çevresel Değişkeni Tanımlamadıysan;
#    Çevresel Değişkeni Tanımlamadan Önce Uygulama Şifresini Edin.
#    (windows için) Komut İstemcinde 'setx EMAIL_PASSWORD "uygulama_sifresi" ' kodunu çalıştır.
#    Doğru Şekilde Tanımlandığından Emin Olduktan Sonra IDE'ni Kapatıp Aç.
#    Adımları Doğru Şekilde Uyguladığın Halde IDE'yi Kapatıp Açmazsan Çalışmayabilir.
password = os.getenv('EMAIL_PASSWORD')

# Fotoğrafı kaydetmek için dosya yolu
image_path = 'captured_image.jpg'

# Fotoğraf çekme
capture_image(image_path)

# E-posta gönderme
send_email(subject, body, to_email, from_email, password, image_path)

# Geçici dosyayı silme
os.remove(image_path)
