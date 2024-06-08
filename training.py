import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
embedding_layer = Dense(128, name='embedding')(x)
embedding_model = Model(inputs=base_model.input, outputs=embedding_layer)

image_folder = "images"
embeddings = []
labels = []

for image_name in os.listdir(image_folder):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_image = cv2.resize(image_rgb, (224, 224))
            face_image = np.expand_dims(face_image, axis=0)
            face_image = preprocess_input(face_image)

            face_embedding = embedding_model.predict(face_image).flatten()

            embeddings.append(face_embedding)

            label = image_name.split('_')[0]
            labels.append(label)
            print(f"Image: {image_name}, Label: {label}")  # Etiketleri kontrol etme
        else:
            print(f"Warning: No face landmarks found in image {image_path}")

if len(embeddings) == 0:
    raise ValueError("Error: No images were processed. Please check the image folder and face detection steps.")

embeddings = np.array(embeddings)
labels = np.array(labels)

le = LabelEncoder()
labels_enc = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels_enc, test_size=0.2, random_state=42)

input_shape = embeddings.shape[1]
classifier_input = Input(shape=(input_shape,))
x = Dense(128, activation='relu')(classifier_input)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
output_layer = Dense(len(le.classes_), activation='softmax')(x)
classifier_model = Model(inputs=classifier_input, outputs=output_layer)

classifier_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = classifier_model.fit(X_train, y_train,
                               epochs=50,
                               batch_size=32,
                               validation_data=(X_test, y_test),
                               callbacks=[early_stopping])

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

embedding_model.save('face_embedding_model.keras')
classifier_model.save('classifier_model.keras')

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model ve etiketler başarıyla kaydedildi.")
