from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.preprocessing import LabelEncoder

import numpy as np
import cv2
import os

# -----------------------------
# STEP 1: LOAD IMAGES
# -----------------------------
data = []
labels = []

dataset_path = "dataset"

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    # Skip if not a folder
    if not os.path.isdir(folder_path):
        continue

    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)

        image = cv2.imread(img_path)

        if image is not None:
            image = cv2.resize(image, (100, 100))
            data.append(image)
            labels.append(folder)

print("Total images loaded:", len(data))

# -----------------------------
# STEP 2: CONVERT DATA
# -----------------------------
data = np.array(data) / 255.0

le = LabelEncoder()
labels = le.fit_transform(labels)

# -----------------------------
# STEP 3: BUILD MODEL
# -----------------------------
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# STEP 4: TRAIN MODEL
# -----------------------------
model.fit(data, labels, epochs=10)

# -----------------------------
# STEP 5: SAVE MODEL
# -----------------------------
model.save("fabric_model.keras")
print("Model saved successfully!")

# -----------------------------
# STEP 6: TEST MODEL
# -----------------------------
test_path = os.path.join(os.getcwd(), "test.jpg")
test_img = cv2.imread(test_path)

if test_img is not None:
    test_img = cv2.resize(test_img, (100, 100))
    test_img = test_img / 255.0
    test_img = test_img.reshape(1, 100, 100, 3)

    prediction = model.predict(test_img)

    predicted_class = np.argmax(prediction)
    class_name = le.inverse_transform([predicted_class])

    print("Predicted Fabric:", class_name[0])
else:
    print("❌ Test image not found! Check file name and location.")