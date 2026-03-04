import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# DATASET PATH
# =========================
dataset_path = r'C:\Users\alkes\OneDrive\Desktop\shoe detection'

# =========================
# DATA GENERATOR
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)

train_generator = train_datagen.flow_from_directory(
    dataset_path + "\\train",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path + "\\train",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# =========================
# LOAD PRETRAINED MODEL
# =========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False   # Freeze base model

# =========================
# ADD CUSTOM CLASSIFIER
# =========================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop]
)

# =========================
# TEST SINGLE IMAGE
# =========================
img_path = r'C:\Users\alkes\OneDrive\Desktop\shoe detection\test\adidas\7.jpg'

class_labels = ['adidas','converse','nike']

img = load_img(img_path, target_size=(224,224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)
predicted_index = np.argmax(prediction, axis=1)[0]
predicted_label = class_labels[predicted_index]
confidence = np.max(prediction) * 100

print(f'\nPredicted: {predicted_label}')
print(f'Confidence: {confidence:.2f}%')

plt.imshow(img)
plt.title(f'Prediction: {predicted_label} ({confidence:.2f}%)')
plt.axis('off')
plt.savefig('prediction_result.png')

print('Prediction image saved as prediction_result.png')