# mask_detector.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# 1 Set dataset directories
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# 2 Preprocess image data
image_size = (150, 150)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='binary')
val_data = val_datagen.flow_from_directory(val_dir, target_size=image_size, batch_size=batch_size, class_mode='binary')

# 3 Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 4 Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5 Train model
history = model.fit(train_data, epochs=5, validation_data=val_data)

# 6 Save model
model.save('mask_detector_model.h5')

print(" Model training complete and saved as 'mask_detector_model.h5'")
