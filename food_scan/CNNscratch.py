# Import Libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set Directories
TRAIN_DIR = 'C:/bangkit/code/dataset/train'
VALIDATION_DIR = 'C:/bangkit/code/dataset/validation'
TEST_DIR = 'C:/bangkit/code/dataset/test'

# Load Train and Validation Datasets with Data Augmentation
def train_val_datasets():
    train_class_names = ['ayam_train', 'apel_train', 'burger_train', 'pizza_train', 'nasigoreng_train', 'telur_train',
                         'tempe_train', 'nasiputih_train', 'kangkung_train', 'jeruk_train', 'tahu_train']
    valid_class_names = ['ayam_valid', 'apel_valid', 'burger_valid', 'pizza_valid', 'nasigoreng_valid', 'telur_valid',
                         'tempe_valid', 'nasiputih_valid', 'kangkung_valid', 'jeruk_valid', 'tahu_valid']

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        batch_size=64,
        image_size=(120, 120),
        label_mode='categorical',
        class_names=train_class_names
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=VALIDATION_DIR,
        batch_size=64,
        image_size=(120, 120),
        label_mode='categorical',
        class_names=valid_class_names
    )

    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

    return train_dataset, validation_dataset


train_dataset, validation_dataset = train_val_datasets()

# Define Model Architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(120, 120, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),  # Add dropout to reduce overfitting
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Add dropout for regularization
        tf.keras.layers.Dense(11, activation='softmax')  # Output for 10 classes
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# Add Early Stopping and Learning Rate Scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

# Train Model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# Plot Training History
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

plot_history(history)

# Evaluate on Test Dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=TEST_DIR,
    batch_size=64,
    image_size=(120, 120),
    label_mode='categorical',
    class_names=['apel', 'burger', 'pizza', 'nasigoreng', 'telur',
                 'tempe', 'nasiputih', 'kangkung', 'jeruk', 'tahu']
)

test_dataset = test_dataset.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
