# Machine-Learning

## Features
- Food-scanner with fine tuned MobileV3 with Tensor Flow
- Nutrition Facts extracting using OCR
- Food recommendation
- Chatbot from Gemini (additional feature)

# Food-Scanner
This section demonstrates how to build an image classification model using **MobileNetV3** and fine-tune it on a custom food dataset for the **Food Scanner** application. Firstly, we have dataset for YOLOv8. The dataset have train, validation, and test directory. Each of them have two sub-directory, they are: images and labels.

Since our team make a CNN model that the input was already cropped to the only predicted food only, we cropped the dataset images with the labels to get specific image of the trained/predicted images, and then risize them to fit MobileNetv3 pre-trained model.

### Collecting Data
We collected Indonesia's common food photos from Roboflow. The data that we collected consist label for YOLO model, so later we cropped the photos by the label to be used. There are three directories in each classes, they are train, validation, and test directories. For the glycemic index we got data from Kaggle.

The link for dataset source:
#### Food glycemic index:
- https://www.kaggle.com/datasets/nandagopll/food-suitable-for-diabetes-and-blood-pressure 
#### Food photos and labels:
- https://universe.roboflow.com/objectdetection-3cl66/tahuxx_goreng/dataset/1 
- https://universe.roboflow.com/objectdetection-3cl66/tmpex_goreng/dataset/1 
- https://universe.roboflow.com/kenzi-lamberto/ayam-tt/dataset/1 
- https://docs.google.com/document/d/1Y46iMA792f-V7w7yvjFmxzLSfzmF34vUmTRCvwGOsSI/edit?tab=t.0  

### Exploring and Preprocessing Data
From the food photos we collected, we realized that the photo's quality was quiet bad for training because having a lot of noise from the background of the food in the photos. Since we have the label for YOLO training, we cropped the the photos with the label so we got clean photos without background (only the food without plate or something else).

We do our preprocessing with this notebook: https://colab.research.google.com/drive/1OlJxIORlhJHMsRh3QpK4ANS3xfSuQmBO?usp=sharing

Before processed: https://drive.google.com/drive/folders/1AMHwHbZSI5TlbBvmy2q0MIP6ufZkLD6x?usp=drive_link

Processed dataset can be found on this repository directory namely `dataset(processed)`.

### Training
#### Key Components:
1. **MobileNetV3**: A lightweight convolutional neural network model pre-trained on ImageNet. It is used as the backbone of the model for feature extraction.
   from tensorflow.keras.applications import MobileNetV3Small
   ```
    def create_model():
      # Load pre-trained MobileNetV3 without the top classification layer
      base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
      # ...
      return model
   ```
3. **Fine-Tuning**: The pre-trained model is fine-tuned on the food dataset to adapt it to the specific classes in our dataset.
   ```
    def create_model():
      # Load pre-trained MobileNetV3 ...
      base_model.trainable = True  # Unfreeze base model layers for fine-tuning
      # ...
      return model
   ```
5. **Data Augmentation**: Random transformations like flip, rotation, and zoom are applied to the training data to prevent overfitting.
   ```
      def train_val_datasets():
          # ... (dataset creation)
          data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1)
          ])
          # ... (apply augmentation to training data)
          return train_dataset, validation_dataset
   ```
7. **Callbacks**: `EarlyStopping` and `ReduceLROnPlateau` are used during training to optimize performance.
   ```
     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
      
     history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        callbacks=[early_stopping, reduce_lr]
     )
   ```
9. **Store** the model into `.keras`.
    ```
    model.save('model_makanan_keras.keras')
    ```
### Imported Library
1. `os`
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0': This line sets an environment variable to disable specific optimizations in TensorFlow. This can be useful for debugging or when encountering performance issues.
2. `numpy`
    - Used for various numerical computations, such as array operations, statistical calculations, and more.
3. `tensorflow`
    - Defines and trains the neural network model.   
    - Prepares and processes the image data.   
    - Evaluates the model's performance.
4. `matplotlib.pyplot`
    - Visualizes the training and validation accuracy and loss curves.
5. `tensorflow.keras.callbacks`
    - Implements EarlyStopping to halt training if the validation loss doesn't improve.
    - Implements ReduceLROnPlateau to reduce the learning rate when validation loss plateaus.
6. `tensorflow.keras.applications`
    - Loads the MobileNetV3 model as the base model for feature extraction.
7. `tensorflow.keras.applications.mobilenet_v3`
    - Provides the preprocess_input function to normalize the input images.
8. `sklearn.utils.class_weight`
    - Computes class weights to balance the impact of different classes during training.

# Nutrition Facts extracting using OCR

# Food recommendation

# Chatbot from Gemini (additional feature)
We are using Gemini as chatbot for only additional feature, the API and the feature made by our team's Mobile Development and Cloud Computing division.
