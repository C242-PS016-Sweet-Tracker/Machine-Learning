# Machine-Learning

## Features
- Food-scanner with fine tuned MobileV3 with Tensor Flow
- Nutrision Facts extracting using OCR
- Food recommendation
- Chatbot from Gemini (additional feature)

## Food-Scanner
This section demonstrates how to build an image classification model using **MobileNetV3** and fine-tune it on a custom food dataset for the **Food Scanner** application. Firstly, we have dataset for YOLOv8. The dataset have train, validation, and test directory. Each of them have two sub-directory, they are: images and labels.

Since our team make a CNN model that the input was already cropped to the only predicted food only, we cropped the dataset images with the labels to get specific image of the trained/predicted images, and then risize them to fit MobileNetv3 pre-trained model.

### Key Components:
1. **MobileNetV3**: A lightweight convolutional neural network model pre-trained on ImageNet. It is used as the backbone of the model for feature extraction.
2. **Fine-Tuning**: The pre-trained model is fine-tuned on the food dataset to adapt it to the specific classes in our dataset.
3. **Data Augmentation**: Random transformations like flip, rotation, and zoom are applied to the training data to prevent overfitting.
4. **Callbacks**: `EarlyStopping` and `ReduceLROnPlateau` are used during training to optimize performance.

## Nutrision Facts extracting using OCR

### Food recommendation

### Chatbot from Gemini (additional feature)
