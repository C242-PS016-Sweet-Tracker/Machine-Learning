import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# Load the saved model (replace 'your_model_path' with the path to your saved model)
model = tf.keras.models.load_model('C:/bangkit/code/model_makanan_keras.keras')

# Define class names corresponding to your model output (assuming 'burger' is in position 3)
class_names = ['ayam', 'burger', 'jeruk', 'kangkung', 'nasgor', 'nasi', 'pizza', 'tahu', 'telur', 'tempe']

image_size = (224, 224)

# Function to preprocess the image and make a prediction
def predict_image(img_path):
    # Load image and resize to (224, 224)
    img = image.load_img(img_path, target_size=image_size)

    # Convert the image to a numpy array and preprocess it
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for MobileNetV3

    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=-1)[0]

    # Get the predicted class label
    predicted_class_name = class_names[predicted_class]

    # Print the prediction and its probability
    print(f"Predicted Class: {predicted_class_name} (Class Index: {predicted_class})")
    print(f"Prediction Probabilities: {predictions}")

    # Optionally, show the image with the predicted label
    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_class_name}")
    plt.show()

# Example: Predict an image from your test directory
img_path = 'C:/bangkit/code/dataset/testing/jeruk1.png'  # Replace with the path to the image you want to test
predict_image(img_path)

