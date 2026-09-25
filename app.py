import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page title
st.title("👕 Fashion MNIST CNN Classifier")

st.write("Upload a Fashion MNIST clothing image and the CNN will predict its class.")

# Load trained model
model = tf.keras.models.load_model("fashion_mnist_cnn.keras")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

# Fashion MNIST classes
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=200)

    # Resize
    image = image.resize((28, 28))

    # Convert to NumPy
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Add channel dimension
    image_array = np.expand_dims(image_array, axis=-1)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    predicted_class = np.argmax(prediction)

    st.success(
        f"Prediction: {class_names[predicted_class]}"
    )
