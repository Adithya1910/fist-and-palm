import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the model and the class names
model = load_model("keras_Model.h5", compile=False)  # Load the pre-trained model
class_names = open("labels.txt", "r").readlines()  # Load the class names for predictions

# Define a function to predict signs
def predict_sign(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Perform the prediction
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, float(confidence_score)


# Display the title and description
st.markdown(
    "<h1 style='text-align: center; color: #FF4500; "
    "font-size: 48px; font-weight: bold; "
    "text-shadow: 2px 2px 4px #000000;'>"
    "Hand Gesture Recognition App"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='font-size: 18px; color: #1E90FF; text-align: center;'>"
    "This application recognizes hand gestures. Please upload images of hands showing gestures, "
    "and the app will predict whether they're fists or palms."
    "</p>",
    unsafe_allow_html=True
)


# Add a sidebar with information about the app
sidebar = st.sidebar
sidebar.header("About")
sidebar.markdown("This app uses a machine learning model to predict whether a hand gesture in an image is a fist or a palm.")
sidebar.markdown("The backend is a model developed using Google's Teachable Machine and trained on our own images of fists and palms.")

# File uploader
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Predict") and uploaded_files:
    cols = st.columns(2)
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file is not None:
            # Read and decode the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the image and perform the prediction
            with cols[i%2]:
                st.image(image_rgb, width=200, caption=f'Image {i+1}: {uploaded_file.name}')
                class_name, confidence_score = predict_sign(image)
                
                # Display the prediction result
                if class_name == '1 Fist':
                    st.markdown("<h3 style='text-align: center; color: blue;'>It's a Fist</h3>", unsafe_allow_html=True)
                elif class_name == '0 Palm':
                    st.markdown("<h3 style='text-align: center; color: green;'>It's a Palm</h3>", unsafe_allow_html=True)
    
    # Display a thank you message
    st.markdown("<h4 style='text-align: center; color: orange;'>Thanks for using our Hand Gesture Recognition App!</h4>", unsafe_allow_html=True)
