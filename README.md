# Hand Gesture Recognition App

This application recognizes hand gestures and predicts whether they are fists or palms. It utilizes a machine learning model trained on images of fists and palms. By uploading an image, the app performs inference using the model and displays the predicted gesture.

## How It Works

- The uploaded image is processed and resized to a standardized input size.
- The pre-trained model is loaded, which has been trained on a dataset of hand gesture images.
- The processed image is fed into the model for prediction.
- The model outputs the predicted class, indicating whether it's a fist or a palm.
- The predicted result is displayed along with the confidence score.

## Background Model

The model used in this app is trained using TensorFlow and Keras based on Google's Teachable Machine approach. The dataset consists of labeled images of both fists and palms, allowing the model to learn and differentiate between the two gestures.

## Requirements

- Python 3.7 or later
- TensorFlow 2.6.0
- OpenCV 4.5.3.56
- Streamlit 1.22.0

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition-app.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the app in your web browser using the provided URL, you will seee the following webpage.
![image](https://github.com/project-pal/U7P2-ML-in-Streamlit-/assets/130244570/1ba40841-b339-4cad-a876-4e6dc8f05ad6)
6. Upload images of hands showing gestures (in JPG, JPEG, or PNG format), you can use the images in "Test images" folder
7. Click the "Predict" button to perform the gesture recognition.
8. The app will display the uploaded image along with the predicted gesture (fist or palm).

## Licences
This project is licensed under the MIT License. See the LICENSE file for more details.
