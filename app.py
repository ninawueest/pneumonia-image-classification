import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
 
# Load your custom regression model
model_path = "pneumonia_model_tl.keras"
model = tf.keras.models.load_model(model_path)
 
labels = ['Normal', 'Pneumonia']
 
# Define regression function
def predict_regression(image):
    # Preprocess image
    image = Image.fromarray(image.astype('uint8'))  # Convert numpy array to PIL image
    image = image.resize((150, 150)).convert('RGB')  # Resize the image to 150x150 and convert to RGB
    image = np.array(image)
    print(image.shape)
    # Predict
    prediction = model.predict(image[None, ...])  # Assuming single regression value
    confidences = {labels[i]: np.round(float(prediction[0][i]), 2) for i in range(len(labels))}
    return confidences
 
# Create Gradio interface
input_image = gr.Image()
output_text = gr.Textbox(label="Predicted Value")
interface = gr.Interface(fn=predict_regression, 
                         inputs=input_image, 
                         outputs=gr.Label(),
                         examples=["Normal_1.jpeg","Pneumonia_1.jpeg"],   
                         description="A simple MLP classification model for image classification using the x-ray normal lungs vs with pneumonia dataset.")
interface.launch()