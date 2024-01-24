import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

class_mapping = {
    0: 'Bacterial Pneumonia',
    1: 'Corona Virus Disease',
    2: 'Normal',
    3: 'Tuberculosis',
    4: 'Viral Pneumonia',
}



interpreter = tf.lite.Interpreter(model_content=open("lungs(90%).tflite", "rb").read())
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classify_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.astype(np.float32)

    # Print the shapes for debugging
    print("Original shape:", img_array.shape)

    # Add batch dimension to the image array
    img_array = np.expand_dims(img_array, axis=0)

    # Adjusting dimensions dynamically based on model input shape
    expected_shape = tuple(input_details[0]['shape'])
    expected_size = np.prod(expected_shape)

    if img_array.size != expected_size:
        print("Warning: Resizing image to match expected size.")
        img_array = np.resize(img_array, expected_shape)

    print("Reshaped shape:", img_array.shape)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output



st.title("Klasifikasi Gambar")
uploaded_file = st.file_uploader("Pilih gambar...")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("")
    st.write("Klasifikasi:")

    output = classify_image(image)

    predicted_class = np.argmax(output)
    predicted_label = class_mapping.get(predicted_class, 'Unknown')
    
    st.write(f"Kelas: {predicted_label}")

    st.image(image, caption="Gambar yang dipilih", use_column_width=True)
