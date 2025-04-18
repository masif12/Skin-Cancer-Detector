import streamlit as st
from PIL import Image
import numpy as np
from utils.predict import predict_image

# Set the page layout
st.set_page_config(page_title="Skin Cancer Detection", page_icon="🧠", layout="centered")

# Title and description
st.title("Skin Cancer Detection App")
st.markdown("""
    Welcome to the **Skin Cancer Detection App**. 
    Upload an image of a skin lesion, and our model will predict whether it is benign or malignant.
    Let's get started!
""")

# Add an image uploader
uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "png", "jpeg"])

# If an image is uploaded
if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width =True)
    st.write("")
    
    # Run prediction on the image
    with st.spinner("Classifying the image..."):
        prediction = predict_image(image)
    
    # Show prediction result
    if prediction == "benign":
        st.success("The lesion is **Benign** 🟢")
    elif prediction == "malignant":
        st.error("The lesion is **Malignant** 🔴")
    else:
        st.warning("Could not classify the image. Please try again.")
    
    # Add a button for more details about the model
    st.markdown("""
        ### Model Information:
        - The model uses **Transfer Learning (ResNet50)**.
        - It's trained on a large dataset of skin images to detect cancer.
        - Accuracy: **92%** on validation data.
    """)

# Footer
st.markdown("""
    ----
    Created by Muhammad Asif. 
    For more information, check out the [GitHub Repository](https://github.com/masif12).
""")
