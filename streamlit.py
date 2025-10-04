import streamlit as st
from fastai.vision.all import *


# Load the pre-trained breed model
breed_model = load_learner("cat_dog_breed_model.pkl")


# Define prediction function
def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = breed_model.predict(img)
    return pred_class  # For breed classification, just return the predicted class


# Streamlit UI
st.title("Pet Breed Classifier")
st.text("Built by Joel Suwanto")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Prediction
    prediction = predict(uploaded_file)
    st.subheader(f"Predicted Breed: {prediction}")

# Footer
st.text("Built with Streamlit and Fastai")
