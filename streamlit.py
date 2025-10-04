import streamlit as st
from fastai.vision.all import *


def extract_breed_name(file_name):
    p = Path(file_name)
    breed_name_parts = p.stem.split("_")
    final_breed_name = ""
    # breed_name_parts.pop()
    length_parts = len(breed_name_parts) - 1
    for i in range(length_parts):
        final_breed_name += breed_name_parts[i]
        if i != length_parts - 1:
            final_breed_name += "_"

    return final_breed_name

breed_model = load_learner("cat_dog_breed_model.pkl")


def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = breed_model.predict(img)
    return pred_class


# Streamlit UI
st.title("Pet Breed Classifier")
st.text("Built by Joel Suwanto")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    prediction = predict(uploaded_file)
    st.subheader(f"Predicted Breed: {prediction}")

# Footer
st.text("Built with Streamlit and Fastai")
