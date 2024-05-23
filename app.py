import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO


# Title
st.title("Image Classification Web App")
st.markdown("This app uses Hugging Face's 'transformers' library to classify images using pre-trained models. The app uses three different models for image classification: swin, convnext and vit. Please select a model to classify the image you put on the left sidebar.")

# Intro
st.sidebar.markdown("**Please provide a Satellite image for classification**")

# Image input via URL
url = st.sidebar.text_input("Image URL")
if url:
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)
    except Exception as e:
        st.sidebar.error("Invalid URL. Please enter a valid URL for an image.")

# Image input via file uploader on the sidebar (but displays image on the main page)
uploaded_file = st.sidebar.file_uploader("Or upload an image", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Documentation about the 3 models
st.sidebar.markdown("## Find more information about the model architecture at the link below :  ")
st.sidebar.markdown("*Vision Transformer (ViT)* https://huggingface.co/docs/transformers/main/en/model_doc/vit")
st.sidebar.markdown("*ConvNext Transformer* https://huggingface.co/docs/transformers/main/en/model_doc/convnext")
st.sidebar.markdown("*Swin Transformer* https://huggingface.co/docs/transformers/main/en/model_doc/swin")

# Image classification function
access_token = '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'
def classify_image1(image):
    pipe1 = pipeline("image-classification", "SolubleFish/swin_transformer-finetuned-eurosat", token=access_token)
    return pipe1(image)
def classify_image2(image):
    pipe2 = pipeline("image-classification", "SolubleFish/image_classification_convnext", token=access_token)
    return pipe2(image)
def classify_image3(image):
    pipe3 = pipeline("image-classification", "SolubleFish/image_classification_vit", token=access_token)
    return pipe3(image)


# Create three columns
col1, col2, col3 = st.columns(3)

# Classification button for classify_image1
if col1.button("Classify Image by Swin"):
    if url or uploaded_file:
        results = classify_image1(image)
        if results:
            # Use markdown to present the results
            for result in results:
                col1.markdown(f"Class name: **{result['label']}** \n\n Confidence: **{str(format(result['score']*100, '.2f'))}**"+"%")
            col1.success("Classification completed.")
        else:
            col1.error("No results found.")
    else:
        col1.error("Please provide an image for classification.")

# Classification button for classify_image2
if col2.button("Classify Image by ConvNext"):
    if url or uploaded_file:
        results = classify_image2(image)
        if results:
            # Use markdown to present the results
            for result in results:
                col2.markdown(f"Class name: **{result['label']}** \n\n Confidence: **{str(format(result['score']*100, '.2f'))}**"+"%")
            col2.success("Classification completed.")
        else:
            col2.error("No results found.")
    else:
        col2.error("Please provide an image for classification.")

# Classification button for classify_image3
if col3.button("Classify Image by ViT"):
    if url or uploaded_file:
        results = classify_image3(image)
        if results:
            # Use markdown to present the results
            for result in results:
                col3.markdown(f"Class name: **{result['label']}** \n\n Confidence: **{str(format(result['score']*100, '.2f'))}**"+"%")
            col3.success("Classification completed.")
        else:
            col3.error("No results found.")
    else:
        col3.error("Please provide an image for classification.")
