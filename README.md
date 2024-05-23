# Image-Classification
 This project come from a Data Science Bootcamp project

Description:

This repository implements a web-based image classification system for satellite imagery using pre-trained transformer models (Swin Transformer, ConvNeXt, and Vision Transformer) from the Hugging Face Transformers library. Users can upload satellite images and receive predictions about their categories through a user-friendly Streamlit interface.

Key Features:

    Transformer-based Classification: Leverages the power of transformer models for accurate and efficient image classification.
    Multi-Model Comparison: Provides insights into the performance of Swin, ConvNeXt, and ViT models on your specific satellite image dataset.
    Streamlit Integration: Offers a convenient and interactive web interface for image upload and classification prediction.
    Modular Structure: Facilitates code organization and future enhancements.

Getting Started:

    Prerequisites: Ensure you have Python 3.x, required libraries (transformers, streamlit, torch/tensorflow), and a GPU (recommended for better performance).
    Cloning the Repository: Use git clone https://github.com/Denis-Frizat/Satellite_Image_Classification.git to clone this repository.
    Installation: Navigate to the cloned directory and run pip install -r requirements.txt to install the necessary dependencies.
    Data Preparation: Place your pre-processed satellite image dataset (training, validation, and testing sets) in the appropriate directory structure specified in the code.
    Running the App: Execute streamlit run app.py to launch the Streamlit application.

Usage:

    Open your web browser and navigate to http://localhost:8501/ (Streamlit's default port).
    Click "Choose File" and select a satellite image for classification.
    Choose the desired pre-trained model (Swin, ConvNeXt, or ViT) from the dropdown menu.
    Click "Classify Image" to receive the predicted category for your image, along with confidence scores for each class (if applicable).

Project Structure:

    Img_class.ipynb: Contains the python code to follow. Importing dataset, preprocessing, fine-tune the model.
    app.py: Contains the core logic for Streamlit app creation, image loading, model selection, and prediction visualization.
    models: You will find the model used on this project on my HuggingFace Hub : https://huggingface.co/SolubleFish
    data: dataset come from https://madm.dfki.de/files/sentinel/EuroSAT.zip
    requirements.txt: Lists all necessary Python libraries for project execution.
    README.txt (This file): Provides project overview, usage instructions, and structure explanation.

Further Considerations:

    Documentation: Consider adding detailed comments or a separate documentation file to explain the code, data preparation, and model selection criteria.
    Customization: The Streamlit interface and model usage can be customized based on your preferences.
    Evaluation: If applicable, include code for evaluating model performance on your satellite image dataset.

License:

MIT license

Contributions:

Feel free to contribute to this project as you want.
