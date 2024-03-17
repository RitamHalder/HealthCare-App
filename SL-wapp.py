import streamlit as st
import requests
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import warnings
from tensorflow.keras.models import load_model

# Function to upload image for cancer detection
import io
warnings.filterwarnings("ignore")
# Disable the warning about Pyplot global use
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="Healthcare Predictions System",
    page_icon=":hospital:",
    layout="wide",
)

# Define the size of the logo
logo_size = (100, 100)

# Displaying the logo in the top-left corner
st.image("models/_78d73381-13d3-4f0b-98d4-7967c29b7d27.jpeg", width=logo_size[0], caption="Ange", use_column_width=False)

# Customizing Streamlit Page Style
def set_page_style():
    st.markdown(
        """
        <style>
            /* Overall page style */
            body {
                font-family: Arial, sans-serif;
                background-color: #222222; /* Dark background */
                color: #ffffff; /* Text color */
            }

            /* Streamlit main container */
            .stApp {
                max-width: 1200px; /* Wider container */
                margin: 0 auto;
                padding: 20px;
            }

            /* Streamlit header */
            .st-hv, .st-hn {
                background-color: #333333; /* Dark gray */
                color: white;
                padding: 10px 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }

            /* Streamlit input fields */
            .stTextInput>div>div>input {
                color: white !important;
                background-color: #444444 !important; /* Darker gray */
                border-radius: 5px;
                border: 1px solid #666666; /* Lighter gray */
                padding: 5px 8px;
                font-size: 10px;
                box-sizing: border-box;
                margin-bottom: 10px;
            }

            /* Streamlit buttons */
            .stButton>button {
                background-color: #4CAF50; /* Green */
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .stButton>button:hover {
                background-color: #45a049; /* Darker green on hover */
            }

            /* Streamlit subheader */
            .st-bs {
                background-color: #333333; /* Dark gray */
                color: white;
                padding: 10px 20px;
                margin-top: 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }

            /* Streamlit plots */
            .stPlotlyChart, .stDataFrame {
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                border-radius: 10px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )



def preprocess_input_image(image):
    image = cv2.resize(image, (224, 224))  # Resize image
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def upload_image():
    st.subheader("Cancer Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button("Predict Cancer"):
            prediction = predict_cancer_class(np.array(image))
            st.success(f"Cancer Prediction: {prediction}")
    else:
        st.warning("Please upload an image for cancer detection.")


def predict_cancer_class(image):
    # Load the trained model
    cancer_model = load_model('models/cancer_detection_model.keras')

    # Preprocess the input image
    input_image = preprocess_input_image(image)

    # Use the trained model to predict
    predictions = cancer_model.predict(input_image)

    # Interpret the prediction
    if predictions[0][0] > 0.5:
        return "Malignant"
    else:
        return "Benign"

def input_disease_features():
    st.subheader("Heart Disease Classification")
    
    # Add Streamlit UI components for input features
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.radio("Gender", ("Male", "Female"))
    imp = st.number_input("Impulse")
    pressure_high = st.number_input("High Blood Pressure")
    pressure_low = st.number_input("Low Blood Pressure")
    glucose = st.number_input("Glucose")
    kcm = st.number_input("KCM")
    troponin = st.number_input("Troponin")
    
    # Map gender to numerical value (0 for Male, 1 for Female)
    gender = 0 if gender == "Male" else 1
    
    # Return the input features as a dictionary
    return {
        "age": age,
        "gender": gender,
        "impluse": imp,
        "pressurehight": pressure_high,
        "pressurelow": pressure_low,
        "glucose": glucose,
        "kcm": kcm,
        "troponin": troponin
    }



def predict_disease(features):
    # Load the trained model
    disease_model = joblib.load('models/train_disease_classification_model.pkl')

    # Convert input features to DataFrame
    features_df = pd.DataFrame(features, index=[0])

    # Predict the class of the input features
    prediction = disease_model.predict(features_df)

    # Return the predicted disease class
    return prediction[0]


# Load the SVM model for breast cancer prediction
breast_cancer_model = joblib.load("models/breast_cancer.pkl")

# Function to predict breast cancer
def predict_breast_cancer(features):
    # Make prediction using the loaded model
    prediction = breast_cancer_model.predict(features)
    return prediction

# Function to input features for breast cancer prediction
def input_breast_cancer_features():
    st.subheader("Breast Cancer Prediction")
    st.write("Please input the following features for breast cancer prediction:")

    # Add Streamlit UI components for input features
    radius_mean = st.number_input("Radius Mean")
    texture_mean = st.number_input("Texture Mean")
    perimeter_mean = st.number_input("Perimeter Mean")
    area_mean = st.number_input("Area Mean")
    smoothness_mean = st.number_input("Smoothness Mean")
    compactness_mean = st.number_input("Compactness Mean")
    concavity_mean = st.number_input("Concavity Mean")
    concave_points_mean = st.number_input("Concave Points Mean")
    symmetry_mean = st.number_input("Symmetry Mean")
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean")
    radius_se = st.number_input("Radius SE")
    texture_se = st.number_input("Texture SE")
    perimeter_se = st.number_input("Perimeter SE")
    area_se = st.number_input("Area SE")
    smoothness_se = st.number_input("Smoothness SE")
    compactness_se = st.number_input("Compactness SE")
    concavity_se = st.number_input("Concavity SE")
    concave_points_se = st.number_input("Concave Points SE")
    symmetry_se = st.number_input("Symmetry SE")
    fractal_dimension_se = st.number_input("Fractal Dimension SE")
    radius_worst = st.number_input("Radius Worst")
    texture_worst = st.number_input("Texture Worst")
    perimeter_worst = st.number_input("Perimeter Worst")
    area_worst = st.number_input("Area Worst")
    smoothness_worst = st.number_input("Smoothness Worst")
    compactness_worst = st.number_input("Compactness Worst")
    concavity_worst = st.number_input("Concavity Worst")
    concave_points_worst = st.number_input("Concave Points Worst")
    symmetry_worst = st.number_input("Symmetry Worst")
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst")

    return {
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
        "concavity_mean": concavity_mean,
        "concave_points_mean": concave_points_mean,
        "symmetry_mean": symmetry_mean,
        "fractal_dimension_mean": fractal_dimension_mean,
        "radius_se": radius_se,
        "texture_se": texture_se,
        "perimeter_se": perimeter_se,
        "area_se": area_se,
        "smoothness_se": smoothness_se,
        "compactness_se": compactness_se,
        "concavity_se": concavity_se,
        "concave_points_se": concave_points_se,
        "symmetry_se": symmetry_se,
        "fractal_dimension_se": fractal_dimension_se,
        "radius_worst": radius_worst,
        "texture_worst": texture_worst,
        "perimeter_worst": perimeter_worst,
        "area_worst": area_worst,
        "smoothness_worst": smoothness_worst,
        "compactness_worst": compactness_worst,
        "concavity_worst": concavity_worst,
        "concave_points_worst": concave_points_worst,
        "symmetry_worst": symmetry_worst,
        "fractal_dimension_worst": fractal_dimension_worst
    }

@st.cache_data 
def load_breast_cancer_data(file_path):
    return pd.read_csv(file_path)

# Function to display breast cancer dataset and analysis
def display_breast_cancer_data():
    st.subheader("Breast Cancer Dataset Analysis")
    
    # Load breast cancer dataset
    breast_cancer_data = load_breast_cancer_data("app/data/breast-cancer.csv")
    
    # Perform analysis or visualizations
    # For example, display a pairplot of some relevant columns
    relevant_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
    pair_grid = sns.pairplot(breast_cancer_data[relevant_columns], hue='diagnosis')
    st.pyplot(pair_grid)


    
def main():
    set_page_style()
    st.title("Healthcare Predictions System")
    
    menu_selection = st.sidebar.selectbox("Menu", ("Cancer Detection", "Heart Disease Classification", "Breast Cancer Prediction"))
    
    if menu_selection == "Cancer Detection":
        upload_image()
    elif menu_selection == "Heart Disease Classification":
        features = input_disease_features()
        if st.button("Predict Disease"):
            prediction = predict_disease(features)
            if prediction == 1:
                st.success("Disease Prediction: Positive")
            else:
                st.success("Disease Prediction: Negative")
    elif menu_selection == "Breast Cancer Prediction":
        features = input_breast_cancer_features()
        if st.button("Predict Breast Cancer"):
            prediction = predict_breast_cancer([list(features.values())])
            st.success(f"Breast Cancer Prediction: {prediction}")
        display_breast_cancer_data()

if __name__ == "__main__":
    main()
