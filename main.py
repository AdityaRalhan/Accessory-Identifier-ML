import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import base64

# Function to convert an image to base64
def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load the background image
background_image = load_image('background.jpeg')  # Change to your image file name

# Load the trained model
model = tf.keras.models.load_model('trained_model_mnist.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# IMAGE PREPROCESSING TO ACCEPT ANY INPUT IMAGE
def process_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to gray scale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))  # Allow only one image
    return img_array

# Add custom CSS styles
st.markdown(f"""
    <style>
        body {{
            background-color: #f0f8ff;  /* Light blue background color */
            font-family: 'Arial', sans-serif;
            margin: 0;  /* Remove default margin */
            padding: 0;  /* Remove default padding */
            height: 100vh;  /* Ensure full viewport height */
            display: flex;
            flex-direction: column;
        }}
            
        [data-testid="stApp"] {{
            background-color: #da81d4;
opacity: 0.8;
background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #da81d4 26px ), repeating-linear-gradient( #e7ee4955, #e7ee49 );
            }}

        [data-testid="stHeader"]{{
            background-color: #da81d4;
opacity: 0.8;
background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #da81d4 26px ), repeating-linear-gradient( #e7ee49 ,#e7ee4955 );
        }}
            
        [data-testid="stBaseButton-secondary"]: hover{{
            background-color: blue;
        }}
        [data-testid="stMarkdownContainer"] p {{
            color: black;
            font-size: 18px
        }}
            
        [data-testid="stMarkdownContainer"] {{
            margin-left: 20px
        }}
            
        .st-emotion-cache-ocqkz7 {{
        display: flex;
        flex-wrap: wrap;
        -webkit-box-flex: 1;
        flex-grow: 1;
        -webkit-box-align: stretch;
        align-items: stretch;
        gap: 2rem;
            
        }}
            
        .st-emotion-cache-15hul6a {{
        display: inline-flex;
        -webkit-box-align: center;
        align-items: center;
        -webkit-box-pack: center;
        justify-content: center;
        font-weight: 400;
        border-radius: 0.5rem;
        min-height: 2.5rem;
        margin: 0px;
        line-height: 1.6;
        color: black;
        width: auto;
        user-select: none;
        background-color: pink;
        border: 1px solid grey;
        }}

        .custom-header {{
            background-image: url(data:image/jpeg;base64,{background_image}); /* Use base64 image */
            padding-top: 320px;
            text-align: center;
            border-radius: 10px;
            background-size: cover; /* Cover the entire header */
            height: 500px;
            
           
        }}


        .st-emotion-cache-13ln4jf {{
        width: 100%;
        padding: 4rem 1rem 10rem;
        max-width: 46rem;
}}
        .custom-header h1{{
            font-family: 'Verdana';
            font-size: 2.5em; /* Adjust size as needed */
        }}
        
        
        .gap {{
            height:50px;
            background-color: rgba(255, 255, 255, 0.0);
 
        }}
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Custom header using HTML
    st.markdown("""
        <div class="custom-header">
            <h1>Welcome to Fashion <3</h1>
        </div>
    """, unsafe_allow_html=True)

with col2:

    
    # Continue with your Streamlit app logic...
    st.write("Upload an image to find output from the given image.")
    st.error("PLEASE USE 28 * 28 greyscale image as this app does not support normal images due to limited training data.")

    # st.markdown('<div class="gap">', unsafe_allow_html=True)  # Opening wrapper div
    # st.markdown('</div>', unsafe_allow_html=True)  # Closing wrapper div
    uploaded_img = st.file_uploader("Upload image ", type=['jpg', 'jpeg', 'png'])

    # Wrapping the image upload and prediction button in a div
    
    if uploaded_img is not None:
        image = Image.open(uploaded_img)
        # Structure of webapp
        col3, col4, col5 = st.columns(3)  # st.columns(number of cols)

        with col3:
            resized_img = image.resize((100, 100))
            st.image(resized_img)

        with col4:
            if st.button('Classify', key='custom-button'):
                img_array = process_image(uploaded_img)

                result = model.predict(img_array)

                predicted_class = np.argmax(result)
                prediction = class_names[predicted_class]

                st.success(f"Prediction is {prediction}")

    
