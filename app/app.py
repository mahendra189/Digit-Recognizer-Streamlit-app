import streamlit as st 
from streamlit_drawable_canvas import st_canvas
import numpy as np 
import cv2
import pickle

def predict_image(img):
    loaded_model = pickle.load(open('knn_classifier.pkl', 'rb'))
    results=loaded_model.predict(img.reshape(1,784))
    return results[0]

def image_transform(img):
    #Resize the canvas image into 28x28 for prediction
    img=cv2.resize(image.image_data.astype(np.uint8),(28,28))
    x_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    img_rescalling= (cv2.resize(img, dsize=(300,300),interpolation=cv2.INTER_NEAREST))
    return x_img,img_rescalling

st.title("Digit Recognizer")
st.write("Write digit in the worst handwritting and see whether the system can identify the right digit or not. ")

stroke_width = st.sidebar.slider("Brush Width",1,100,10)

col1,col2 = st.beta_columns(2)
with col1:
    col1.header("Write any digit")
    image = st_canvas(
            height=300,
            width=300,
            stroke_width=stroke_width,
            stroke_color='#ffffff',
            background_color='#000000',
            drawing_mode='freedraw'
            )
    predict_digit = st.button(label='Predict')

if image.image_data is not None and predict_digit == True:
    img_data,img_rescalling = image_transform(image.image_data)
    predicted_digit = predict_image(img_data)

    with col2:
        col2.header("Predicted Digit")
        st.image(img_rescalling)
        st.write("predicted_digit")
        st.write(predicted_digit)
    