import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image 
from cli import run_pix2tex
# from unet.cli import runUnet

st.title('Learning to Parse PDFs to Latex')
st.subheader('Upload an image of a picture with LaTeX')


@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 


st.subheader("Home")
image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])

if image_file is not None:

    file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}

    img = load_image(image_file)
    
    st.write('Your image has been uploaded!')
    st.image(img)

    st.write('Solving...')

    # output, seg = runUnet(img)
    
    # print(output.dtype)

    

    # st.image(output/np.max(output))
    a = run_pix2tex(img)
    a = a.replace('\\', '\\\\')
    st.write(a)
    # print(a)
    st.write('Rendered!')
    # a = a.replace('~', ' ')
    # url = 'https://render.githubusercontent.com/render/math?math=' + a.replace(' ', '%')
    # st.image(url)
    # # st.write(url)
    # # print(a)