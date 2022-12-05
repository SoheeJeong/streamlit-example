from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from prediction import predict
import PIL.Image as Image
import numpy as np

"""
# Predict your MBTI with your face!
"""

def main():
    # 만약 이미지를 업로드 했다면 원본 이미지를 업로드이미지로 설정, 아니라면 데모 이미지로 설정
    image_uploaded = st.file_uploader("Please upload your image:")
    if image_uploaded:
        image_origin = Image.open(image_uploaded)
    else:
        image_origin = Image.open('demo.jpg')
    image_origin = np.array(image_origin.convert('RGB'))
    st.image(image_origin)
    
    if st.button("Predict my MBTI"):
        result,image2,ypred = predict(image_origin)
        # result = "ISFJ"
        st.text(result)
        st.image(image2)
        st.text(ypred)

if __name__ == "__main__":
    main()