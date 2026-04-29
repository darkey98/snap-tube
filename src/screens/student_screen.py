import streamlit as st
from src.ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
import numpy as np 
from PIL import Image

def student_screen():

    style_background_dashboard()
    style_base_layout()

    c1 , c2  = st.columns(2, vertical_alignment="center", gap="xxlarge")
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go Back To Home", type="secondary", key ='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type'] = None
            st.rerun()
    
 
    st.space()
    st.space()
    st.header("Login Using Face ID", text_alignment="center")
    photo_source = st.camera_input("Position Your Face In The Center")
    if photo_source:
        np.array(Image.open(photo_source))