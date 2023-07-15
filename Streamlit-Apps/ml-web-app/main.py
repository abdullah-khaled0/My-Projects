import streamlit as st
from functions import *
import warnings
warnings.filterwarnings("ignore")


st.title('Data + ML + Streamlit = Power')

csv_file = st.file_uploader("Upload CSV", type=["csv"])

if st.button('Load Default Data'):
    csv_file = "Streamlit-Apps\ml-web-app\diabetes.csv"

if csv_file is not None:
    body(csv_file)
