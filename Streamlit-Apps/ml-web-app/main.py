import streamlit as st
from functions import *
import warnings
warnings.filterwarnings("ignore")


st.title('Data + ML + Streamlit + my intelligence🙃 = Power')

csv_file = st.file_uploader("Upload CSV", type=["csv"])

if csv_file is not None:
    body(csv_file)