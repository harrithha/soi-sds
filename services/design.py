import streamlit as st
import json
import pandas as pd


def refresh():
    pass


def show_format(file_dir):
    with open(file_dir, "r") as f:
        format = json.load(f)
    format_data = pd.DataFrame.from_dict(format)
    st.header("Format")
    st.markdown("### Please uplaod a CSV file in the following format")
    st.dataframe(format_data)
