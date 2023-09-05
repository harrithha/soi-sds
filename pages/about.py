import streamlit as st
import webbrowser
from os import path


def app():
    url = path.abspath("components/about.html")
    if st.button('About Us Page'):
        webbrowser.open_new_tab(url)
