import streamlit as st


def app():
    st.markdown("""
    ### Documentation

    #### Technologies Used

    ##### 1. Streamlit

    This web application is created using streamlit. Streamlit is an open source app framework in Python language. 
    It helps us create web apps for data science and machine learning in a short time. It is compatible with major 
    Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.

    Repository: https://github.com/streamlit/streamlit \n
    Website: https://streamlit.io/

    ##### 2. Plotly

    Plotly is free and open-source software, licensed under the MIT license. Plotly enables Python users to create 
    beautiful interactive web-based visualizations that can be displayed in Jupyter notebooks, saved to standalone 
    HTML files, or served as part of pure Python-built web applications using Dash. 

    Repository: https://github.com/plotly \n
    Website: https://plotly.com/

    ##### 3. Multi-page App

    A simple Framework for Creating Truly Multipage Streamlit apps. It is shared under MIT license.

    Repository: https://github.com/akshanshkmr/streamlit-multipage

    ### Using our Web-App

    Using the app is intuitive and straight-forward. A short video as shown below explains the functionalities 
    of our application in brief.

    """)
    video_file = open("components/graphics/video.mp4", "rb").read()
    st.video(video_file, start_time=0)
