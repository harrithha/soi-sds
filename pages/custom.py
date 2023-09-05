from turtle import color
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as pgo
from matplotlib import pyplot as plt
import plotly.express as px


def app(data):
    col1, col2 = st.columns(2)
    x = ['tce_period', 'tce_time0bk', 'tce_impact', 'tce_duration', 'tce_depth',
         'tce_model_snr', 'tce_prad', 'tce_eqt', 'tce_steff', 'tce_slogg', 'tce_sradius']
    y = x.copy()
    z = x.copy()

    with col1:
        option_1 = st.selectbox(
            'X-Axis',
            (x), key=1)

        st.write('Selected:', option_1)

    with col2:
        y.remove(option_1)
        option_2 = st.selectbox(
            'Y-Axis',
            (y), key=2)

        st.write('Selected:', option_2)


    # defining surface and axes
    fig = px.scatter(data_frame=data, x=option_1, y=option_2, title='Scatter plot',color='predicted', width=700, height=700)
    fig.update_traces(marker_size = 2.5)
    st.plotly_chart(fig)

