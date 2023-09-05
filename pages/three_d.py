import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as pgo
from matplotlib import pyplot as plt
import plotly.express as px


def app(data):
    col1, col2, col3 = st.columns(3)
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

    with col3:
        z.remove(option_1)
        z.remove(option_2)
        option_3 = st.selectbox(
            'Z-Axis',
            (z), key=3)

        st.write('Selected:', option_3)

    # defining surface and axes
    fig = px.scatter_3d(data_frame=data, x=option_1, y=option_2, z=option_3,
                        title='3D Plot', width=700, height=700, color='predicted')
    fig.update_traces(marker_size=1.5)
    st.plotly_chart(fig)
