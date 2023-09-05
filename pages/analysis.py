import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as pgo


def app(df):

    predicted_labels = ['AFP', 'PC', 'NTP']
    values = [len(df[df['predicted'] == 'AFP']), len(df[df['predicted'] == 'PC']), len(df[df['predicted'] == 'NTP'])]

    fig = pgo.Figure(
        pgo.Pie(
        labels = predicted_labels,
        values = values,
        hoverinfo = "label+percent",
        textinfo = "value"
    ))

    st.header("Prediction")
    st.plotly_chart(fig)
