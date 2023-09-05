import pandas as pd
import numpy as np
import streamlit as st



header1=['NAME OF THE UNIVERSITY','QS rank','No.of FTE students','No.of students per staff','International students','Female:Male Ratio','Overall','Teaching','Research','Citations','Industry Income','International Outlook']
df=pd.read_excel('2022.xlsx',names=header1)

print(df)


st.write('<span><h1 style="color:purple"><center>UNIVERSITY RANKING ANALYSIS</center></h1></span>',unsafe_allow_html=True)
