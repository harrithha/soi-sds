import streamlit as st
import pandas as pd
from io import BytesIO


@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


@st.cache
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def download_block(df, file_name, title):
    st.markdown("### "+title)
    st.markdown("##### Set File name")
    csv = convert_df(df)
    df_xlsx = to_excel(df)
    name = st.text_input('*No extension', file_name)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("📥 Download CSV",
                           data=csv,
                           file_name=name+".csv")
    with col2:
        st.download_button(label='📥 Download Text',
                           data=csv,
                           file_name=name+".txt")
    with col3:
        st.download_button(label='📥 Download Excel',
                           data=df_xlsx,
                           file_name=name+".xlsx")


def app(df, name):
    file_name = name.split(".csv")[0]
    df2 = df.copy(deep=True)
    df2 = df2["predicted"]
    download_block(df2, file_name+"_predictions", "Download Only Predictions")
    download_block(df, file_name+"_predicted", "Download Data & Predictions")
