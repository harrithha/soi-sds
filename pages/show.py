import streamlit as st
import pandas as pd


def left_align(s, props='text-align: left !important;'):
    return props


def bold(s, props="font-weight:bold !important"):
    return props


hide_table_row_index = """
        <style>
        tbody th {display:none;text-align:right !importnat;}
        .blank {display:none}
        .col_heading {text-align: left !important;font-weight:bold;color:black !important;}
        </style>
        """


def app(data):

    val_columns = ['Number Of Rows', 'Number Of Columns']
    number_Data = [[data.shape[0], data.shape[1]]]
    val_df = pd.DataFrame(number_Data, columns=val_columns)
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(val_df.style.applymap(left_align))
    st.write('**Columns present are**', data.columns.tolist())
    if data.shape[0] < 6:
        st.dataframe(data)
    else:
        columns = st.multiselect('Select Columns', data.columns.tolist(
        )+['All'], default=[data.columns.tolist()[0], data.columns.tolist()[1], data.columns.tolist()[-1]])

        if 'All' in columns:
            columns = data.columns.tolist()

        customized_data = data[columns]

        col1, col2 = st.columns(2)
        with col1:
            start = st.number_input(
                "Select Start Row", step=1, min_value=1, max_value=data.shape[0], value=1)
        with col2:
            end = st.number_input("Select End Row", step=1,
                                  min_value=5, max_value=data.shape[0], value=5)

        customized_data = customized_data.iloc[start-1:end]
        # if start!=end and len(columns)!=0:
        # index = [i for i in range(start,end+1)]
        # customized_data.insert(loc=0,column='',value=index)
        st.dataframe(customized_data.style.applymap(left_align))
        # st.table(customized_data.style.applymap(left_align))
