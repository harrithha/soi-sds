import pandas as pd
import numpy as np
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

import joblib
import streamlit as st


from streamlit import selectbox
from streamlit import number_input
import pickle

from st_aggrid import AgGrid
from pandas import option_context
from IPython.display import display,HTML

header1=['NAME OF THE UNIVERSITY','QS rank','No.of FTE students','No.of students per staff','International students','Female:Male Ratio','Overall','Teaching','Research','Citations','Industry Income','International Outlook']
df=pd.read_excel('2022.xlsx',names=header1)

print(df)


st.write('<span><h1 style="color:purple"><center>UNIVERSITY RANKING ANALYSIS</center></h1></span>',unsafe_allow_html=True)


col1=st.columns(2)

header1=['NAME OF THE UNIVERSITY','QS rank','No.of FTE students','No.of students per staff','International students','Female:Male Ratio','Overall','Teaching','Research','Citations','Industry Income','International Outlook']
df=pd.read_excel('2022.xlsx',names=header1)

df['Female:Male Ratio']=df['Female:Male Ratio'].apply(lambda x:str(x).replace("n/a"," "))
df['Female:Male Ratio']=df['Female:Male Ratio'].replace(r'^\s*$',np.nan,regex=True)
df['Female:Male Ratio']=df['Female:Male Ratio'].astype('float64') 
Female_Male_Ratio_mean=np.around(df['Female:Male Ratio'].mean(),1)
df['Female:Male Ratio']=df['Female:Male Ratio'].fillna(value=0.0)

df['Overall']=df['Overall'].apply(lambda x:str(x).replace("n/a"," "))
df['Overall']=df['Overall'].replace(r'^\s*$',np.nan,regex=True)
df['Overall']=df['Overall'].astype('float64') 
Overall_mean=np.around(df['Overall'].mean(),1)
df['Overall']=df['Overall'].fillna(value=0.00)

Teaching_mean=np.around(df['Teaching'].mean(),1)
df['Teaching']=df['Teaching'].fillna(value=0.0)

df['Research']=df['Research'].apply(lambda x:str(x).replace("n/a"," "))
df['Research']=df['Research'].replace(r'^\s*$',np.nan,regex=True)
df['Research']=df['Research'].astype('float64') 
Research_mean=np.around(df['Research'].mean(),1)
df['Research']=df['Research'].fillna(value=0.0)

df['Citations']=df['Citations'].apply(lambda x:str(x).replace("n/a"," "))
df['Citations']=df['Citations'].replace(r'^\s*$',np.nan,regex=True)
df['Citations']=df['Citations'].astype('float64')
Citations_mean=np.around(df['Citations'].mean(),1)
df['Citations']=df['Citations'].fillna(value=0.0)

df['Industry Income']=df['Industry Income'].apply(lambda x:str(x).replace("n/a"," "))
df['Industry Income']=df['Industry Income'].replace(r'^\s*$',np.nan,regex=True)
df['Industry Income']=df['Industry Income'].astype('float64')
IndustryIncome_mean=np.around(df['Industry Income'].mean(),1)
df['Industry Income']=df['Industry Income'].fillna(value=0.0)

df['International Outlook']=df['International Outlook'].apply(lambda x:str(x).replace("n/a"," "))
df['International Outlook']=df['International Outlook'].replace(r'^\s*$',np.nan,regex=True)
df['International Outlook']=df['International Outlook'].astype('float64')
International_outlook_mean=np.around(df['International Outlook'].mean(),1)
df['International Outlook']=df['International Outlook'].fillna(value=0.0)



df['NAME OF THE UNIVERSITY'].iloc[202:251]='202-251'
df['No.of FTE students'].iloc[202:251]=df['No.of FTE students'].iloc[202:251].median()
df['No.of students per staff'].iloc[202:251]=df['No.of students per staff'].iloc[202:251].median()
df['International students'].iloc[202:251]=df['International students'].iloc[202:251].median()
df['Female:Male Ratio'].iloc[202:251]=df['Female:Male Ratio'].iloc[202:251].median()
df['Overall'].iloc[202:251]=df['Overall'].iloc[202:251].median()
df['Teaching'].iloc[202:251]=df['Teaching'].iloc[202:251].median()
df['Research'].iloc[202:251]=df['Research'].iloc[202:251].median()
df['Citations'].iloc[202:251]=df['Citations'].iloc[202:251].median()
df['Industry Income'].iloc[202:251]=df['Industry Income'].iloc[202:251].median()
df['International Outlook'].iloc[202:251]=df['International Outlook'].iloc[202:251].median()

df['NAME OF THE UNIVERSITY'].iloc[251:300]='251-300'
df['No.of FTE students'].iloc[251:300]=df['No.of FTE students'].iloc[251:300].median()
df['No.of students per staff'].iloc[251:300]=df['No.of students per staff'].iloc[251:300].median()
df['International students'].iloc[251:300]=df['International students'].iloc[251:300].median()
df['Female:Male Ratio'].iloc[251:300]=df['Female:Male Ratio'].iloc[251:300].median()
df['Overall'].iloc[251:300]=df['Overall'].iloc[251:300].median()
df['Teaching'].iloc[251:300]=df['Teaching'].iloc[251:300].median()
df['Research'].iloc[251:300]=df['Research'].iloc[251:300].median()
df['Citations'].iloc[251:300]=df['Citations'].iloc[251:300].median()
df['Industry Income'].iloc[251:300]=df['Industry Income'].iloc[251:300].median()
df['International Outlook'].iloc[251:300]=df['International Outlook'].iloc[251:300].median()


df['NAME OF THE UNIVERSITY'].iloc[300:405]='300-405'
df['No.of FTE students'].iloc[300:405]=df['No.of FTE students'].iloc[300:405].median()
df['No.of students per staff'].iloc[300:405]=df['No.of students per staff'].iloc[300:405].median()
df['International students'].iloc[300:405]=df['International students'].iloc[300:405].median()
df['Female:Male Ratio'].iloc[300:405]=df['Female:Male Ratio'].iloc[300:405].median()
df['Overall'].iloc[300:405]=df['Overall'].iloc[300:405].median()
df['Teaching'].iloc[300:405]=df['Teaching'].iloc[300:405].median()
df['Research'].iloc[300:405]=df['Research'].iloc[300:405].median()
df['Citations'].iloc[300:405]=df['Citations'].iloc[300:405].median()
df['Industry Income'].iloc[300:405]=df['Industry Income'].iloc[300:405].median()
df['International Outlook'].iloc[300:405]=df['International Outlook'].iloc[300:405].median()


df['NAME OF THE UNIVERSITY'].iloc[405:502]='405-502'
df['No.of FTE students'].iloc[405:502]=df['No.of FTE students'].iloc[405:502].median()
df['No.of students per staff'].iloc[405:502]=df['No.of students per staff'].iloc[405:502].median()
df['International students'].iloc[405:502]=df['International students'].iloc[405:502].median()
df['Female:Male Ratio'].iloc[405:502]=df['Female:Male Ratio'].iloc[405:502].median()
df['Overall'].iloc[405:502]=df['Overall'].iloc[405:502].median()
df['Teaching'].iloc[405:502]=df['Teaching'].iloc[405:502].median()
df['Research'].iloc[405:502]=df['Research'].iloc[405:502].median()
df['Citations'].iloc[405:502]=df['Citations'].iloc[405:502].median()
df['Industry Income'].iloc[405:502]=df['Industry Income'].iloc[405:502].median()
df['International Outlook'].iloc[405:502]=df['International Outlook'].iloc[405:502].median()


df['NAME OF THE UNIVERSITY'].iloc[502:600]='502-600'
df['No.of FTE students'].iloc[502:600]=df['No.of FTE students'].iloc[502:600].median()
df['No.of students per staff'].iloc[502:600]=df['No.of students per staff'].iloc[502:600].median()
df['International students'].iloc[502:600]=df['International students'].iloc[502:600].median()
df['Female:Male Ratio'].iloc[502:600]=df['Female:Male Ratio'].iloc[502:600].median()
df['Overall'].iloc[502:600]=df['Overall'].iloc[502:600].median()
df['Teaching'].iloc[502:600]=df['Teaching'].iloc[502:600].median()
df['Research'].iloc[502:600]=df['Research'].iloc[502:600].median()
df['Citations'].iloc[502:600]=df['Citations'].iloc[502:600].median()
df['Industry Income'].iloc[502:600]=df['Industry Income'].iloc[502:600].median()
df['International Outlook'].iloc[502:600]=df['International Outlook'].iloc[502:600].median()


df['NAME OF THE UNIVERSITY'].iloc[600:800]='600-800'
df['No.of FTE students'].iloc[600:800]=df['No.of FTE students'].iloc[600:800].median()
df['No.of students per staff'].iloc[600:800]=df['No.of students per staff'].iloc[600:800].median()
df['International students'].iloc[600:800]=df['International students'].iloc[600:800].median()
df['Female:Male Ratio'].iloc[600:800]=df['Female:Male Ratio'].iloc[600:800].median()
df['Overall'].iloc[600:800]=df['Overall'].iloc[600:800].median()
df['Teaching'].iloc[600:800]=df['Teaching'].iloc[600:800].median()
df['Research'].iloc[600:800]=df['Research'].iloc[600:800].median()
df['Citations'].iloc[600:800]=df['Citations'].iloc[600:800].median()
df['Industry Income'].iloc[600:800]=df['Industry Income'].iloc[600:800].median()
df['International Outlook'].iloc[600:800]=df['International Outlook'].iloc[600:800].median()


df['NAME OF THE UNIVERSITY'].iloc[800:1002]='800-1002'
df['No.of FTE students'].iloc[800:1002]=df['No.of FTE students'].iloc[800:1002].median()
df['No.of students per staff'].iloc[800:1002]=df['No.of students per staff'].iloc[800:1002].median()
df['International students'].iloc[800:1002]=df['International students'].iloc[800:1002].median()
df['Female:Male Ratio'].iloc[800:1002]=df['Female:Male Ratio'].iloc[800:1002].median()
df['Overall'].iloc[800:1002]=df['Overall'].iloc[800:1002].median()
df['Teaching'].iloc[800:1002]=df['Teaching'].iloc[800:1002].median()
df['Research'].iloc[800:1002]=df['Research'].iloc[800:1002].median()
df['Citations'].iloc[800:1002]=df['Citations'].iloc[800:1002].median()
df['Industry Income'].iloc[800:1002]=df['Industry Income'].iloc[800:1002].median()
df['International Outlook'].iloc[800:1002]=df['International Outlook'].iloc[800:1002].median()


df['NAME OF THE UNIVERSITY'].iloc[1002:1201]='1002-1201'
df['No.of FTE students'].iloc[1002:1201]=df['No.of FTE students'].iloc[1002:1201].median()
df['No.of students per staff'].iloc[1002:1201]=df['No.of students per staff'].iloc[1002:1201].median()
df['International students'].iloc[1002:1201]=df['International students'].iloc[1002:1201].median()
df['Female:Male Ratio'].iloc[1002:1201]=df['Female:Male Ratio'].iloc[1002:1201].median()
df['Overall'].iloc[1002:1201]=df['Overall'].iloc[1002:1201].median()
df['Teaching'].iloc[1002:1201]=df['Teaching'].iloc[1002:1201].median()
df['Research'].iloc[1002:1201]=df['Research'].iloc[1002:1201].median()
df['Citations'].iloc[1002:1201]=df['Citations'].iloc[1002:1201].median()
df['Industry Income'].iloc[1002:1201]=df['Industry Income'].iloc[1002:1201].median()
df['International Outlook'].iloc[1002:1201]=df['International Outlook'].iloc[1002:1201].median()


df['NAME OF THE UNIVERSITY'].iloc[1201:1662]='1201-1662'
df['No.of FTE students'].iloc[1201:1662]=df['No.of FTE students'].iloc[1201:1662].median()
df['No.of students per staff'].iloc[1201:1662]=df['No.of students per staff'].iloc[1201:1662].median()
df['International students'].iloc[1201:1662]=df['International students'].iloc[1201:1662].median()
df['Female:Male Ratio'].iloc[1201:1662]=df['Female:Male Ratio'].iloc[1201:1662].median()
df['Overall'].iloc[1201:1662]=df['Overall'].iloc[1201:1662].median()
df['Teaching'].iloc[1201:1662]=df['Teaching'].iloc[1201:1662].median()
df['Research'].iloc[1201:1662]=df['Research'].iloc[1201:1662].median()
df['Citations'].iloc[1201:1662]=df['Citations'].iloc[1201:1662].median()
df['Industry Income'].iloc[1201:1662]=df['Industry Income'].iloc[1201:1662].median()
df['International Outlook'].iloc[1201:1662]=df['International Outlook'].iloc[1201:1662].median()


df.drop_duplicates(subset=None, keep='first', inplace=True)

print("hello")


education = selectbox("SELECT THE NAME OF THE UNIVERSITY",df)

i=0

while(df['NAME OF THE UNIVERSITY'].iloc[i]!=education):
    i+=1


if st.button('Predict'):

        st.write('<h5 style="color:blue"><center>INFORMATION REGARDING THE UNIVERSITY BASED ON THE USER SEARCH</center></h5>',unsafe_allow_html=True)

        pd.set_option('display.max_rows', None)

        # display all the  columns
        pd.set_option('display.max_columns', None)

        # set width  - 100
        pd.set_option('display.width', 100)

        

        x=df.loc[:,['International students','Teaching','Research','Citations','International Outlook']]
        y=df['QS rank']

       
        model = joblib.load('RFR_Model.pkl')
        prediction=model.predict(x)

        #print(np.floor(prediction[:10],0))
       
        print(df['QS rank'])
        print(prediction)
       
        df['QS rank'].iloc[:]=prediction[:]

        print(df['QS rank'].iloc[:200])

        print(df)

        pd.set_option('max_colwidth', 60)
        
        st.table(df.loc[i,:])
        #display(df.loc[i:i])
        #AgGrid(rf, height=500, fit_columns_on_grid_load=True)

        pd.set_option('display.max_rows', 200)

        # display all the  columns
        pd.set_option('display.max_columns', 12)

        # set width  - 100
        pd.set_option('display.width', 1000)

        st.write('<h5 style="color:red"><center>INFORMATION REGARDING ALL THE UNIVERSITIES LISTED IN THE SEARCH BAR</center></h5>',unsafe_allow_html=True)
        
        rf=st.dataframe(df)
        
        

        

        
        
        




        
       
       

        

      




