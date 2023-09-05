import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

header1=['NAME OF THE UNIVERSITY','QS rank','No.of FTE students','No.of students per staff','International students','Female:Male Ratio','Overall','Teaching','Research','Citations','Industry Income','International Outlook']
df=pd.read_excel('2022.xlsx',names=header1)

print(df)
print(df.shape) #total number of rows and columns


df.info()   #dtype and type of object(null or non-null)

print(df.describe())    #mean,median,std
print(df.nunique())     #number of unique values in each column

print("\n")

print((df.isna().sum()/df.shape[0])*100)    #Percentage of total number of empty/null values in each column

print("\n")
print("\n")

#print((df['Overall'].value_counts()/df['Overall'].shape[0])*100)  #distribution of values in this column


print("\n")
print("+++++++++++++++++++++")
print("CLEANING THE RAW DATA")
print("+++++++++++++++++++++")
print("\n")

print((df.isna().sum()/df.shape[0])*100)    #total number of empty/null values in each column


Teaching_mean=np.around(df['Teaching'].mean(),1)
df['Teaching']=df['Teaching'].fillna(value=Teaching_mean)

df['Research']=df['Research'].apply(lambda x:str(x).replace("n/a"," "))
df['Research']=df['Research'].replace(r'^\s*$',np.nan,regex=True)
df['Research']=df['Research'].astype('float64') 
Research_mean=np.around(df['Research'].mean(),1)
df['Research']=df['Research'].fillna(value=Research_mean)

df['Citations']=df['Citations'].apply(lambda x:str(x).replace("n/a"," "))
df['Citations']=df['Citations'].replace(r'^\s*$',np.nan,regex=True)
df['Citations']=df['Citations'].astype('float64')
Citations_mean=np.around(df['Citations'].mean(),1)
df['Citations']=df['Citations'].fillna(value=Citations_mean)

df['Industry Income']=df['Industry Income'].apply(lambda x:str(x).replace("n/a"," "))
df['Industry Income']=df['Industry Income'].replace(r'^\s*$',np.nan,regex=True)
df['Industry Income']=df['Industry Income'].astype('float64')
IndustryIncome_mean=np.around(df['Industry Income'].mean(),1)
df['Industry Income']=df['Industry Income'].fillna(value=IndustryIncome_mean)

df['International Outlook']=df['International Outlook'].apply(lambda x:str(x).replace("n/a"," "))
df['International Outlook']=df['International Outlook'].replace(r'^\s*$',np.nan,regex=True)
df['International Outlook']=df['International Outlook'].astype('float64')
International_outlook_mean=np.around(df['International Outlook'].mean(),1)
df['International Outlook']=df['International Outlook'].fillna(value=International_outlook_mean)


print("\n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("CHECKING IF ALL THE DATA EXCEPT OVERALL AND FEMALE:MALE RATIO HAS BEEN CLEANED")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("\n")

print(df.isna().sum()/df.shape[0]*100)

print("\n")
print("++++++++++++++++++")
print("DATA VISUALISATION")
print("++++++++++++++++++")
print("\n")

print("+++++++")
print("MATPLOT")
print("+++++++")

top10=df.head(10)
top10_f=top10.loc[:,['QS rank','International students','Overall','Teaching','Research','Citations','Industry Income','International Outlook']]

print(top10_f)

plt.figure(figsize=(15,6))
plt.plot(top10_f,label=['QS rank','International students','Overall','Teaching','Research','Citations','Industry Income','International Outlook'])
plt.legend(loc='lower right')

plt.title('Top 10 Universities')
plt.show()

top100=df.head(100)
top100_f=top100.loc[:,['QS rank','International students','Overall','Teaching','Research','Citations','Industry Income','International Outlook']]

print(top10_f)

plt.figure(figsize=(15,6))
plt.plot(top100_f,label=['QS rank','International students','Overall','Teaching','Research','Citations','Industry Income','International Outlook'])
plt.legend(loc='lower right')

plt.title('Top 100 Universities')
plt.show()

#Scatter plot,word cloud and pie chart--showing which country has maximum number of universities

print("\n")
print("++++++++++++++++++++")
print("BOX AND WHISKER PLOT")
print("++++++++++++++++++++")
print("\n")

f,ax=plt.subplots(1,5,figsize=(50,5))

sns.boxplot(data=df['QS rank'],ax=ax[0],color='hotpink')
ax[0].set_xlabel('QS rank')

sns.boxplot(data=df['No.of FTE students'],ax=ax[1],color='indigo')
ax[1].set_xlabel('No. of FTE students')

sns.boxplot(data=df['No.of students per staff'],ax=ax[2],color='grey')
ax[2].set_xlabel('No.of students per staff')

sns.boxplot(data=df['International students'],ax=ax[3],color='magenta')
ax[3].set_xlabel('International students')

sns.boxplot(data=df['Overall'],ax=ax[4],color='violet')
ax[4].set_xlabel('Overall')

f,ax=plt.subplots(1,5,figsize=(50,5))

sns.boxplot(data=df['Teaching'],ax=ax[0],color='red')
ax[0].set_xlabel('Teaching')

sns.boxplot(df['Research'],ax=ax[1],color='yellow')
ax[1].set_xlabel('Research')

sns.boxplot(data=df['Citations'],ax=ax[2],color='orange')
ax[2].set_xlabel('Citations')

sns.boxplot(df['Industry Income'],ax=ax[3],color='green')
ax[3].set_xlabel('Industry Income')

sns.boxplot(df['International Outlook'],ax=ax[4],color='blue')
ax[4].set_xlabel('International Outlook')

plt.show()

print("\n")
print("+++++++++++++")
print("+++++++++++++")
print("DATA MODELING")
print("+++++++++++++")
print("+++++++++++++")
print("\n")

print("+++++++++++++++++")
print("LINEAR REGRESSION")
print("+++++++++++++++++")

x=df.drop(['NAME OF THE UNIVERSITY','QS rank','Female:Male Ratio'],axis = 1)
y=df['QS rank']

# Splitting the dataset into training and test set

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=0) 

#Fitting the Simple Linear Regression model to the training dataset  
  
LR=LinearRegression()  
LR.fit(x_train, y_train)  
print(LR.score(x_test,y_test))

y_pred=LR.predict(x_test)

print(y_test.head(10))

print(y_pred[:10])

XGB=XGBRegressor()
XGB.fit(x_train, y_train)  
print(XGB.score(x_test,y_test))

y_pred=XGB.predict(x_test)

print(y_test.head(10))

print(y_pred[:10])

RFR=RandomForestRegressor()
RFR.fit(x_train, y_train)  
print(RFR.score(x_test,y_test))

y_pred=RFR.predict(x_test)

print(y_test.head(10))

print(y_pred[:10])




















