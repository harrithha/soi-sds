import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

from xgboost import XGBRegressor

import joblib
import pickle




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


print("\n")
print("+++++++++++++++++++++")
print("+++++++++++++++++++++")
print("CLEANING THE RAW DATA")
print("+++++++++++++++++++++")
print("+++++++++++++++++++++")
print("\n")

print((df.isna().sum()/df.shape[0])*100)    #total number of empty/null values in each column

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


print("\n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("CHECKING IF ALL THE DATA EXCEPT OVERALL AND FEMALE:MALE RATIO HAS BEEN CLEANED")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("\n")

print(df.isna().sum()/df.shape[0]*100)


print("\n")
print("++++++++++++++++++")
print("++++++++++++++++++")
print("DATA VISUALISATION")
print("++++++++++++++++++")
print("++++++++++++++++++")
print("\n")


print("+++++++")
print("MATPLOT")
print("+++++++")

top10=df.head(10)
top10_f=top10.loc[:,['QS rank','No.of students per staff','International students','Female:Male Ratio','Overall','Teaching','Research','Citations','Industry Income','International Outlook']]

print(top10_f)

plt.figure(figsize=(15,6))
plt.plot(top10_f,label=['QS rank','No.of students per staff','International students','Female:Male Ratio','Overall','Teaching','Research','Citations','Industry Income','International Outlook'])
plt.legend(loc='lower right')

plt.title('Top 10 Universities')
plt.show()

top100=df.head(500)
top100_f=top100.loc[:,['QS rank','No.of students per staff','International students','Female:Male Ratio','Overall','Teaching','Research','Citations','Industry Income','International Outlook']]
print(top10_f)

plt.figure(figsize=(15,6))
plt.plot(top100_f,label=['QS rank','No.of students per staff','International students','Female:Male Ratio','Overall','Teaching','Research','Citations','Industry Income','International Outlook'])
plt.legend(loc='lower right')

plt.title('Top 100 Universities')
plt.show()


plt.figure(figsize=(15,6))

plt.subplot(2,5,1)
plt.plot(df['Teaching'],df['Overall'])
plt.xlabel('Teaching')
plt.ylabel('Overall')
plt.legend(loc='lower right')


plt.subplot(2,5,2)
plt.plot(df['Research'],df['Overall'])
plt.xlabel('Research')
plt.ylabel('Overall')
plt.legend(loc='lower right')


plt.subplot(2,5,3)
plt.plot(df['Citations'],df['Overall'])
plt.xlabel('Citations')
plt.ylabel('Overall')
plt.legend(loc='lower right')


plt.subplot(2,5,4)
plt.plot(df['Industry Income'],df['Overall'])
plt.xlabel('Industry Income')
plt.ylabel('Overall')
plt.legend(loc='lower right')


plt.subplot(2,5,5)
plt.plot(df['International Outlook'],df['Overall'])
plt.xlabel('International Outlook')
plt.ylabel('Overall')
plt.legend(loc='lower right')


plt.show()



#Scatter plot,word cloud and pie chart--showing which country has maximum number of universities

print("\n")
print("++++++++++++++++++++")
print("BOX AND WHISKER PLOT")
print("++++++++++++++++++++")
print("\n")


f,ax=plt.subplots(1,6,figsize=(50,5))

sns.boxplot(data=df['QS rank'],ax=ax[0],color='red')
ax[0].set_xlabel('QS rank')

sns.boxplot(data=df['No.of FTE students'],ax=ax[1],color='indigo')
ax[1].set_xlabel('No. of FTE students')

sns.boxplot(data=df['No.of students per staff'],ax=ax[2],color='grey')
ax[2].set_xlabel('No.of students per staff')

sns.boxplot(data=df['International students'],ax=ax[3],color='magenta')
ax[3].set_xlabel('International students')

sns.boxplot(data=df['Female:Male Ratio'],ax=ax[4],color='brown')
ax[4].set_xlabel('Female:Male Ratio')

sns.boxplot(data=df['Overall'],ax=ax[5],color='violet')
ax[5].set_xlabel('Overall')

f,ax=plt.subplots(1,5,figsize=(50,5))

sns.boxplot(data=df['Teaching'],ax=ax[0],color='hotpink')
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
print("+++++++++++++++++++++++")
print("CORRELATION MATRIX PLOT")
print("+++++++++++++++++++++++")
print("\n")


Names=['QS rank','No.of FTE students','No.of students per staff','Female:Male Ratio','International students','Overall','Teaching','Research','Citations','Industry Income','International Outlook']

correlations = df.corr()
fig = plt.figure(figsize=(10,100))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations)
ticks = np.arange(0,11,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(Names)
ax.set_yticklabels(Names)

for tick in ax.get_xticklabels():
    tick.set_rotation(90)
plt.show()

print("++++++++++++")
print("SNS HEAT MAP")
print("++++++++++++")


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)
plt.show()

print("+++++++++++++++")
print("REGRESSION PLOT")
print("+++++++++++++++")

for i in range(1,len(header1)):
    X=df[header1[i]]
    Y=df['Overall']

    plt.figure(i)
    sns.regplot(x=X,y=Y,data=df)
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

x=df.loc[:,['International students','Teaching','Research','Citations','International Outlook']]
y=df['QS rank']

# Splitting the dataset into training and test set

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=42) 

#Fitting the Simple Linear Regression model to the training dataset  
  
LR=LinearRegression()  
LR.fit(x_train, y_train)
print(LR.score(x_train,y_train))  
print(LR.score(x_test,y_test))


y_pred=LR.predict(x_test)

print(y_test.head(10))

print(y_pred[:10])

print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print("==============================================")


XGB=XGBRegressor()
XGB.fit(x_train, y_train)  
print(XGB.score(x_train,y_train))
print(XGB.score(x_test,y_test))

y_pred=XGB.predict(x_test)

print(y_test.head(10))

print(y_pred[:10])

print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print("==============================================")



RFR=RandomForestRegressor()
RFR.fit(x_train, y_train) 
print(RFR.score(x_train,y_train)) 
print(RFR.score(x_test,y_test))

y_pred=RFR.predict(x_test)

#print(accuracy_score(y_test,y_pred))

print(y_test.head(10))

print(y_pred[:10])

print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

 
pickle.dump(RFR, open('model.pkl','wb'))
joblib.dump(RFR,'RFR_Model.pkl')


































