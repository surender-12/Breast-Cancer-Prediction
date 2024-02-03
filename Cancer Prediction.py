#!/usr/bin/env python
# coding: utf-8

# ## Breast Cancer Prediction Using Python

# In[2]:


# importing libraries
import numpy
import pandas as pd
import seaborn as sns


# In[3]:


# reading data from the file
df=pd.read_csv("C:\Users\User\Downloads\DA Project")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


# return all the columns with null values count
df.isna().sum()


# In[7]:


# return the size of dataset
df.shape


# In[8]:


# remove the column
df=df.dropna(axis=1)


# In[9]:


# shape of dataset after removing the null column
df.shape


# In[10]:


# describe the dataset
df.describe()


# In[11]:


# Get the count of malignant<M> and Benign<B> cells
df['diagnosis'].value_counts()


# In[12]:


sns.countplot(df['diagnosis'],label="count")


# In[13]:


# label encoding(convert the value of M and B into 1 and 0)
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)


# In[14]:


df.head()


# In[15]:


sns.pairplot(df.iloc[:,1:5],hue="diagnosis")


# In[16]:


# get the correlation
df.iloc[:,1:32].corr()


# In[17]:


# visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:10].corr(),annot=True,fmt=".0%")


# In[21]:


# split the dataset into dependent(X) and Independent(Y) datasets
X=df.iloc[:,2:31].values
Y=df.iloc[:,1].values


# In[22]:


# spliting the data into trainning and test dateset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


# In[23]:


# feature scaling
from sklearn.preprocessing import StandardScaler
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)


# In[22]:


# models/ Algorithms

def models(X_train,Y_train):
        #logistic regression
        from sklearn.linear_model import LogisticRegression
        log=LogisticRegression(random_state=0)
        log.fit(X_train,Y_train)
        
        
        #Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
        tree.fit(X_train,Y_train)
        
        #Random Forest
        from sklearn.ensemble import RandomForestClassifier
        forest=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
        forest.fit(X_train,Y_train)
        
        print('[0]logistic regression accuracy:',log.score(X_train,Y_train))
        print('[1]Decision tree accuracy:',tree.score(X_train,Y_train))
        print('[2]Random forest accuracy:',forest.score(X_train,Y_train))
        
        return log,tree,forest
        


# In[23]:


model=models(X_train,Y_train)


# In[24]:


# testing the models/result

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for i in range(len(model)):
    print("Model",i)
    print(classification_report(Y_test,model[i].predict(X_test)))
    print('Accuracy : ',accuracy_score(Y_test,model[i].predict(X_test)))


# In[25]:


# prediction of random-forest
pred=model[2].predict(X_test)
print('Predicted values:')
print(pred)
print('Actual values:')
print(Y_test)


# In[26]:


from joblib import dump
dump(model[2],"Cancer_prediction.joblib")


# In[ ]:




