#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, KFold

from sklearn.metrics import recall_score, roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score,                             classification_report, confusion_matrix


# In[3]:


from sklearn.linear_model import LogisticRegression


# In[4]:


df_selected=pd.read_csv('../Data/after_correlation_select_high_corr.csv')


# In[5]:


df_selected.shape


# In[6]:


df_selected.head()


# In[7]:


featuer_remove=[]


# In[8]:


featuer_remove.extend(['Unnamed: 0'])


# In[9]:


featuer_remove


# In[10]:


df_selected.drop(featuer_remove, axis=1, inplace=True)


# In[11]:


df_selected.head()


# In[12]:


df_selected.SalePrice.value_counts().head()
df_selected.columns


# In[ ]:





# In[13]:


# Standarize the data


# In[14]:


X = df_selected.drop('SalePrice', axis = 1)
Y = df_selected.SalePrice


# In[15]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.30, random_state=0)


# In[16]:


len(xtrain)


# In[17]:


len(xtest)


# In[18]:


mms = StandardScaler()
mms.fit(xtrain)
xtrain_scaled = mms.transform(xtrain)


# In[ ]:





# In[19]:


# Logistics regression


# In[20]:


logisticRegr = LogisticRegression()


# In[21]:


logisticRegr.fit(xtrain_scaled, ytrain)


# In[22]:


xtest_scaled = mms.transform(xtest)


# In[ ]:





# In[ ]:





# In[23]:


lr_pred = logisticRegr.predict(xtest_scaled)


# In[35]:


logisticRegr.score(xtest_scaled,ytest)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




