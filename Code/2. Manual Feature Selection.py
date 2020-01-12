#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv('../Data/Preprocess_train_Data.csv')


# In[3]:


df.head()


# In[ ]:





# In[4]:


#Manual Feature Selection 

df.iloc[0:5,0:10]


# In[5]:


def datavisualization(x_feture,y_feture):
    sns.catplot(x=x_feture, y=y_feture, data=df,kind='bar');
                     
    


# In[6]:


datavisualization('MSSubClass','SalePrice')


# In[7]:


features_to_be_removed = []


# In[8]:


datavisualization('Street','SalePrice')


# In[9]:


features_to_be_removed.extend(['MSZoning', 'Id','Street'])
features_to_be_removed


# In[10]:


datavisualization('LotShape','SalePrice')


# In[11]:


features_to_be_removed.extend(['LandContour','Utilities'])
features_to_be_removed


# In[12]:


df.iloc[0:5,11:20]


# In[13]:


datavisualization('BldgType','SalePrice')


# In[14]:


features_to_be_removed.extend(['LandSlope','Condition1','Condition2','HouseStyle'])
features_to_be_removed


# In[15]:


df.iloc[0:5,21:30]


# In[16]:


datavisualization('RoofMatl','SalePrice')


# In[17]:


datavisualization('Exterior2nd','SalePrice')


# In[18]:


features_to_be_removed.extend(['ExterQual','ExterCond'])
features_to_be_removed


# In[19]:


df.iloc[0:5,30:40]


# In[20]:


datavisualization('BsmtQual','SalePrice')


# In[21]:


features_to_be_removed.extend(['BsmtFinType1','BsmtFinType2','BsmtFinSF2','Heating'])
features_to_be_removed


# In[22]:


df.iloc[0:5,40:50]


# In[23]:


features_to_be_removed.extend(['HeatingQC','CentralAir','Electrical','LowQualFinSF','BsmtFullBath','BsmtHalfBath'])


# In[24]:


df.iloc[0:5,50:60]


# In[25]:


features_to_be_removed


# In[26]:


len(set(features_to_be_removed))


# In[27]:


df_selected = df.drop(list(set(features_to_be_removed)), axis = 1)


# In[28]:


df_selected.shape


# In[29]:


df_selected.to_csv('../Data/after_manual_selection.csv', index = False)


# In[ ]:




