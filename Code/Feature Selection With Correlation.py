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

from sklearn.utils import shuffle, class_weight
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[3]:


import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[4]:


init_notebook_mode(connected=True)
cf.go_offline()


# In[5]:


df_selected = pd.read_csv('../Data/after_manual_selection.csv')


# In[6]:


df_selected.head()


# In[7]:


feture_to_be_remove=[]


# In[8]:


feture_to_be_remove.extend(['LotShape','LotConfig','Neighborhood','BldgType'])           


# In[9]:


feture_to_be_remove.extend(['3SsnPorch','ScreenPorch','PoolArea','MiscVal'])


# In[10]:


feture_to_be_remove


# In[ ]:





# In[11]:


feture_to_be_remove.extend(['SaleType','SaleCondition'])


# In[12]:


feture_to_be_remove.extend(['GarageQual','GarageCond','PavedDrive'])


# In[13]:


feture_to_be_remove


# In[14]:


df = df_selected.drop(list(set(feture_to_be_remove)), axis = 1)


# In[15]:


df.shape


# In[16]:


df.describe().T


# In[17]:


# Feature orrelation


# In[18]:


corr = df.corr(method = 'spearman')


# In[19]:


layout = cf.Layout(height=600,width=600)
corr.abs().iplot(kind = 'heatmap', layout=layout.to_plotly_json(), colorscale = 'RdBu')


# In[20]:


new_corr = corr.abs()
new_corr.loc[:,:] = np.tril(new_corr, k=-1) # below main lower triangle of an array
new_corr = new_corr.stack().to_frame('correlation').reset_index().sort_values(by='correlation', ascending=False)


# In[21]:


new_corr[new_corr.correlation > 0.4]


# In[22]:


df.shape


# In[23]:


#Correlation with target variable


# In[24]:


corr_with_target = df.corrwith(df.SalePrice).sort_values(ascending = False).abs().to_frame('correlation_with_target').reset_index().head(20)
unique_values = df_selected.nunique().to_frame('unique_values').reset_index()
corr_with_unique = pd.merge(corr_with_target, unique_values, on = 'index', how = 'inner')


# In[25]:


corr_with_unique


# In[26]:


df.shape


# In[27]:


newdf=df[['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]


# In[28]:


final_df=pd.DataFrame(newdf)


# In[29]:


final_df.shape


# In[30]:


final_df.head()


# In[31]:


final_df.to_csv('../Data/after_correlation_select_high_corr.csv')


# In[ ]:





# In[ ]:





# In[ ]:




