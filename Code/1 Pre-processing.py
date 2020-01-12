#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[3]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[4]:


df = pd.read_csv('../Data/train.csv')


# In[5]:


df.shape


# In[10]:


df.head()


# In[11]:


df.describe()


# In[12]:


df.select_dtypes(include=['object']).columns


# In[13]:


df.select_dtypes(include=['int64','float64']).columns


# In[16]:


df.MSZoning.value_counts()


# In[17]:


#Find the missing columns and their types


# In[18]:


df_dtypes = pd.merge(df.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index(),
         df.dtypes.to_frame('feature_type').reset_index(),
         on = 'index',
         how = 'inner')


# In[19]:


df_dtypes.sort_values(['missing_value', 'feature_type'])


# In[20]:


#Check columns have more than  500  missing values


# In[21]:


missing_df = df.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index()


# In[22]:


miss_feature = list(missing_df[missing_df.missing_value >= 500]['index'])


# In[23]:


miss_feature


# In[24]:


print(sorted(miss_feature))


# In[25]:


df.drop(miss_feature, axis = 1, inplace = True)


# In[26]:


df.shape


# In[27]:


#Remove constant features


# In[28]:


def find_constant_features(dataFrame):
    const_features = []
    for column in list(dataFrame.columns):
        if dataFrame[column].unique().size < 2:
            const_features.append(column)
    return const_features


# In[29]:


const_features = find_constant_features(df)


# In[30]:


const_features


# In[31]:


#Remove Duplicate rows


# In[32]:


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups


# In[33]:


duplicate_cols = duplicate_columns(df)


# In[34]:


duplicate_cols


# In[35]:


# Categorical Encoding Process


# In[48]:


target_encoded_features=df.select_dtypes(include=['object']).columns


# In[53]:


len(target_encoded_features)


# In[54]:


target_encoded_features


# In[68]:


import category_encoders as ce


# In[73]:


ordinal_encoder = ce.OrdinalEncoder(cols =['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition'])


# In[74]:


ordinal_encoder.fit(df)


# In[75]:


encoded_df = ordinal_encoder.transform(df)


# In[76]:


encoded_df.head(100)


# In[78]:


encoded_df.select_dtypes(include=['object']).columns


# In[79]:


df_dtypes = pd.merge(encoded_df.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index(),
         encoded_df.dtypes.to_frame('feature_type').reset_index(),
         on = 'index',
         how = 'inner')


# In[80]:


df_dtypes.sort_values(['missing_value', 'feature_type'])


# In[ ]:


# Missing Value Manage 


# In[100]:


mean_value=encoded_df['MasVnrArea'].mean()
encoded_df['MasVnrArea']=encoded_df['MasVnrArea'].fillna(mean_value)


# In[101]:


mean_value2=encoded_df['GarageYrBlt'].mean()
encoded_df['GarageYrBlt']=encoded_df['GarageYrBlt'].fillna(mean_value2)


# In[102]:


mean_value3=encoded_df['LotFrontage'].mean()
encoded_df['LotFrontage']=encoded_df['LotFrontage'].fillna(mean_value2)


# In[106]:


df_dtypes = pd.merge(encoded_df.isnull().sum(axis = 0).sort_values().to_frame('missing_value').reset_index(),
         encoded_df.dtypes.to_frame('feature_type').reset_index(),
         on = 'index',
         how = 'inner')


# In[107]:


df_dtypes.sort_values(['missing_value', 'feature_type'])


# In[108]:


encoded_df.head(100)


# In[113]:


encoded_df.drop(['Id'], axis = 1)


# In[116]:


encoded_df.to_csv('../Data/Preprocess_train_Data.csv')


# In[ ]:




