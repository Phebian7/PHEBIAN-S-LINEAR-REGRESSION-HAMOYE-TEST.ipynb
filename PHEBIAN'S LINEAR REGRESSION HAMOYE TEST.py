#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Appliance Energy Prediction Data


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df= pd.read_csv('Downloads\energydata_complete.csv')
df


# In[5]:


#Show data set columns


# In[3]:


df.columns


# In[ ]:


#Checking for null and non-null objects


# In[4]:


df.info()


# In[ ]:


#Drop columns "date" and "lights" from the dataset


# In[5]:


df_1= df.drop(columns= ['date','lights'])
df_1


# In[ ]:


#Normalize dataset using the MixMaxScaler


# In[6]:


from sklearn.preprocessing import MinMaxScaler


# In[7]:


scaler= MinMaxScaler()


# In[8]:


norm_df= pd.DataFrame(scaler.fit_transform(df_1), columns=df_1.columns)
feat_df= norm_df.drop(columns=['Appliances'])
App_target= norm_df['Appliances']


# In[ ]:


##Creating a model, fitting the model into training and testing dataset


# In[9]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[10]:


x_train, x_test, y_train, y_test= train_test_split(feat_df, App_target, test_size=0.3, random_state=42)


# In[11]:


model= LinearRegression()


# In[12]:


model.fit(x_train, y_train)


# In[ ]:


#getting the predicted value


# In[13]:


predicted_values= model.predict(x_test)


# In[ ]:


#finding the mean absolute error (mae)


# In[14]:


from sklearn.metrics import mean_absolute_error


# In[15]:


mae= mean_absolute_error(y_test, predicted_values)


# In[16]:


round(mae,2)


# In[ ]:


#finding R-squared


# In[17]:


from sklearn.metrics import r2_score


# In[18]:


r2_score= r2_score(y_test, predicted_values)


# In[19]:


round(r2_score, 2)


# In[ ]:


#residual sum of squares RSS np.sum(np.square(targ_test - pred val)


# In[20]:


rss= np.sum(np.square(y_test-predicted_values))


# In[21]:


round(rss, 3)


# In[ ]:


##finding the root mean squared error


# In[22]:


from sklearn.metrics import mean_squared_error


# In[23]:


rmse= np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 2)


# In[24]:


#import Lasso and Ridge regression methods


# In[25]:


from sklearn.linear_model import Lasso


# In[26]:


lasso_reg= Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)


# In[28]:


from sklearn.linear_model import Ridge


# In[29]:


ridge_reg= Ridge(alpha=0.5)
ridge_reg.fit(x_train, y_train)


# In[31]:


def get_weights_df(model, feat, col_name):
    #this functipn returns the weight of every features
    weights=pd.Series(model.coef_, feat.columns).sort_values()
    weights_df= pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Features', col_name]
    weights_df[col_name].round(3)
    return weights_df

linear_model_weights= get_weights_df(model, x_train, 'Linear_Model_Weight')
ridge_weights_df= get_weights_df(ridge_reg, x_train, 'Ridge_Weight')
lasso_weights_df= get_weights_df(lasso_reg, x_train, 'Lasso_Weight')

final_weights= pd.merge(linear_model_weights, ridge_weights_df, on='Features')
final_weights= pd.merge(final_weights, lasso_weights_df, on='Features')
final_weights


# In[ ]:





# In[ ]:




