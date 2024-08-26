#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import LabelEncoder,StandardScaler


# In[69]:


df=pd.read_csv('yield_df.csv')


# In[70]:


df.head()


# In[71]:


df.describe()


# In[72]:


df.info()


# In[73]:


df.isnull().sum()


# In[74]:


df['yield']=df['hg/ha_yield']
df.drop('hg/ha_yield',axis=1,inplace=True)


# In[75]:


df.tail()


# In[76]:


yield_top_10_region = df.groupby('Area')['yield'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.barplot(x=yield_top_10_region.index, y=yield_top_10_region.values,palette='coolwarm')
plt.title('crop yield distribution of top 10 regions')
plt.xlabel('region')
plt.ylabel('yield')


# In[77]:


yield_top_10_crops = df.groupby('Item')['yield'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(14,6))
sns.barplot(x=yield_top_10_crops.index,y=yield_top_10_crops.values,palette='coolwarm')
plt.title('crop yield distribution of top 10 crops')
plt.xlabel('crop')
plt.ylabel('yield')


# In[78]:


corr = df.corr(numeric_only = True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlations')
plt.show()


# In[79]:


yield_recent_years = df.groupby('Year')['yield'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.lineplot(x=yield_recent_years.index, y=yield_recent_years.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Yield')
plt.title('Yield Variation in Recent Years')
plt.grid(True)
plt.show()


# In[80]:


yield_early_years = df.groupby('Year')['yield'].mean().sort_values(ascending=True).head(10)

plt.figure(figsize=(12, 6))
sns.lineplot(x=yield_early_years.index, y=yield_early_years.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Yield')
plt.title('Yield Variation in Early Years')
plt.grid(True)
plt.show()


# In[81]:


rainfall_yield_relation = df.groupby('average_rain_fall_mm_per_year')['yield'].mean()
rainfall_yield_relation


# In[82]:


plt.figure(figsize=(12, 6))
sns.scatterplot(x=rainfall_yield_relation.index, y=rainfall_yield_relation.values)
plt.xlabel('Average Rainfall (mm per year)')
plt.ylabel('Yield')
plt.title('Relationship Between Average Rainfall and Yield')
plt.grid(True)
plt.show()


# In[83]:


df.head()


# In[84]:


df_new = df.rename(columns={
 "average_rain_fall_mm_per_year": "Rainfall",
 "pesticides_tonnes": "Pesticides",
 "avg_temp": "Avg_Temp"
})


# In[85]:


country = LabelEncoder()
crop = LabelEncoder()
df_new['Country_Encoded'] = country.fit_transform(df_new['Area'])
df_new['Crop_Encoded'] = crop.fit_transform(df_new['Item'])


# In[86]:


X= df_new[['Country_Encoded', 'Crop_Encoded', 'Pesticides', 'Avg_Temp', 'Rainfall']]
y = df_new['yield']


# In[87]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=42)


# In[34]:


print("Shapes of training data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("\nShapes of testing data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# In[35]:


LR=LinearRegression()
LR.fit(X_train,y_train)


# In[36]:


LR.score(X_test,y_test)


# In[37]:


y_pred=LR.predict(X_test)


# In[38]:


mean_squared_error(y_test,y_pred)


# In[39]:


mean_absolute_error(y_test,y_pred)


# In[40]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[41]:


r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)


# In[ ]:





# In[42]:


ax1=sns.distplot(y_test,color='r',hist=False,label='Acual value')
sns.distplot(y_pred,color='b',hist=False,label='preducation',ax=ax1)
plt.title('Actual vs preducation Values')
plt.show()
plt.close()


# In[43]:


rfr=RandomForestRegressor()


# In[44]:


rfr.fit(X_train,y_train)


# In[45]:


rfr.score(X_test,y_test)


# In[46]:


y_pred=rfr.predict(X_test)


# In[47]:


print("r2_score:",r2_score(y_test, y_pred)*100)


# In[48]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()


# In[49]:


ax1=sns.distplot(y_test,color='r',hist=False,label='Acual value')
sns.distplot(y_pred,color='b',hist=False,label='preducation',ax=ax1)
plt.title('Actual vs preducation Values')
plt.show()
plt.close()


# In[50]:


DT=DecisionTreeRegressor()


# In[51]:


DT.fit(X_train,y_train)


# In[52]:


DT.score(X_test,y_test)


# In[53]:


y_pred=DT.predict(X_test)


# In[54]:


print("r2_score:",r2_score(y_test, y_pred)*100)


# In[55]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()


# In[89]:


ax1=sns.distplot(y_test,color='r',hist=False,label='Acual value')
sns.distplot(y_pred,color='b',hist=False,label='preducation',ax=ax1)
plt.title('Actual vs preducation Values')
plt.show()
plt.close()


# In[ ]:





# In[ ]:





# In[ ]:




