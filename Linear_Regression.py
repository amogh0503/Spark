#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


url = "http://bit.ly/w-data" #dataset url
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head() #view few cells of imported data


# In[9]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='.')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[19]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  # creating x and y for plotting


# In[20]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)  # Test and Train split of data


# In[22]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[23]:


line = regressor.coef_*X+regressor.intercept_ # y = mx + c

# Plotting for the test data
plt.scatter(X, y)  
plt.plot(X, line); 
plt.show()


# In[26]:


y_pred = regressor.predict(X_test) # Predicting the scores
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[49]:


#testing for 9.25 hr/day study
hours = np.array(9.25).reshape(1,-1)
pred = regressor.predict(hours)
print("No of Hours = ",hours)
print("Percentage of Student =",pred)


# In[50]:


# Error in model
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




