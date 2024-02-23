#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import numpy as np
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[5]:


data = pd.read_csv('WineQT.csv', sep=',', header=0)
data.columns = data.columns.str.replace(' ','_')
data


# In[13]:


data.isnull().sum()


# In[14]:


data.describe().T


# In[15]:


data.hist(bins=30, figsize=(10,10))
plt.show()


# In[16]:


plt.figure(figsize=(12, 12))
sns.heatmap(data.corr()>0.7, annot=True, cbar=False)
plt.show()


# In[17]:


plt.bar(data['quality'], data['alcohol'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.show()


# In[18]:


data['quality'].unique()


# In[19]:


data['quality'] = np.where(data['quality']>5, 1, 0)


# In[20]:


data['quality'].head()


# In[21]:


X = data.drop('quality', axis=1)
y = data['quality']


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[23]:


scaler = StandardScaler()


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[25]:


scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)


# In[27]:


scaled_x_train.shape


# In[30]:


sns.set_style('ticks')
sns.set_context("notebook", font_scale= 1.1)

colnm = data.columns.tolist()[:11]
plt.figure(figsize = (10, 8))
for i in range(12):
    plt.subplot(4,3,i+1)
    sns.boxplot(x ='quality', y = data.columns[i], data = data,  width = 0.6)    
    plt.ylabel(data.columns[i],fontsize = 12)
plt.tight_layout()


# In[ ]:




