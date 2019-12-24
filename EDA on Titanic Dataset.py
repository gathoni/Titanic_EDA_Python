#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset EDA
# 
# An EDA of Titanic Dataset from Kaggle using Python

# ### Import Libraries

# # import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline

# In[5]:


titanic = pd.read_csv("https://raw.githubusercontent.com/krishnaik06/EDA1/master/titanic_train.csv")


# In[4]:


titanic.head()


# ### EDA

# In[6]:


# Missing Data
titanic.isnull()


# In[8]:


# Visualizing the missing values
sns.heatmap(titanic.isnull(),yticklabels = False, cbar=False)


# #### Age column has missing values with variation in occurence
# #### Cabin column has the most missing values with variation in occurence

# In[9]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=titanic)


# In[11]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=titanic,palette='RdBu_r')


# In[13]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=titanic,palette='rainbow')


# In[14]:


sns.distplot(titanic['Age'].dropna(),kde=False,color='blue',bins=40)


# In[15]:


titanic['Age'].hist(bins=30,color='purple',alpha=0.3)


# In[16]:


sns.countplot(x='SibSp',data=titanic)


# ### Data Cleaning

# Fill in the missing age data instead of dropping it. One way of doing this is by filling in the mean age of the passengers (Imputation). However we can be smater about this and check teh average age by passenger class. 
# i.e.

# In[19]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=titanic,palette='rainbow')


# The wealthier passengers tend to be older, which makes sense. We'll use the averages to impute based on pclass for age

# In[20]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        
        elif Pclass == 2:
            return 29
        
        else:
            return 24
        
    else:
        return Age


# In[21]:


# Replace the null value in the age column with the impute_age function
titanic['Age'] = titanic[['Age','Pclass']].apply(impute_age,axis=1)


# In[22]:


# Check the heat map again
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False)


# The Age column has been replaced with the imputed mean age and thus there are no longer null values
# 
# We will drop the cabin column since we have so many null values

# In[24]:


titanic.drop('Cabin',axis=1,inplace=True)


# In[25]:


titanic.head()


# In[26]:


# Check the heat map again
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False)


# No missing values

# In[27]:


titanic.dropna(inplace=True)


# ## Converting Categorical Feautures
# 
# We'll need to convert categorical feautures to dummy variables using pandas!

# In[28]:


titanic.info()


# In[30]:


# Apply dummy categories
pd.get_dummies(titanic['Embarked'],drop_first=True).head()


# In[31]:


sex = pd.get_dummies(titanic['Sex'],drop_first=True)
embark =  pd.get_dummies(titanic['Embarked'],drop_first=True)


# In[32]:


titanic.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[33]:


titanic.head()


# In[34]:


titanic = pd.concat([titanic,sex,embark],axis=1)


# In[35]:


titanic.head()


# Survived column is our dependent feature and the rest are dependent feature

# ## Building a Logistic Regression Model
# 
# We start by splitting our data into a training set and a test set
# 
# ### Train Test Split

# In[37]:


titanic.drop('Survived',axis=1).head()


# In[38]:


titanic['Survived'].head()


# In[39]:


from sklearn.model_selection import train_test_split


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(titanic.drop('Survived',
                                                                axis=1),titanic['Survived']
                                                                ,test_size=0.30,random_state=101)


# ## Training and Predicting

# In[42]:


from sklearn.linear_model import LogisticRegression


# In[46]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
    
logmodel =  LogisticRegression()
logmodel.fit(x_train,y_train)


# In[47]:


predictions = logmodel.predict(x_test)


# In[49]:


from sklearn.metrics import confusion_matrix


# In[51]:


accuracy=confusion_matrix(y_test,predictions)


# In[52]:


accuracy


# In[53]:


from sklearn.metrics import accuracy_score


# In[54]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[56]:


predictions


# ## Evaluation
# 
# We can check precision,recall,f1-score using classification report!

# In[57]:


from sklearn.metrics import classification_report


# In[58]:


print(classification_report(y_test,predictions))


# 82% accuracy for predictions is not so bad

# In[ ]:




