#!/usr/bin/env python
# coding: utf-8

# # Suicide Prediction using Machine Learning.
# 
# Here is the outline of the procedure:
# 1. Question
# 2. Understand the data
# 3. Manipulate the data
# 4. Fit & test models
# 5. Deploy models
# 
# Parts 1 & 2 have been completed in the R Markdown Notebook in the notebooks directory. Here we will complete parts 3,4, & 5. 
# 
# # Part 3. Manipulate the Data
# 
# Here, we will prepare the data for analysis. 
# 
# First we will impute the data for the missing variables. We discard variables with more than 30% missing. 

# In[2]:


# !pip install missingno # to examine missing data


# In[3]:


import numpy as np 
import pandas as pd 
baseline = pd.read_csv("../../data/t1_beta.csv")
core = pd.read_csv("../../data/core_labels.csv")


# In[71]:
# prepare missing data
baseline = baseline.replace(" ", np.nan)

# In[72]:
import missingno as msno
import matplotlib.pyplot as plt

idx = list(range(0, (baseline.shape[1]-50), 50))
i = idx[50]

for i in idx:
    index_name = str(i) +  " - " + str(i+50)
    test = baseline.iloc[:, i:i+50]
    missing = msno.matrix(test)
    plt.savefig(index_name, format="png")


# In[73]:


print(msno.bar(test))


# In[74]:


msno.heatmap(test)


# In[75]:


msno.dendrogram(test)


# In[2]:


baseline.head()


# In[3]:


print("Shape:", baseline.shape)


# We can see that there are 1514 rows with 2903 variables for the baseline data.

# In[6]:


# Variables we want are saved in a txt file in the current directory 
get_ipython().system('ls')


# In[9]:


filename = "model1_vars.txt"

# using the with construct closes the file automatically when finished with it
with open(filename, "r") as file:
    variables = file.readlines()
variables = [i.strip() for i in variables]

# Now we subset the data based on the variables we are intersted in. 
model1_data = baseline[variables]


# In[10]:


# Ideation
x = model1_data.drop('Suicidal_Thoughts_12m', axis=1)
x = x.replace(" ", np.NaN)

y = model1_data[['Suicidal_Thoughts_12m']]

# Attempts
# x = model1_data.drop('Suicide_Attempts_12m', axis=1)
# y = model1_data[['Suicide_Attempts_12m']]


# In[11]:


to_drop1 = variables[21:32]
to_drop2 = variables[38:51]

to_drop = to_drop1 + to_drop2


# In[13]:


list(baseline)


# First i am going to test a model on basic features only.

# In[ ]:





# # 3. Fit the model(s)
# 
# Here we will be testing many models:
# 1. Logistic regression
# 2. Support Vector Machines
# 3. Tree models
# 4. Artificial Neural Networks
# 
# ### 3.1. Logistic Regression

# In[ ]:





# ### 3.2. Support Vector Machine

# In[ ]:





# ### 3.3. Tree Models

# In[ ]:





# ### 3.4. Artificial Neural Network

# In[ ]:





# **REFERENCES**
# 
# Franklin, J., Riberio, J., Fox, K., Bentley K, Kleiman, E., Huang, X., & Nock, M. (2017). 			     	Psychological Bulletin , 187-232.
# 
# Hack, L., Jovanovic, T., Carter, S., Ressler, K., & Smith, A. (2017). Suicide prediction using machine 	   	learning techniques in screening clinician derived data. Biological Psychiatry, 361-361.
# 
# Walsh, C., Ribeiro, J., & Franklin, J. (2017). Predicting Risk of Suicide Attempts Over Time Through   	Machine Learning. Clinical Psychological Science, 457-469.
# 
# # End.
