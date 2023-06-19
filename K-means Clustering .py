#!/usr/bin/env python
# coding: utf-8

# # FIND OPTIMUM NO OF CLUSTERS IN IRIS DATASET

# Importing Libraries 

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


# Importing the dataset

# In[2]:


data=pd.read_csv(r"C:\Users\Lenovo\Desktop\Iris.csv")
data.iloc[:,1:5].head(5)


# Data Pre-processing

# In[3]:


X = data.iloc[:,1:5].values
print(X)


# Finding the number of Optimum clusters 

# In[4]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# We choose the number of clusters as '3'

# Implementing the model

# In[5]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
Y_kmeans = kmeans.fit_predict(X)


# Model Visualisation

# In[6]:


plt.scatter(X[Y_kmeans == 0, 0], X[Y_kmeans == 0, 1],s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[Y_kmeans == 1, 0], X[Y_kmeans == 1, 1],s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[Y_kmeans == 2, 0], X[Y_kmeans == 2, 1],s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')
plt.legend()


# In[ ]:




