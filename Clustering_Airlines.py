#!/usr/bin/env python
# coding: utf-8

# # Unsupervised ML - Clustering

# ### 1. Importing Libraries 

# In[1]:


# import hierarchical clustering libraries

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sn 
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage


# ### 2. Importing data

# In[2]:


airline_data=pd.read_csv('data-Table_airlines.csv', sep=';')
airline_data


# ### 3. Data understanding 

# #### 3.1 Initial Analysis

# In[3]:


#shape of the dataset 
airline_data.shape


# In[4]:


#checking the datatypes 
airline_data.dtypes


# In[5]:


#checking for null values 
airline_data.isna().sum()


# ### 4. Data preprocessing 

# In[6]:


airline_data.head()


# In[7]:


#Renaming columns 
airline_data.rename(columns={'Award?': 'Award', 'ID#':'ID'}, inplace=True)


# In[8]:


#Dropping irrelevant columns 
airline_data=airline_data.drop(labels=['ID'], axis=1)
airline_data.head()


# ## 5. Hierarchial Clustering 

# In[9]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[10]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airline_data.iloc[:,1:])
df_norm


# In[11]:


# create dendrogram
plt.figure(figsize=(10, 7))

dendrogram = sch.dendrogram(sch.linkage(df_norm, method='single'), orientation='top')


# In[12]:


from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
get_ipython().run_line_magic('pinfo2', 'pdist')


# In[13]:


xdist=pdist(df_norm,metric="euclidean")


# In[14]:


linked = linkage(xdist, 'ward')

plt.figure(figsize=(20, 15))
dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()


# In[15]:


# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')
hc


# In[16]:


# save clusters for chart
y_hc = hc.fit_predict(df_norm)
y_hc


# In[17]:


Clusters=pd.DataFrame(y_hc,columns=['Clusters'])
Clusters


# In[18]:


airline_data['clusters'] = Clusters


# In[19]:


airline_data


# In[20]:


airline_data[airline_data['clusters']==0]


# In[21]:


airline_data[airline_data['clusters']==1]


# In[22]:


airline_data[airline_data['clusters']==2]


# In[23]:


airline_data[airline_data['clusters']==3]


# In[26]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(airline_data['clusters'],airline_data['Balance']) 


# ## 6. Kmeans 

# In[27]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
plt.style.use("seaborn-darkgrid")

kmodel = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto',init="k-means++")


# In[28]:


airline_km=pd.read_csv('data-Table_airlines.csv', sep=';')
airline_km


# In[29]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_airline = scaler.fit_transform(df_norm)
scaled_airline


# In[30]:


import warnings
warnings.filterwarnings('ignore')


# In[31]:


wcss = []

for i in range(1,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(scaled_airline)
    wcss.append(kmeans.inertia_) #wcss
   


# In[32]:


wcss


# In[33]:


plt.plot(range(1,10),wcss)


# In[ ]:





# ## 7. DB Scan

# In[34]:


scaled_airline


# In[35]:


# DBSCAN Clustering
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=1,min_samples=4)
dbscan.fit(scaled_airline)


# In[36]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[37]:


# Assign clusters to the data set
# Assign clusters to the data set
airline_db=airline_data.copy()
airline_db=airline_db.drop(labels='clusters', axis=1)

airline_db['clusters_dbscan']=dbscan.labels_
airline_db


# In[38]:


airline_db.groupby('clusters_dbscan').agg(['mean']).reset_index()


# In[40]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(airline_db['clusters_dbscan'],airline_db['Balance']) 


# In[ ]:





# In[ ]:




