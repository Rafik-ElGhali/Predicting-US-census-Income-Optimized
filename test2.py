# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:07:35 2023

@author: Mega-PC
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import cdist
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv('adult.data', header=None, delimiter=', ', names=[
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
] )


data= data.replace('?', np.nan)


array_of__string_columns = data.select_dtypes(include='object').columns.tolist()



df_data = data.copy()


missing_values_count = data.isnull().sum()

total_cells=np.product(data.shape)
total_missing= missing_values_count.sum ()



percent_missing = (total_missing/total_cells) * 100
print(percent_missing)


# Function to extract unique values from specific columns

def extract_unique_values(data, columns):
#Dictionary containing unique values for each column
     unique_values_dict = {}
     for column_name in columns:
         unique_values = data[column_name].unique()
         unique_values_dict[column_name] = unique_values
         
     return unique_values_dict
 
#test=extract_unique_values(data,['workclass']) 
#print(test)  



def replace_unique_values(data, columns):
     for column_name in columns:
         test = extract_unique_values(data, [column_name])
         unique_values = dict()
         for j, element in enumerate(test[column_name]):
             if pd.isna(element):
                 unique_values[element]=-1
             
             else:  
                unique_values[element] = j
             
             
         df_data[column_name] = df_data[column_name].replace(test[column_name], unique_values.values())
     return df_data

df_data = replace_unique_values(df_data, array_of__string_columns )


print(df_data.head())


df_test=df_data.copy()


df_minus_one_workclass = df_test[df_test['workclass'] == -1].copy() 
df_minus_one_occupation = df_test[df_test['occupation'] == -1].copy() 
df_minus_one_native_country=df_test[df_test["native_country"]== -1].copy()



df_workclass_without_minus_one= df_test[df_test['workclass'] != -1].copy()
df_occupation_without_minus_one= df_test[df_test['occupation'] != -1].copy()
df_native_country_without_minus_one= df_test[df_test['native_country'] != -1].copy()



# Step 1: Prepare the feature matrix for KNN algorithm

X = df_workclass_without_minus_one

# Step 2: Instantiate and fit the KNN model
k = 5  # Set the number of nearest neighbors to consider
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(X)

# Step 3: Iterate over each row in df_minus_one
for row_index, row_data in df_minus_one_workclass.iterrows():
    # Extract the feature values from the current row
    features = row_data

    
    # Find the k nearest neighbors of the current row
    distances, indices = knn.kneighbors([features])
    
    # Extract the workclass values of the nearest neighbors
    nearest_neighbor_values = df_test.loc[indices[0], 'workclass']
    
    # Remove -1 and NaN values from nearest_neighbor_values
    nearest_neighbor_values = nearest_neighbor_values[nearest_neighbor_values != -1].dropna()
    
    # Replace the -1 value in df_minus_one with the most frequent value from the nearest neighbors
    most_frequent_value = nearest_neighbor_values.mode().iloc[0]
    df_minus_one_workclass.loc[row_index, 'workclass'] = most_frequent_value

# Print the updated df_minus_one
print(df_minus_one_workclass)



# Step 1: Prepare the feature matrix for KNN algorithm
Y = df_occupation_without_minus_one

# Step 2: Instantiate and fit the KNN model
k = 5  # Set the number of nearest neighbors to consider
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(Y)

# Step 3: Iterate over each row in df_minus_one
for row_index, row_data in df_minus_one_occupation.iterrows():
    # Extract the feature values from the current row
    features = row_data

    
    # Find the k nearest neighbors of the current row
    distances, indices = knn.kneighbors([features])
    
    # Extract the workclass values of the nearest neighbors
    nearest_neighbor_values = df_test.loc[indices[0], 'occupation']
    
    # Remove -1 and NaN values from nearest_neighbor_values
    nearest_neighbor_values = nearest_neighbor_values[nearest_neighbor_values != -1].dropna()
    
    # Replace the -1 value in df_minus_one with the most frequent value from the nearest neighbors
    most_frequent_value = nearest_neighbor_values.mode().iloc[0]
    df_minus_one_occupation.loc[row_index, 'occupation'] = most_frequent_value

# Print the updated df_minus_one
print(df_minus_one_occupation)



# Step 1: Prepare the feature matrix for KNN algorithm
Z = df_native_country_without_minus_one

# Step 2: Instantiate and fit the KNN model
k = 5  # Set the number of nearest neighbors to consider
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(Z)

# Step 3: Iterate over each row in df_minus_one
for row_index, row_data in df_minus_one_native_country.iterrows():
    # Extract the feature values from the current row
    features = row_data

    
    # Find the k nearest neighbors of the current row
    distances, indices = knn.kneighbors([features])
    
    # Extract the workclass values of the nearest neighbors
    nearest_neighbor_values = df_test.loc[indices[0], 'native_country']
    
    # Remove -1 and NaN values from nearest_neighbor_values
    nearest_neighbor_values = nearest_neighbor_values[nearest_neighbor_values != -1].dropna()
    
    # Replace the -1 value in df_minus_one with the most frequent value from the nearest neighbors
    most_frequent_value = nearest_neighbor_values.mode().iloc[0]
    df_minus_one_native_country.loc[row_index, 'native_country'] = most_frequent_value

# Print the updated df_minus_one
print(df_minus_one_native_country)
 
df_clean_data=df_test.copy()



list_index_workclass= df_minus_one_workclass.index
list_index_occupation= df_minus_one_occupation.index
list_index_native_country= df_minus_one_native_country.index

df_clean_data['workclass'].loc[list_index_workclass]=df_minus_one_workclass['workclass']
df_clean_data['occupation'].loc[list_index_occupation]=df_minus_one_occupation['occupation']
df_clean_data['native_country'].loc[list_index_native_country]=df_minus_one_native_country['native_country']