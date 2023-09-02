# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:22:30 2023

@author: Mega-PC
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
from scipy.spatial import distance


data = pd.read_csv('adult.data', header=None, delimiter=', ', names=[
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
] )


data= data.replace('?', np.nan)


array_of__string_columns = data.select_dtypes(include='object').columns.tolist()


df_data = data.copy()

sns.set_style("whitegrid");
sns.FacetGrid(df_test, hue='workclass', size=5) \
.map(plt.scatter, 'age', 'fnlwgt','education') \
.add_legend();
plt.show()



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

df_minus_one = df_test[df_test['workclass'] == -1].copy() 

#df_test = df_test.drop('workclass', axis=1)


print("DataFrame with -1 values:")
print(df_minus_one.head())








# Step 1: Select the first row in df_minus_one
row_index = df_minus_one.index[0]
row_data = df_minus_one.loc[row_index]

# Step 2: Extract the feature values from the first row
features = row_data.drop('workclass')

# Step 3: Calculate the distances between rows in df_data and the first row in df_minus_one, neglecting -1 values
df_features = df_data.drop('workclass', axis=1)  # Drop the 'workclass' column
df_features_no_minus_one = df_features.replace(-1, np.nan)
features_no_minus_one = features.replace(-1, np.nan)

# Calculate the Jaccard distance for common features
distances = df_features_no_minus_one.apply(
    lambda x: distance.jaccard(features_no_minus_one.loc[x.dropna().index], x.dropna()),
    axis=1
)

# Step 4: Find the most similar rows based on the distances, excluding -1 and NaN values
n = 5  # Set the number of most similar rows to consider
valid_indices = distances[~distances.isin([-1, np.nan])].argsort()[:n]
most_similar_values = df_data.loc[valid_indices, 'workclass']

# Remove NaN and -1 values from most_similar_values
most_similar_values = most_similar_values.dropna().replace(-1, np.nan)

# Convert non-finite values to NaN
most_similar_values = most_similar_values.replace([np.inf, -np.inf], np.nan)

# Step 5: Replace the -1 value in df_minus_one with the most frequent non-finite-corrected integer value
most_frequent_value = most_similar_values.dropna().astype(int).mode().iloc[0]
df_minus_one.loc[row_index, 'workclass'] = most_frequent_value

# Print the updated df_minus_one
print(df_minus_one)

    plt.figure()
    sns.scatterplot(data=df_data, x='age', y='hours_per_week')
    sns.scatterplot(data=df_data.iloc[indices[0]], x='age', y='hours_per_week', marker='o', color='red')
    sns.scatterplot(data=row_data.to_frame().T, x='age', y='hours_per_week', marker='x', color='green')
    plt.xlabel('Age')
    plt.ylabel('Hours per Week')
    plt.title('KNN: Nearest Neighbors')
    plt.legend(['Data Points', 'Nearest Neighbors', 'Current Row'])
    plt.show()

# Step 1: Prepare the feature matrix for KNN algorithm
#X = df_test.drop('workclass', axis=1)  # Drop the 'workclass' column
X = df_workclass_without_minus_one

# Step 2: Instantiate and fit the KNN model
k = 5  # Set the number of nearest neighbors to consider
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(X)

# Step 3: Iterate over each row in df_minus_one
for row_index, row_data in df_minus_one_workclass.iterrows():
    # Extract the feature values from the current row
    features = row_data
    #features = row_data
    
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