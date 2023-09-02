# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 09:37:29 2023

@author: Mega-PC
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist



data = pd.read_csv('adult.data', header=None, delimiter=', ', names=[
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
] )


data= data.replace('?', np.nan)


array_of__string_columns = data.select_dtypes(include='object').columns.tolist()


df_data = data.copy()




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


# Select the first row in df_minus_one
row_index = df_minus_one.index[0]
row_data = df_minus_one.loc[row_index]

# Extract the 'age' value from the first row
age_value = row_data['age']
fnlwgt_value= row_data['fnlwgt']

features = row_data.drop('workclass')

df_features = df_test.drop('workclass', axis=1)
target_features = features.values.reshape(1, -1)
#distances = cdist(df_features, target_features, 'cosine')


similarities = cosine_similarity(df_features, target_features)
#similarities = cosine_similarity(df_test[['age','fnlwgt','workclass']], [[age_value,-1, fnlwgt_value]])

similarity_threshold = 0.5
n=5
valid_similarities = similarities[similarities > similarity_threshold]


#most_similar_indices = distances.argsort()[:5]
#most_similar_values = df_test.loc[most_similar_indices].ravel()

# Convert the most_similar_indices variable to a list
#most_similar_indices = most_similar_indices.tolist()


# Convert the most_similar_values variable to a list
#most_similar_values = most_similar_values.squeeze()



# Update the value in df_minus_one
#most_similar_index = most_similar_indices[0]

#df_minus_one.loc[row_index, 'workclass'] = most_similar_values[0]

if len(valid_similarities) > 0:
    most_similar_indices = valid_similarities.argsort()[-n:][::-1]
    most_similar_values = df_test.loc[most_similar_indices, 'workclass']
    
    # Step 5: Replace the -1 value in df_minus_one with the most similar value
    most_frequent_value = most_similar_values.mode().iloc[0]
    df_minus_one.loc[row_index, 'workclass'] = most_frequent_value






# Calculate the correlations between 'age' and 'workclass' in df_data
#correlations = df_test.corr().loc['fnlwgt', 'workclass']

#most_correlated_index = df_test[df_test['fnlwgt'] == fnlwgt_value]['workclass'].idxmax()
#most_correlated_value = df_test.loc[most_correlated_index, 'workclass']

# Replace the -1 value in df_minus_one with the most correlated value
#df_minus_one.loc[row_index, 'workclass'] = most_similar_value

# Print the updated df_minus_one
print(df_minus_one)