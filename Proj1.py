# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:17:14 2023

@author: Mega-PC
"""
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




# extracting data from a txt file and organized into a Dataframe table

data = pd.read_csv('adult.data', header=None, delimiter=', ', names=[
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
] )






#data.columns=columns

#data.to_csv('adult.csv', index=False)

#print("jawekbehi")

# replace ? with nan
data= data.replace('?', np.nan)

#examining missing values
print("Missing values distribution: ")
print(data.isnull().mean())
print("good")

#We have found that the highest rates of missing values distribution are with :
    #occupation: 0.056601  5%
    #workclass : 0.056386  5%
    #native_country: 0.017905  1%    
#%%
#data["workclass"]=data["workclass"].fillna("Private")

#x=data["age"]
#y=data["education_num"]

#correlation= x.corr(y, numeric_only=True)
#print("The correlation number between ", x," and capital_loss", y ,": ", correlation)

#plt.scatter(x,y)

df1=pd.DataFrame(data=data, columns=['age','education_num','hours_per_week'])
#df1.corr(method='pearson')

corrMatrix = df1.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

#%%
fig = plt.figure(figsize=(10,5))
sns.countplot('workclass', hue='income', data=data)
plt.tight_layout()
plt.show()
#%%
fig = plt.figure(figsize=(20,5))
sns.countplot('native_country', hue='income', data=data)
plt.tight_layout()
plt.show()
#%%
df2=pd.DataFrame(data=data, columns=['age','education_num','hours_per_week'])

#%%

df3=data[pd.isna(data['workclass'])]
df4=data[pd.isna(data['occupation'])]
df5=data[pd.isna(data['native_country'])]

similarity_matrix = cosine_similarity (df3.fillna(0), data)
most_similar_rows = similarity_matrix.argmax(axis=1)
new_df = data.iloc[most_similar_rows]

print(new_df)

#%%

df6=data.copy()



def extract_unique_values(data,array_of_columns):
    unique_values=data[array_of_columns].unique()
    return unique_values

test=extract_unique_values(data,'marital_status')
print(test)



def workclass_to_numeric(status):
    workclass = dict()
    for i, element in enumerate(test0):
        workclass[element]=i
    status_numeric = [workclass.get(s) for s in status] #list comprehension
    return status_numeric



numeric_values = workclass_to_numeric(test0)
df6['workclass'] = df6['workclass'].replace(test0, numeric_values)



df6['workclass'] = data['workclass'].str.strip().apply(workclass_to_numeric)

# def extract_unique_values(data,array_of_columns):
#     unique_values=data[array_of_columns].unique()
#     return unique_values


#def Replace_unique_values(status):
#    for i in array_of_columns:
#         test=extract_unique_values(data, array_of_columns[i])
#         array_of_columns[i]=dict()
#         for j,element in enumerate(test):
#             array_of_columns[i][element]=j
#         status_numeric = [array_of_columns[i].get(s) for s in status] 
#         return status_numeric
#     numeric_values=status_numeric(test)
#     
#     df6[array_of_columns[i]]=df6[array_of_columns[i]].replace(test,numeric_values)
#     return df6




#df6= df6.fillna(-1)


df3=df6[pd.isna(data['workclass'])]
df3= df3.fillna(-1)
df4=df6[pd.isna(data['occupation'])]
df4= df4.fillna(-1)
df5=df6[pd.isna(data['native_country'])]
df5= df5.fillna(-1)

arr=df3.to_numpy()
print(arr)

arr1=df6.to_numpy()
print(arr1)


#for i in range(arr.shape[0]):
    #for j in range(arr.shape[1]):
        #if pd.isna(arr[i][j]):
            # Calculate cosine similarity between the cell in arr and all cells in arr1
            #similarity_scores = cosine_similarity(arr[i:i+1], arr1[:, j:j+1])
            # Find the most similar cell in arr1
            #most_similar_cell = arr1[:, j][similarity_scores.argmax()]
            # Replace the NaN value in arr with the most similar cell value
            #arr[i][j] = most_similar_cell

# Convert arr back to a DataFrame
#new_df = pd.DataFrame(arr, columns=df6.columns)

# Print the new DataFrame
#print(new_df)

#comparison=df6.compare(df3, align_axis=0, keep_shape=True )

#similarity_matrix = cosine_similarity (arr, arr1)
#most_similar_rows = similarity_matrix [0].argmax()
#new_df = df6.iloc[most_similar_rows]

#print(new_df)


#pub=df6 ['age', 'fnlwgt', 'education', 'education_num',
#'marital_status', 'occupation', 'relationship', 'race', 'sex',
#'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

#print (pub)

#def find_sim():
#   if arr[0,0]==arr1[0,0]:
#   
#      return 1
#   else:
#       return 0
#print(find_sim)    
²

def identify_rows_with_minus_one(df):
    print("Identifying rows with -1 in 'workclass' column...")
    return df[df['workclass'] == -1]

def compute_cosine_similarity(df):
    print("Computing cosine similarity...")
    return cosine_similarity(df.drop(columns=['workclass']))

def find_replacement(df, df_replacements, rows_with_minus_one, index, similarity_matrix):
    print("Finding replacement for index", index)
    current_row_index = rows_with_minus_one.loc[index].name
    current_row = df.drop(columns=['workclass']).loc[current_row_index].values.reshape(1, -1)
    similarities = cosine_similarity(current_row, df.drop(columns=['workclass']))
    most_similar_row_index = np.argmax(similarities)
    most_similar_workclass = df.loc[most_similar_row_index, 'workclass']
    df_replacements.loc[current_row_index, 'workclass'] = most_similar_workclass
    similarity_matrix = compute_cosine_similarity(df_replacements)
    return df_replacements, similarity_matrix

def impute_workclass(df_data):
    df_replacements = df_data.copy()
    rows_with_minus_one = identify_rows_with_minus_one(df_data)
    similarity_matrix = compute_cosine_similarity(df_data)
    iterations = 0

    while iterations < 20:
        for index in rows_with_minus_one.index:
            df_replacements, similarity_matrix = find_replacement(df_data, df_replacements, rows_with_minus_one, index, similarity_matrix)
            iterations += 1
            if iterations >= 20:
                break
        rows_with_minus_one = identify_rows_with_minus_one(df_replacements)
        if iterations >= 20:
            break

    return df_replacements['workclass']


df_replacements = impute_workclass(df_data)

# Print the 'workclass' column with the new replacements
print(df_replacements.head())

# Print the original 'workclass' column from df_data
print(df_data['workclass'].head())

print("\nUpdated df_data:")
print(df_test.head())
