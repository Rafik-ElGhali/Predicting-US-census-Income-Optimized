# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:07:35 2023

@author: Rafik-El-Ghali
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier


from scipy.spatial.distance import cdist
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

from sklearn.feature_selection import VarianceThreshold 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


#Calculate correlation between each single feature and the label

array_of__string_columns_prime = df_clean_data.select_dtypes(include='int').columns.tolist()

#def calculate_correlation(data,columns):
#    for column_name in columns:
#       df_corr=pd.DataFrame(data=data, columns=[column_name,'income']) 
#       corrMatrix = df_corr.corr()
#       sns.heatmap(corrMatrix, annot=True)

#       sns.pairplot(df_corr[[ column_name, 'income']])
#       plt.show()
       
#    return print(plt.show())

#calculate_correlation(df_clean_data, array_of__string_columns_prime)

#fig, ax = plt.subplots(1,3, figsize=(18, 8))

#Calculate correlation between all the features and the label (Correlation to Target Variable)
#corr=df_clean_data.corr('pearson')[['income']].sort_values(by='income', ascending=False)
#sns.heatmap(corr, ax=ax[0], annot=True)
#corr1=df_clean_data.corr('spearman')[['income']].sort_values(by='income', ascending=False)
#sns.heatmap(corr1,ax=ax[1], annot=True)
#corr2=df_clean_data.corr('kendall')[['income']].sort_values(by='income', ascending=False)
#sns.heatmap(corr2,ax=ax[2], annot=True)


#X=df_clean_data.iloc[:,1:14] 
#Y=df_clean_data.iloc[:,-1]
#print (X)

X = df_clean_data.values[:, :14]
Y = df_clean_data.values[:,14]

#print(Y)

#best_features= SelectKBest(score_func=chi2, k=3)
#fit= best_features.fit(X,Y)

#df_scores= pd.DataFrame(fit.scores_)
#df_columns= pd.DataFrame(X.columns)

#features_scores= pd.concat([df_columns, df_scores], axis=1)
#features_scores.columns= ['Features', 'Score']
#features_scores.sort_values(by = 'Score',ascending=False)


# Preparing data for Training and testing 

# Initializing a VarianceThreshold object to remove low-variance features
selector = VarianceThreshold(0.005)


# Applying the feature selection by transforming the feature matrix X
X = selector.fit_transform(X)

# Printing the shapes of the feature matrix X and target vector Y
X.shape,Y.shape

#PCA

# Standardizing the transformed feature matrix using StandardScaler
x_Std=StandardScaler().fit_transform(X)

pca = PCA(n_components=10)

principalComponents = pca.fit_transform(x_Std)

principalDf = pd.DataFrame(data = principalComponents)
print(principalDf)

from sklearn.model_selection import train_test_split
principalDf_train,principalDf_test,Y_train,Y_test = train_test_split(principalDf,Y,test_size = 0.2)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(principalDf_train,Y_train)
pred=model.predict(principalDf_test)


from sklearn import metrics
pca_acc=metrics.accuracy_score(Y_test,pred)*100
print(pca_acc)


#Logistic Regression

from sklearn.linear_model import LogisticRegression
reg_lr = LogisticRegression(random_state=5)
reg_lr.fit(principalDf_train,Y_train)
pred_lr=reg_lr.predict(principalDf_test)


from sklearn import metrics
lr_acc=metrics.accuracy_score(Y_test,pred_lr)*100
print(lr_acc)

#KNN

from sklearn.neighbors import KNeighborsClassifier
model_KNN = KNeighborsClassifier(n_neighbors=15)
model_KNN.fit(principalDf_train,Y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=15, p=2,
           weights='uniform')
pred_KNN = model_KNN.predict(principalDf_test)
knn_acc = model_KNN.score(principalDf_test,Y_test)*100
print(knn_acc)

#Naive Bayes
#GaussianNB

from sklearn.naive_bayes import GaussianNB
model_gnb=GaussianNB()
model_gnb.fit(principalDf_train,Y_train)
GaussianNB(priors=None)
pred_gnb = model_gnb.predict(principalDf_test)
gnb_acc = metrics.accuracy_score(Y_test,pred_gnb)*100
print(gnb_acc)

#BernoulliNB

from sklearn.naive_bayes import BernoulliNB
model_bnb=BernoulliNB()
model_bnb.fit(principalDf_train,Y_train)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
pred_bnb = model_bnb.predict(principalDf_test)
bnb_acc = metrics.accuracy_score(Y_test,pred_bnb)*100
print(bnb_acc)

#SVM

from sklearn.svm import SVC
model_svc=SVC()
model_svc.fit(principalDf_train,Y_train)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
pred_svc = model_svc.predict(principalDf_test)
svc_acc = metrics.accuracy_score(Y_test,pred_svc)*100
print(svc_acc)


#Decision Tree


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
model_tree= DecisionTreeClassifier() #max_leaf_nodes=12 #random_state=1
model_tree.fit(principalDf_train,Y_train)

pred_tree = model_tree.predict(principalDf_test)
tree_acc=metrics.accuracy_score(Y_test,pred_tree)*100
print(tree_acc)


#Entropy

model_tree1 = DecisionTreeClassifier(criterion="entropy") #max_leaf_nodes=12 #random_state=1
model_tree1.fit(principalDf_train,Y_train)
pred_tree1 = model_tree1.predict(principalDf_test)
tree1_acc= metrics.accuracy_score(Y_test,pred_tree1)*100
print(tree1_acc)


#RandomForest

b = RandomForestClassifier(max_leaf_nodes=14)
b.fit(principalDf_train,Y_train)
b_pred = b.predict(principalDf_test)
bacc=metrics.accuracy_score(Y_test,b_pred)*100
bacc

#Entropy

b1 = RandomForestClassifier(criterion="entropy",max_leaf_nodes=14)
b1.fit(principalDf_train,Y_train)
b1_pred = b1.predict(principalDf_test)
b1acc=metrics.accuracy_score(Y_test,b1_pred)*100
print(b1acc)

#Methods:

#BaggingClassifier

from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(max_samples=0.5,max_features=0.5)
bagging.fit(principalDf_train,Y_train)
pred_E_BC = bagging.predict(principalDf_test)
bc=metrics.accuracy_score(Y_test,pred_E_BC)*100
print(bc)

#GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
model_E_GBC = GradientBoostingClassifier(n_estimators=200,learning_rate=.02)
model_E_GBC.fit(principalDf_train,Y_train)
pred_E_GBC = model_E_GBC.predict(principalDf_test)
gbcacc = metrics.accuracy_score(Y_test,pred_E_GBC)*100
print(gbcacc)


#VotingClassifier

from sklearn.ensemble import VotingClassifier
model1 = DecisionTreeClassifier()
model2 = LogisticRegression()
model3 = SVC()
model_E_VC = VotingClassifier(estimators=[('DT',model1),('LR',model2),('SVC',model3)])
model_E_VC.fit(principalDf_train,Y_train)
pred_E_VC = model_E_VC.predict(principalDf_test)
vc=metrics.accuracy_score(Y_test,pred_E_VC)*100
print(vc)

accuracyScore = [pca_acc,lr_acc,knn_acc,gnb_acc,bnb_acc,svc_acc,tree_acc,tree1_acc,bacc,b1acc,bc,gbcacc,vc]
algoName = ['PCA', 'LR', 'KNN' , 'GNB', 'BNB' , 'SVM' , 'DT' , 'EDT', 'RF' , 'ERF','BC','GBC','VC']
plt.scatter(algoName, accuracyScore)
plt.grid()
plt.title('Algorithm Accuracy Comparision')
plt.xlabel('Algorithm')
plt.ylabel('Score in %')
plt.show()

print(svc_acc)
