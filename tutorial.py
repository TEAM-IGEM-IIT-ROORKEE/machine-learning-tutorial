# -*- coding: utf-8 -*-

#importing necessary dependencies and libraries
import pandas as pd
import numpy as np
from scipy.io import loadmat

#different files 

train_filename = './training.xlsx' 
annotation_filename = './identify_match.xlsx'
chemogenomics_filename = './phenotype_data.xlsx'
validation_filename = './validation.xlsx'

#chemogenomics data reading

chemogenomics_file = pd.read_excel(chemogenomics_filename) # reading chemogenomics data file
print('Chemogenomics - Sample')
print(chemogenomics_file.head())

probelist = chemogenomics_file.iloc[1:,0] # list of genes
print('List of Genes - Sample')
print(probelist.head())
probelist = np.array(probelist)

conditions = list(chemogenomics_file.keys()[1:]) #list of antibiotics or drug-conditions
print('List of Antibioitcs - Sample')
print(conditions)
conditions = np.array(conditions)

plist = np.empty_like(probelist)
    
for i in range(probelist.shape[0]):
    tem = probelist[i].split('-')
    try:
        plist[i] = tem[1]
    except:
        plist[i] = tem[0]

#already quantile normalized data
pnum_array = loadmat('./preprocessed.mat') #importing data from matlab after quantile normalization processing
phenotype_num = np.array(pnum_array['phenotype_num'])

#phenotype_data : interaction scores between drugs and genes
phenotype_data = np.load('./phenotype_data.npy')

#phenotype_labels : labels of different genes which have atleast one significant interaction with drugs
phenotype_labels = np.load('./phenotype_labels.npy')

#preparing training dataset

train_file = pd.read_excel(train_filename)
print('Training Dataset')
print(train_file)

interaction_pairs = np.array(train_file.iloc[:,:2]) #storing interaction pair
interaction_score = np.array(train_file.iloc[:,2]) #storing scores of particular interaction pair
drugs_all = np.unique(interaction_pairs) # finding unique drugs present in list of interaction pairs

annotations = pd.read_excel(annotation_filename)

drug_id = np.array(annotations.iloc[:,0]) #drug id, used in interaction data

chemgen_id = np.array(annotations.iloc[:,1]) # chemgen id, used in chemogenomics data

drugnames_all = np.copy(drugs_all) #copying, used to have two different arrays with unique entries one with abbr and one with detailed

for i in range(drug_id.shape[0]):
    drugnames_all[drugnames_all==drug_id[i]] = chemgen_id[i]

drugpairsname_cell = np.copy(interaction_pairs)

for i in range(drug_id.shape[0]):
    drugpairsname_cell[drugpairsname_cell==drug_id[i]] = chemgen_id[i]

ix = np.zeros(drugpairsname_cell.shape) # if drug at particular position is present in our list of data, it turns the value here to 1 else 0

for i in range(drugpairsname_cell.shape[0]):
    for j in range(drugpairsname_cell.shape[1]):
        if drugpairsname_cell[i,j] in conditions:
            ix[i,j] = 1
    
ix = ix.astype(np.uint8)
ix = np.where(ix[:,0]*ix[:,1]) #finding index in drug pairs cell where both drugs have data available

train_interactions = drugpairsname_cell[ix] #list of pairs whose data is available and hence can be used for training the model
    
train_scores = interaction_score[ix] #interaction score corresponding to interaction pairs

traindrugs = np.unique(train_interactions) #finding unique drugs in whole training set

pos = np.zeros(traindrugs.shape) # with the idea to keep the indexes where data of particular drug of training set lies in phenotype data

for i in range(pos.shape[0]):
    pos[i] = np.where(conditions == traindrugs[i])[0][0]
    
pos = pos.astype(np.uint16)

trainchemgen = phenotype_data[:,pos] #dataset of the whole phenotype relevant to training set
trainchemgen = trainchemgen.astype(np.uint8)

#preparing validation dataset

validation_file = pd.read_excel(validation_filename) #loading test interaction data
print('Validation Dataset')
print(validation_file.head())

validation_pairs = np.array(validation_file.iloc[:,:2])
validation_score = np.array(validation_file.iloc[:,2])

val_drugs_all = np.unique(validation_pairs) #finding unique drugs in test interaction data

val_drugnames_all = np.copy(val_drugs_all) 

for i in range(drug_id.shape[0]):
    val_drugnames_all[val_drugnames_all==drug_id[i]] = chemgen_id[i] #replacing unique drug names with the one in the official chemogenomics data

val_drugpairnames_all = np.copy(validation_pairs)

for i in range(drug_id.shape[0]):
    val_drugpairnames_all[val_drugpairnames_all==drug_id[i]] = chemgen_id[i] #replacing drug interaction pair names with official chemogenomics data
    
val_ix = np.zeros(val_drugpairnames_all.shape) #finding which test interactions pairs have their complete data in chemogenomics profile

for i in range(val_drugpairnames_all.shape[0]):
    for j in range(val_drugpairnames_all.shape[1]):
        if val_drugpairnames_all[i,j] in conditions:
            val_ix[i,j] = 1

val_ix = val_ix.astype(np.uint8)
val_ix = np.where(val_ix[:,0]*val_ix[:,1])

val_interactions = val_drugpairnames_all[val_ix] #list of test interaction pairs whose complete data is available
    
val_scores = validation_score[val_ix] #list of interaction scores

valdrugs = np.unique(val_interactions) # list of unique drugs present in test dataset

val_pos = np.zeros(valdrugs.shape) #finding index where data of that drug is located

for i in range(val_pos.shape[0]):
    val_pos[i] = np.where(conditions == valdrugs[i])[0][0]

val_pos = val_pos.astype(np.uint16)

valchemgen = phenotype_data[:,val_pos]
valchemgen = valchemgen.astype(np.uint8)

#training dataset

xtrain = []
ytrain = []
for i in range(train_interactions.shape[0]):
    ix1 = []
    list_drugs = train_interactions[i]
    for d in list_drugs:
        if d in traindrugs:
            index = np.where(traindrugs==d)[0][0]
            ix1.append(index)

    t1 = trainchemgen[:,ix1[0]]
    t2 = trainchemgen[:,ix1[1]]
    sigma = t1 + t2 #calculating similarity
    delta = t1 + t2 
    delta[delta!=1] = 0 #calculating uniqueness

    t3 = np.concatenate((sigma,delta),axis=0)
    
    xtrain.append(t3)
    ytrain.append(train_scores[i])

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

#validation dataset

xval = []
yval = []

for i in range(val_interactions.shape[0]):
    ix1 = []
    list_drugs = val_interactions[i]
    for d in list_drugs:
        if d in valdrugs:
            index = np.where(valdrugs==d)[0][0]
            ix1.append(index)

    t1 = valchemgen[:,ix1[0]]
    t2 = valchemgen[:,ix1[1]]
    sigma = t1 + t2
    delta = t1 + t2
    delta[delta==2] = 0
    t3 = np.concatenate((sigma,delta),axis=0)
    
    xval.append(t3)
    yval.append(val_scores[i])

xval = np.array(xval)
yval = np.array(yval)


#import dependencies or necessary libraries
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score # to check coefficient of determination
from scipy.stats import spearmanr #to check for spearman correlation coefficient and p-value for significance

#initialization of model
rf_model = RandomForestRegressor()

#training our model
rf_model.fit(xtrain,ytrain)

#validation of our model
ptrain = rf_model.predict(xtrain)
pval = rf_model.predict(xval)

#performance evaluation
print('Training Dataset Metrics\n')
print('Coefficient of Determination: ',r2_score(ytrain,ptrain))
print('Pearsons Correlation: ',np.corrcoef(ytrain,ptrain)[0,1])
print('Spearman Correlation: ',spearmanr(ytrain,ptrain)[0])
print('p-value: ',spearmanr(ytrain,ptrain)[1])
print('\n\n')

print('Validation Dataset Metrics\n')
print('Coefficient of Determination: ',r2_score(yval,pval))
print('Pearsons Correlation: ',np.corrcoef(yval,pval)[0,1])
print('Spearman Correlation: ',spearmanr(yval,pval)[0])
print('p-value: ',spearmanr(yval,pval)[1])

#importing library
import matplotlib.pyplot as plt

#training set plot
plt.scatter(ytrain,ptrain)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.plot()

#clearing matplotlib plots
plt.clf()

#validation set plot
plt.scatter(yval,pval)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.plot()

#providing labels to each and every feature

labels = []

for i in range(xtrain.shape[-1]):
  if i < 3853:
    labels.append('Sigma_'+phenotype_labels[i])
  else:
    labels.append('Delta_'+phenotype_labels[i-3853])

labels = np.array(labels)

#feature importance array

feature_importance = rf_model.feature_importances_

#finding the top 20 features
top_indexes = feature_importance.argsort()[-20:][::-1]

top_features = labels[top_indexes]
top_features = top_features.reshape(-1,1)

print('Top Features \n')
print(top_features)

#conclusion of best feature
print('The feature is ',top_features[0,0].split('_')[0],' score of gene ',top_features[0,0].split('_')[1])
