# Machine Learning Tutorial

## Repository of Machine Learning presentation and tutorial presented by iGEM IIT Roorkee during All India iGEM Meet (AIIM) 2020

**The link to the presentation, [here](https://docs.google.com/presentation/d/1PHBz7g_uvkript21ROB1X8EltrD39bWBa56pXOJ5rC4/edit?usp=sharing)** 

**Brief about the Tutorial**

The tutorial is inspired from the following paper, **Chandrasekaran *et al.*, "Chemogenomics and orthology‐based design of antibiotic combination therapies", *Mol Syst Biol (2016)12:872***, you can view the paper [here](https://www.embopress.org/doi/10.15252/msb.20156777).

It uses chemogenomic data profile of various antibiotics *i.e.* interaction between chemical drug and gene to predict interaction score between two antibiotics in *Escherichia coli*.

```
#importing necessary dependencies and libraries
import pandas as pd
import numpy as np
from scipy.io import loadmat
```

**Chemogenomic Profiling**

Chemogenomic profile measures fitness of gene‐knockout strains treated with a particular antibiotic.

The chemogenomic data profile of different antibiotics is made available from, **Nichols *et al.*, "Phenotypic landscape of a bacterial cell", *Cell 144: 143-156 (2011)***, you can view the paper [here](https://www.cell.com/fulltext/S0092-8674(10)01374-7). The chemogenomic dataset provided has data of 324 different drug conditions and 3979 genes. 

Here is the sample of chemogenomic profile of Amoxicillin in different conditions with different genes

![alt text](https://drive.google.com/uc?export=view&id=1ceOfjrS3L5vmm4UP_heQRCHhvU0FTUQt)

```
#different files 

train_filename = './training.xlsx' 
annotation_filename = './identify_match.xlsx'
chemogenomics_filename = './phenotype_data.xlsx'
validation_filename = './validation.xlsx'
```

```
#chemogenomics data reading

chemogenomics_file = pd.read_excel(chemogenomics_filename) # reading chemogenomics data file
print('Chemogenomics - Sample')
print(chemogenomics_file.head())
```

```
probelist = chemogenomics_file.iloc[1:,0] # list of genes
print('List of Genes - Sample')
print(probelist.head())
probelist = np.array(probelist)
```

```
conditions = list(chemogenomics_file.keys()[1:]) #list of antibiotics or drug-conditions
print('List of Antibioitcs - Sample')
print(conditions)
conditions = np.array(conditions)
```

**Dataset Preprocessing**

The given dataset is preprocessed using **quantile normalization** to make all different distributions identical

The drug-gene interaction scores less than -2 were considered significant for any particular drug and gene pair. Hence, this conversion leads to formation of a binary dataset. The binary dataset of above sample of chemogenomic profile is displayed below,

![alt text](https://drive.google.com/uc?export=view&id=1F3EUDFeaH294haLMMMrNOiIRr2x77F3t)

Due to this preprocessing, the number of genes in the dataset are reduced to 3853 from 3979, since for 126 genes, there was not even a single drug condition, for whom there is no significant interaction score *i.e.* less than -2.

```
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
```

Try to run either A or B part below.
A is just an extended version of B, *i.e.* the arrays used directly in B were obtained after running code part A

**Part A**
```
#usual preprocessing technique 
#it would take a lot of time, so skipping this part by utilizing the end product iself

cell_z1_list = [] #to capture the list of genes which are more sensitive
lte = [] #to capture list of such genes for every particular condition
list_phenotype = [] #to find the unique list of such genes

for i in range(phenotype_num.shape[-1]):
    te = plist[phenotype_num[:,i]<-1*z] # for particular condition, finding the most important genes
    cell_z1_list.append(te) # appending list of such genes in list
    lte.append(len(te)) # keeping in account the number of such genes for particular condition
    for j in range(len(te)): 
        list_phenotype.append(te[j]) # for finding the number of such genes in the whole dataset
        
np_array = np.array(list_phenotype)
phenotype_labels = np.unique(np_array) #list of genes in whole dataset

nichols_t = np.zeros((phenotype_labels.shape[0],phenotype_num.shape[-1])) #storing binary dataset

for i in range(nichols_t.shape[0]):
    for j in range(nichols_t.shape[-1]):
        for k in range(len(cell_z1_list[j])):    
            if phenotype_labels[i] == cell_z1_list[j][k]:
                nichols_t[i,j] = 1
                
phenotype_data = np.copy(nichols_t)
```
**Part B**
```
#phenotype_data : interaction scores between drugs and genes
phenotype_data = np.load('./phenotype_data.npy')

#phenotype_labels : labels of different genes which have atleast one significant interaction with drugs
phenotype_labels = np.load('./phenotype_labels.npy')
```

**Drug Combination Dataset**

The dataset used in this tutorial is same as the one used in **Chandrasekaran *et al.*, 2016** which has pair-wise combination of 19 different antibiotics, *i.e.* 171 combinations. The interaction score of a pair of antibiotics is calculated using **Loewe additive model**.

The details of 19 different antibiotics is shown in table below. 

![alt text](https://drive.google.com/uc?export=view&id=1gLj4l7wHMt0K9Y35HChz7o9R_Qdb_MCI)

*Note : This table is adopted from the same paper mentioned above, and is exactly the same as Table 1 of the paper*

**Dataset Division**

In every machine learning setting, dataset is generally divided into two parts,

*   **Training dataset:** This portion of dataset is used in training any machine learning algorithm and learn different parameters and weights involved in the model. Pairwise combinations involving 15 drugs (shown in table) *i.e.* 105 combinations, are taken as training dataset.

*   **Validation dataset:** This portion of dataset is used in validating machine learning model for real-world performance. Pairwise combinations involving 4 drugs (shown in italics in table) *i.e.* total of 6 combinations among themselves and 60 combinations with those drugs in training dataset which makes it total of 66 combinations in validation set.

Here is the overview of a simple machine learning approach


![alt text](https://drive.google.com/uc?export=view&id=1JvtEp2MPe7nEYWvNWk417-pX9koTfeK3)

```
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
```

```
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
```






