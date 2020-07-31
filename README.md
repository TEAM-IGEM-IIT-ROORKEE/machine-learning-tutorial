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

Try to run either A or B part below.
B is just an extended version of A, *i.e.* the arrays used in A were obtained after running code part B

**Part A**
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

**Part B**
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

```
#phenotype_data : interaction scores between drugs and genes
phenotype_data = np.load('./phenotype_data.npy')
```

```
#phenotype_labels : labels of different genes which have atleast one significant interaction with drugs
phenotype_labels = np.load('./phenotype_labels.npy')
```


