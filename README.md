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
