#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pickle
import numpy as np


# In[ ]:


FILE_PATH = '.'


# In[ ]:


def similarity_interaction(df_tmp):
    
    """
    This function will clean protein-protein similarity. 
    
    Args: protein-protein similarity dataframe

    Returns: cleran protein-protein similarity dataframe
    
    """
    
    # number of unique genes before cleaning data
    print(f'number of drug/gene is :{len(set(list(df_tmp)+ df_tmp.index))}') 
        
    df_tmp.columns = pd.Index(map(lambda x : x.lstrip('hsa'), df_tmp.columns))
    df_tmp.index = pd.Index(map(lambda x : x.lstrip('hsa'), df_tmp.index))

    # set diagonal as NAN (similarity of drugs with itself)
    df_tmp.values[[np.arange(len(df_tmp))]*2] = np.nan 
    
    # stack() gives all links similarities
    df_tmp = df_tmp.stack().reset_index().rename(columns={'level_0':'to', 'level_1':'from', 0:'weight'}) 

    # remove dublicate similarity because a->b is same as b->a (remove rows with same (from,to))
    # combining (from,to) columns in a new coulmn as a list, and sort it 
    df_tmp['to_from'] = df_tmp[['to','from']].apply(list, axis=1).apply(sorted) 
    df_tmp['to_from'] = df_tmp['to_from'].astype(str)
    df_tmp = df_tmp.drop_duplicates(subset='to_from').drop(columns='to_from')
    
    # number of unique genes after cleaning data    
    print(f'number of drug-drug/gene-gene similarities: {df_tmp.shape[0]}')
    
    return(df_tmp)


# In[ ]:


# load the mapping (the dictionary from ChEMBL to Gene ID)
with open(f'{FILE_PATH}/ChEMBL2Entrez.pkl', 'rb') as f:
    ChEMBL2Entrez = pickle.load(f)
Entrez2ChEMBL = dict((y,x) for x,y in ChEMBL2Entrez.items())


# ### Creat edgelists

# #### 1- DTI

# In[ ]:


DTI = pd.read_csv(f'{FILE_PATH}/DTI_known_ChEMBL.csv')
DTI.head()


# In[ ]:


# find max durg id (protein IDs start after it)
max_drugId = DTI['from'].max()

# create a dictionary for mapping protein IDs to numeric ID (protein IDs start from: max_drugId+1)
protein_ID = list(set(DTI['to']))
protein2num = {k: v+max_drugId+1 for v, k in enumerate(sorted(protein_ID))}

with open(f'{FILE_PATH}protein2num.pkl', 'wb') as fp:
            pickle.dump(protein2num, fp)

DTI_newID = DTI.replace(protein2num)
DTI_newID.to_csv(f'{FILE_PATH}edgelis/DTI.csv')

DTI_newID.head()


# #### 2- PPS

# In[ ]:


PPS_seq = pd.read_csv(f'{FILE_PATH}/PPS(seq)_known_ChEMBLid_P556.csv', index_col=0)
PPS_seq


# In[ ]:


# Creating Edgelist for protein-protein similarity
ChEMBL_id = pd.read_csv(f'{FILE_PATH}/uniprot_Seq.csv')
protein_list = list(ChEMBL_id["ChEMBL_ID"])

PPS_seq = pd.read_csv(f'{FILE_PATH}/PPS(seq)_known_ChEMBLid_P{len(protein_list)}.csv', index_col=0)
PPS_seq.columns = list(ChEMBL_id["ChEMBL_ID"])
PPS_seq["index"] = list(ChEMBL_id["ChEMBL_ID"])
PPS_seq = PPS_seq.set_index("index")
PPS_seq = PPS_seq.rename_axis(None)
#PPS_seq_edglist = similarity_interaction(PPS_seq)
#PPS_seq_edglist.to_csv(f'{FILE_PATH}/PPS_seq_edglist_df.csv')
#PPS_seq_edglist.head()


# In[ ]:


# change protein IDs to numeric IDs
PPS_seq_newID = PPS_seq_edglist.replace(protein2num)
PPS_seq_newID.to_csv(f'{FILE_PATH}/edgelis/PPS_seq.csv')

PPS_seq_newID.head()

