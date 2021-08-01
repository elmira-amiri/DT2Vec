#!/usr/bin/env python
# coding: utf-8

import csv
import subprocess
import re
import os
import pandas as pd
from itertools import chain
from tqdm import tqdm
import numpy as np
import pickle

# Reading DTI
DTI = pd.read_csv('DTI_known_ChEMBL.csv')
print (f'Number of known +/- Drug-target interactions: {len(DTI)}')
DTI.head()

# ChEMBL_id to Entrez_id (from uniprot)
with open('ChEMBL2Entrez.pkl', 'rb') as f:
    ChEMBL2Entrez = pickle.load(f)
ChEMBL2Entrez

# Replacing ChEMBL ids with Entrez
DTI_temp = DTI.replace(ChEMBL2Entrez)
print(f'number of unique protein in DTI: {len(set(list(DTI_temp["to"])))}')
DTI_temp['to_ChEMBL'] = DTI["to"]
DTI_temp.head()

target_Entrez_ids = pd.DataFrame(list(set(DTI.replace(ChEMBL2Entrez)["to"])))
target_Entrez_ids.to_csv("Entrez_protein_DTI.csv",index=False, header=False)

