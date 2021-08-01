# # Drug-drug similarity
# - https://www.nature.com/articles/nprot.2014.151

import csv
import subprocess
import re
import os
import pandas as pd
from itertools import chain
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

T_CUTOFF = 0 # Specify the TC cutoff. This option is useful if only the TCs of similar molecules above the established cutoff are needed. Otherwise, set T_CUTOFF=0 to provide all TC pair values.
FINGERPRINT = 'MACCS' # Specify fingerprint (‘FP2’, ‘FP3’, ‘FP4’ or ‘MACCS’)

def sim_one_ChEMBL(FINGERPRINT, T_CUTOFF): 
    """
    This function calculates similarity of a drug with the other drugs 
    
    Args: 
        FINGERPRINT: Fingerprint (‘FP2’, ‘FP3’, ‘FP4’ or ‘MACCS’)
        T_CUTOFF: The established cutoff

    Returns: A list of pairs of compounds and the relevant TC that quantifies the level of similarity between them.
    """
    
    input_temp = open('temp_SMILES.txt', 'r')

    # Create a dictionary of chemicals to be compared:
    input_dict = dict()

    # Read the input and the files previously created:
    for line in input_temp:
        newline = line.split()
        if len(newline) != 2:
            continue
        smiles = newline[0]
        id = newline[1]
        input_dict[id] = smiles

    input_temp.close()

    # Open the results file (.csv file):
    f = open('/TC_results.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['chemical1', 'chemical2', 'TC'])

    # For each chemical in input list, calculate the TC between that chemical and all other chemicals in the
    # input list using Open Babel:
    for chemical1 in tqdm(input_dict):
        babel_command = 'obabel -ismi -:"%s" temp_SMILES.txt -ofpt -xf%s' %(input_dict[chemical1], FINGERPRINT)
        output = subprocess.Popen(babel_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read and parse output from Open Babel:
    TC_list = []
    TC_list_tmp = []

    while True:
        line = output.stdout.readline().decode('ascii')
        if line != '':
            newline = re.split('>|=', line)
           

            if len(newline) > 2:
                id_catcher = newline[1].split()
                chemical2 = id_catcher[0]
                TC = float(newline[2])
                TC_list.append((chemical2, TC))
        else:
            break

    # Write the TCs exceeding the cutoff to the output file
    for chemical2, TC in TC_list:
        if TC >= T_CUTOFF and chemical1 != chemical2:
            writer.writerow([chemical1, chemical2, TC])
            TC_list_tmp.append((chemical1, chemical2, TC))
            
    del TC_list
    f.close()
    
    return TC_list_tmp


df_SMILES_main = pd.read_csv('temp_SMILES_main.txt', sep='\t', header=None)
# a copy of all SMILES. It will be updated to create the DDS matrix
df_SMILES_main.to_csv('temp_SMILES.txt',sep='\t', index=False, header=False)
SMILES_id = df_SMILES_main[0].to_list()[::-1]
print(f"Number of drug: {len(SMILES_id)}\n")

df_SMILES_main.head()

i = 0
TC_list_final = [] # All drug-drug similarity
for c in SMILES_id[:-1]:
    i += 1
    TC_list_tmp = sim_one_ChEMBL(FINGERPRINT, T_CUTOFF)
    TC_list_final.append(TC_list_tmp)
    
    # After calculating the drug (c) similarity, it will be removed from list (we have already added the similarity of durg (c) with all drugs) 
    df = pd.read_csv('temp_SMILES.txt', skipfooter = 1, engine='python',sep='\t', header=None)
    df.to_csv('temp_SMILES.txt',sep='\t', index=False, header=False)
    print(i)
    
    # Because the data is big, it is collected gradually 
    if (i % 100 == 0) | (i>len(SMILES_id)-10):
        df_TC_list_final = pd.DataFrame(list(chain(*TC_list_final)))
        df_TC_list_final[0] = df_TC_list_final[0].str.replace('CHEMBL', '')
        df_TC_list_final[1] = df_TC_list_final[1].str.replace('CHEMBL', '')
        df_TC_list_final.to_csv(f'DDS_known_ChEMBLid{i}_T{len(SMILES_id)}.csv')

DDS = pd.read_csv('DDS_known_ChEMBLid_T548.csv')
DDS.head()

