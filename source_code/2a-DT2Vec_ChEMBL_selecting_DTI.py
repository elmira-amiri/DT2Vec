# # Seleceting DTI from ChEMBL
# - CheMBL dataset: https://www.ebi.ac.uk/chembl/g/#browse/activities


import pickle
import pandas as pd

FILE_PATH = '.'
MIN_INTERACTION = 100 #min number of drug interaction
HIGHLY_POS = 5.5 #min positive'pChEMBL Value' 
HIGHLY_NEG = 0 #negative pChEMBL Value' 
MIN_pChEMBL_VALUE = 0

def all_unknown_drug(set_all_real_drug, set_all_real_target, dic_all_real_interactions):
    """
    This function create all unkown drug-target interactions and save it (.pkl)
    
    Args: 
        set_all_real_drug: list of unique drug names
        set_all_real_target: list of unique target names
        dic_all_real_interactions: dictionary of all known drug-target interactions    
    """
    
    first = 0
    step = 50000
    len_total = 0 
    # Sinece the data is big --> spilit it to small parts and save them separately
    for i in range(step, len(set_all_real_drug), step):
        set_all_real_drug_temp = set_all_real_drug[first:i]
        all_unknown_interactions = {(x,y) for x in set_all_real_drug_temp for y in set_all_real_target}
        all_unknown_interactions.difference_update(dic_all_real_interactions.keys())
        pd.DataFrame(all_unknown_interactions).to_csv('/unknown_interactions_'+ f'{first}_{i}.csv')
        with open(f'/unknown_interactions_'+f'{first}_{i}.pkl', 'wb') as fp:
            pickle.dump(all_unknown_interactions, fp)
        print(f'\nindex: {first}_{i}')
        print(f'len: {len(all_unknown_interactions)}')
        first = i 
        len_total += len(all_unknown_interactions)
        del all_unknown_interactions 
        del set_all_real_drug_temp

    if first < len(set_all_real_drug):
        set_all_real_drug_temp = set_all_real_drug[first:len(set_all_real_drug)]
        all_unknown_interactions = {(x,y) for x in set_all_real_drug_temp for y in set_all_real_target}
        all_unknown_interactions.difference_update(dic_all_real_interactions.keys())
        pd.DataFrame(all_unknown_interactions).to_csv('/unknown_interactions_'+ f'{first}_{len(set_all_real_drug)}.csv')
        with open(f'/unknown_interactions_'+ f'{first}_{len(set_all_real_drug)}.pkl', 'wb') as fp:
            pickle.dump(all_unknown_interactions, fp)
        print(f'\nindex: {first}_{len(set_all_real_drug)}')
        print(f'len: {len(all_unknown_interactions)}')
        len_total += len(all_unknown_interactions)
        del all_unknown_interactions 
        del set_all_real_drug_temp
   
    del dic_all_real_interactions 
    print(f'\nlen unkown DTI: {len_total}')
    return None


# ## ChEMBL
# 
# (https://www.ebi.ac.uk/chembl/g/#search_results/all)
# 
# ========== Selecting criteria part-1 ============
# 
# with:
# - Homo sapiens
# - SINGLE PROTEIN
# - Drugs: with at least 'MIN_INTERACTION' interaction (select those drugs that we have enough known interaction)
# 
# without:
# - Smiles
# 

ChEMBL = pd.read_csv(f'{FILE_PATH}/CHEMBL27_find_negative.csv', sep=';')
ChEMBL = pd.read_csv(f'{FILE_PATH}/CHEMBL27_find_negative.csv', sep=';')
ChEMBL = ChEMBL[ChEMBL['Target Type']=='SINGLE PROTEIN']
ChEMBL = ChEMBL[ChEMBL['Smiles'].notna()]

MIN_INTERACTION = 0

# Selecting those drug that we have a least n interactions (+/-)
df_num_drug_interaction = pd.DataFrame(ChEMBL['Molecule ChEMBL ID'].value_counts()).reset_index()
list_high_interacted_drug = list(df_num_drug_interaction[df_num_drug_interaction['Molecule ChEMBL ID']>=MIN_INTERACTION]['index'])
ChEMBL = ChEMBL[ChEMBL['Molecule ChEMBL ID'].isin(list_high_interacted_drug)]


# ## Finding positive DTI based on 'pChEMBL Value'
# 
# ========== Selecting criteria part-2 ============
# 
# with:
# - 'pChEMBL Value'
# - 'IC50' 
# -  Comment: ['Active', 'active', 'NaN']
# 
# without:
# - 'Data Validity Comment': ['Potential author error', 'Potential transcription error']
# 

ChEMBL_positive_interaction = ChEMBL.sort_values('pChEMBL Value', ascending=False).dropna(subset=['pChEMBL Value'])
ChEMBL_positive_interaction = ChEMBL_positive_interaction[ChEMBL_positive_interaction['Comment'].isin(['Active','active'])|ChEMBL_positive_interaction['Comment'].isnull()]
ChEMBL_positive_interaction = ChEMBL_positive_interaction[ChEMBL_positive_interaction['Data Validity Comment'].isnull()]                            
ChEMBL_positive_interaction = ChEMBL_positive_interaction[ChEMBL_positive_interaction['Standard Type'].isin(['IC50'])]

# Select all interaction with positive 'pChEMBL Value'(as an weighted interaction)
ChEMBL_positive_interaction = ChEMBL_positive_interaction[ChEMBL_positive_interaction['pChEMBL Value']>= MIN_pChEMBL_VALUE]
duplicateRows_df = ChEMBL_positive_interaction[ChEMBL_positive_interaction.duplicated(['Molecule ChEMBL ID','Target ChEMBL ID'])]
# Removing duplicate negative interactions
ChEMBL_positive_interaction['tuple_of_int'] = ChEMBL_positive_interaction[['Molecule ChEMBL ID', 'Target ChEMBL ID']].apply(tuple, axis=1)
dic_ChEMBL_positive_interaction = pd.Series(ChEMBL_positive_interaction['pChEMBL Value'].values, index=ChEMBL_positive_interaction['tuple_of_int']).to_dict()

ChEMBL_positive_interaction[['Molecule ChEMBL ID','Target ChEMBL ID','pChEMBL Value', 'Comment']].head()


# ## Negative samples from ChEMBL
# 
# ========== Selecting criteria part-3 ============
# 
# with: 
# - 'Comment':['Not Active','Not Active (inhibition < 50% @ 10 uM and thus dose-reponse curve not measured)',
# 'inactive','Inactive']
# 
# without:
# - 'pChEMBL Value'
# - 'Data Validity Comment': ['Outside typical range']
# 

# Selecting negative interaction  
list_neg = ['Not Active (inhibition < 50% @ 10 uM and thus dose-reponse curve not measured)','Not Active','inactive', 'Inactive']
ChEMBL_negative_interaction = ChEMBL[ChEMBL['Comment'].isin(list_neg)][['Molecule ChEMBL ID','Target ChEMBL ID','pChEMBL Value', 'Comment', 'Data Validity Comment','Standard Type']]
ChEMBL_negative_interaction = ChEMBL_negative_interaction [ChEMBL_negative_interaction['pChEMBL Value'].isnull()]
ChEMBL_negative_interaction = ChEMBL_negative_interaction[ChEMBL_negative_interaction['Data Validity Comment'].isnull()]

# Removing duplicate negative interactions
ChEMBL_negative_interaction['tuple_of_int'] = ChEMBL_negative_interaction[['Molecule ChEMBL ID', 'Target ChEMBL ID']].apply(tuple, axis=1)
dic_ChEMBL_negative_interaction = pd.Series(ChEMBL_negative_interaction['pChEMBL Value'].values, index=ChEMBL_negative_interaction['tuple_of_int']).to_dict()

ChEMBL_negative_interaction[['Molecule ChEMBL ID', 'Target ChEMBL ID', 'pChEMBL Value', 'Comment']].head()


# ## All drug-target interactions (+/-)

# All konown negative and positive interactions
ChEMBL_negative_interaction = ChEMBL_negative_interaction[['Molecule ChEMBL ID','Target ChEMBL ID','pChEMBL Value']].fillna(0)
ChEMBL_positive_interaction = ChEMBL_positive_interaction[['Molecule ChEMBL ID','Target ChEMBL ID','pChEMBL Value']]
all_real_interactions = ChEMBL_negative_interaction.append(ChEMBL_positive_interaction)

all_real_interactions.head()


ChEMBL = pd.read_csv(f'{FILE_PATH}/CHEMBL27_find_negative.csv', sep=';') 
phase_dict = dict(zip(ChEMBL['Molecule ChEMBL ID'], ChEMBL['Molecule Max Phase']))


# ### More analysis for discusstion


CHEMBL3869 = all_real_interactions[all_real_interactions['Target ChEMBL ID']=='CHEMBL3869']
CHEMBL3869 = CHEMBL3869[CHEMBL3869['pChEMBL Value']>=5.5]
CHEMBL3869['phase'] = CHEMBL3869['Molecule ChEMBL ID'].map(phase_dict)
CHEMBL3869.to_csv('CHEMBL3869_drugs.csv')

all_real_interactions[all_real_interactions['Molecule ChEMBL ID']=='CHEMBL1873475']['Target ChEMBL ID'].value_counts()

# Assigned the averaged for DTI with multiple pChEMBL Value
all_real_interactions['tuple_of_int'] = all_real_interactions[['Molecule ChEMBL ID', 'Target ChEMBL ID']].apply(tuple, axis=1)
dic_all_real_interactions = pd.Series(all_real_interactions['pChEMBL Value'].values, index=all_real_interactions['tuple_of_int']).to_dict()

duplicated_pChEMBL_Value = pd.DataFrame(all_real_interactions['tuple_of_int'].value_counts())
duplicated_pChEMBL_Value[duplicated_pChEMBL_Value['tuple_of_int']>1] 

for i in duplicated_pChEMBL_Value.index:
    median = all_real_interactions[all_real_interactions['tuple_of_int']==i]['pChEMBL Value'].median()
    mean = all_real_interactions[all_real_interactions['tuple_of_int']==i]['pChEMBL Value'].mean()
    if (median == 0.0) and (mean < 1.5): 
        avg_pChEMBL = 0.0
    else:
        avg_pChEMBL = mean
    
    dic_all_real_interactions[i] = avg_pChEMBL  

df_all_real_DTI = pd.DataFrame.from_dict(dic_all_real_interactions, orient='index', columns=['weight'])
df_all_real_DTI.index = pd.MultiIndex.from_tuples(df_all_real_DTI.index)
df_all_real_DTI.index.names = ['from', 'to']
df_all_real_DTI = df_all_real_DTI.reset_index()

df_all_real_DTI.head()


# Selecting strong postive and negative interactions 
df_high_pos = df_all_real_DTI[df_all_real_DTI['weight']>= HIGHLY_POS]
df_high_neg = df_all_real_DTI[df_all_real_DTI['weight']<= HIGHLY_NEG]


real_drug = len(set(all_real_interactions['Molecule ChEMBL ID']))
print(f'Total Num of unique drug: {real_drug}')

real_gene = len(set(all_real_interactions['Target ChEMBL ID']))
print(f'Total Num of unique gene: {real_gene}')

print(f'Num of possible DTI: {real_gene*real_drug}\n')

print(f'Num of highly + interactions >= {HIGHLY_POS}: {len(df_high_pos)}')
print(f'Num of + interactions>{HIGHLY_NEG } and <{HIGHLY_POS}: {len(dic_all_real_interactions)-len(df_high_neg)-len(df_high_pos)}')
print(f'Num of - interactions: {len(df_high_neg)}\n')


print(f'Total Num of DTI: {len(dic_all_real_interactions)} (for creating embedding graph/training and test)')
print(f'Total Num of highly + and - : {len(df_high_neg)+len(df_high_pos)} (for binary classification)')
print(f'Total Num of unkown DTI: {real_gene*real_drug- len(dic_all_real_interactions)} (for prediction/experimental)')

# Creating dictionary from ChEMBL-ID to smiles (removed dublicated drugs ID)
ChEBL2Smiles_tmp = pd.Series(ChEMBL['Smiles'].values, index=ChEMBL['Molecule ChEMBL ID']).to_dict()
keys = set(df_all_real_DTI['from']).intersection(set(ChEBL2Smiles_tmp.keys()))
df_ChEBL2Smiles = pd.DataFrame.from_dict({k:ChEBL2Smiles_tmp[k] for k in keys}, orient='index').reset_index()
df_ChEBL2Smiles.to_csv('/SMILES_main.txt', sep='\t', index=False, header=False)

df_Smilesdict2ChEBL = pd.DataFrame()
df_Smilesdict2ChEBL['index'] = df_ChEBL2Smiles[0].copy()
df_Smilesdict2ChEBL[0] = df_ChEBL2Smiles['index'].copy()
df_Smilesdict2ChEBL.to_csv('/temp_SMILES_main.txt', sep='\t', index=False, header=False)

df_all_real_DTI.to_csv('/DTI_known_ChEMBLid_originalID.csv')
df_all_real_DTI['from'] = df_all_real_DTI['from'].str.replace('CHEMBL', '')
df_all_real_DTI.to_csv('/DTI_known_ChEMBL.csv', index= False)

# ## Selecting unknown interactions (prediction-dataset)

set_all_real_target = list(set(all_real_interactions['Target ChEMBL ID']))
set_all_real_drug = list(set(all_real_interactions['Molecule ChEMBL ID']))

# Run this once to find all unknown interation between all drugs and targets
all_unknown_drug(set_all_real_drug, set_all_real_target, dic_all_real_interactions)




