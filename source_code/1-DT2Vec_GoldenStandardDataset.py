# # DT2Vec on Golden-standard dataset
# - GSD is available at:  http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/  


import os
import csv
import timeit
import networkx
import community
import numpy as np
import pandas as pd
from ggplot import *
import networkx as nx
import scipy.sparse as sps
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    matthews_corrcoef,
    auc,
    average_precision_score)
from sklearn import metrics
from matplotlib import pyplot
from collections import Counter
from sklearn.metrics import roc_curve
from community import community_louvain
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from xgboost import XGBClassifier, plot_importance
from sknetwork.clustering import Louvain, BiLouvain, modularity, bimodularity
from sknetwork.utils import bipartite2undirected, membership_matrix
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph
from sknetwork.hierarchy import LouvainHierarchy, BiLouvainHierarchy
from sknetwork.hierarchy import cut_straight, dasgupta_score, tree_sampling_divergence
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph, svg_dendrogram
from sklearn.decomposition import PCA


SCALE_POS_RATIO = False # If we have an imbalance dataset change it to True 
MAIN_DIR = '.'
os.chdir(MAIN_DIR)


# #  Functions
def calculate_metrics(y, y_pred):
    """
    This function calculates machine learning metrics
    
    Args: real lables and predicted labels

    Returns: A dictionary containing all the results
    """
    
    Precision = precision_score(y, y_pred, average='binary')
    Recall = recall_score(y, y_pred, average='binary')
    result = {
        'accuracy': metrics.accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='binary'),
        'recall': recall_score(y, y_pred, average='binary'),
        'f1_score': f1_score(y, y_pred, average='binary'),
        'average_precision_score': average_precision_score(y, y_pred),
        #'f2_score':(5 * Precision * Recall) / (4 * Precision + Recall)
    }
    
    try:
        result['ROC'] = metrics.roc_auc_score(y, y_pred)
    except:
        pass

    return result


def downloading_data(): 
    """
    This function downloads the golden-standard dataset from 
    websit 'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/'
    and save them in 'MAIN_DIR'
    
    """
    
    file_ids = [
        'bind_orfhsa_drug_e.txt', 'bind_orfhsa_drug_ic.txt', 'bind_orfhsa_drug_gpcr.txt', 'bind_orfhsa_drug_nr.txt',
        'e_simmat_dc.txt', 'ic_simmat_dc.txt', 'gpcr_simmat_dc.txt', 'nr_simmat_dc.txt',
        'e_simmat_dg.txt', 'ic_simmat_dg.txt', 'gpcr_simmat_dg.txt', 'nr_simmat_dg.txt',
        'e_admat_dgc.txt', 'ic_admat_dgc.txt', 'gpcr_admat_dgc.txt', 'nr_admat_dgc.txt'
    ]

    for file_id in file_ids:
        print(f'Downloading file {file_id}')

        data_endpt = f'http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/{file_id}'
        print(data_endpt)
        response = requests.get(data_endpt)

        with open(file_id, "wb") as output_file:
            output_file.write(response.content)

    # Convert files to csv
    for file_id in file_ids:
        print(f'Converting file {file_id[:-4]}.csv')

        with open(file_id, 'r', encoding='mac_roman') as fin:
            cr = csv.reader(fin, delimiter='\t')
            # Replace commas with full stops

            filecontents = [line for line in cr]
            filecontents_new = list()
            for x in filecontents:
                line = [line.replace(',', '.x') for line in x]
                filecontents_new.append(line)

        with open(file_id[:-4] + '.csv', 'w') as fou:
            cw = csv.writer(fou, quotechar='', quoting=csv.QUOTE_NONE, escapechar='\\')
            cw.writerows(filecontents_new)


def reading_interarction():
    """
    This function integarat different DTI datasets in 
    golden-standard dataset (e, ic, gpcr, and nr)
    
    Returns: A DataFrame containing all DTI
    """
    
    file_list = ['bind_orfhsa_drug_e', 'bind_orfhsa_drug_ic', 'bind_orfhsa_drug_gpcr','bind_orfhsa_drug_nr']
    df = pd.DataFrame()
    for file in file_list:
        df_tmp = pd.read_csv(f'{file}.csv', header= None)
        print(f'len of {file} is {df_tmp.shape[0]}')
        
        df = df.append(df_tmp)
        
    df = df.rename(columns={0:'to', 1:'from'})
    df['to'] = df['to'].map(lambda x: x.lstrip('hsa:'))
    df['weight'] = 1
    return df



def reading_similarity(file_list):
    """
    This function integrates similarity matrix and convertes to a DataFrame
    
    Args: Drug/target similarity file names

    Returns: A DataFram containing of similarities 
    """

    df_total = pd.DataFrame()
    for file in file_list:
        df_tmp = pd.read_csv(f'{file}.csv').set_index(['Unnamed: 0'])
        print(f'number of drug/gene in {file} file is :{len(set(list(df_tmp) + df_tmp.index))}')

        # remove "hsa" for genes
        df_tmp.columns = pd.Index(map(lambda x: x.lstrip('hsa'), df_tmp.columns))
        df_tmp.index = pd.Index(map(lambda x: x.lstrip('hsa'), df_tmp.index))

        df_tmp.values[[np.arange(len(df_tmp))] * 2] = np.nan
        df_tmp = df_tmp.stack().reset_index().rename(columns={'level_0': 'to', 'level_1': 'from', 0: 'weight'})
        df_tmp = df_tmp[df_tmp['weight'] != 0]

        df_total = df_total.append(df_tmp)

    num_total = len(set(list(df_total['from']) + list(df_total['to'])))           
    print(f'\nTotal number of unique gene/drug: {num_total}')

    return df_total



def mergeing_sim_matrix (df, input_data= 'drug'):
    """
    This function merges drugs/targets similarity as an adjacency matrix
    
    Args: 
        df: A DataFram containing of similarities
        input_data: The type of data ('drug' or 'target')
    
    Returns: Drugs/targets adjacency matrix
    """
    nodes = df.iloc[:, 0].tolist() + df.iloc[:, 1].tolist()
    nodes = sorted(list(set(nodes)))
    node_dic = {k: v for v, k in enumerate(sorted(nodes))}
    node_dic_rev = dict((y, x) for x, y in node_dic.items())
    nodes =[(v, k) for k, v in node_dic.items()]
    df_ID = df.copy()
    for i in range(len(nodes)):
        df_ID = df_ID.replace(nodes[i][1], nodes[i][0])

    A = np.array(df_ID.values.tolist())
    i, j, weight = A[:, 0], A[:, 1], A[:, 2]
    # find the dimension of the square matrix
    dim = max(len(set(i)), len(set(j)))

    B = sps.lil_matrix((dim, dim))
    for i, j, w in zip(i, j, weight):
        B[i, j] = w

    B = B.todense()
    B_temp1 = B + B.T
    
    n = B_temp1.shape[0]
    di = np.diag_indices(n)
    B_temp1[di] = 1
        
    B_new = pd.DataFrame(B_temp1)
    B_new = B_new.rename(index=node_dic_rev).rename(columns=node_dic_rev)
    
    if input_data == 'target':
        B_new['from'] = 'hsa' + B_new['from'].astype(str)
        B_new['to'] = 'hsa' + B_new['to'].astype(str)
        
        # It will used in comparing different methods as an input data
        B_new.to_csv('all_PPI.txt', sep='\t')
    else:
        B_new.to_csv('all_DDI.txt', sep='\t')
        
    return B_new


def ROC_curve(X, model, y):
    """
    This function plots the ROC curve and shows the best threshold 
    
    Args: 
        X: A DataFram containing features
        model: Trained machine learning model
        y: A list of the labels
    
    Returns: The value of best threshold
    """
    # predict probabilities
    pred_test_proba = model.predict_proba(X)
    # keep probabilities for the positive outcome only
    pred_test_proba = pred_test_proba[:, 1]
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y, pred_test_proba)
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('\n Best threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    best_threshold = thresholds[ix]
    # plot the roc curve for the model
    pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
    pyplot.plot(fpr, tpr, marker='.', label='XGBoost')
    pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best threshold')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()

    return pyplot, best_threshold



def adjacency_matrix():
    """
    This function creates adjacency matrix for DTI(for finding negative interactions)
    
    Returns: DTI adjacency matrix
    """
    DTI_matri_file = ['e_admat_dgc','ic_admat_dgc', 'gpcr_admat_dgc','nr_admat_dgc'] 
    all_dict = {}
    for i in DTI_matri_file:
        base_file = i.split('.')[0]
        all_dict[base_file] = pd.read_csv('./'+ f'{base_file}.csv', index_col=0)    

    df_temp1 = pd.concat([all_dict['ic_admat_dgc'], all_dict['e_admat_dgc']], sort=False)
    df_temp2 = pd.concat([df_temp1, all_dict['gpcr_admat_dgc']], sort=False)
    all_interaction_matri = pd.concat([df_temp2, all_dict['nr_admat_dgc']], sort=False).fillna(0)
    all_interaction_matri.to_csv('golden_standard_dataset/all_interaction.csv')

    return all_interaction_matri


def get_data(df):
    """
    This function separates features (as x) and labels (as y)
    
    Args: A Datafram containing features and labels
        
    Returns: 
        x: Features
        y: labels
    """
    df['label'] = df['weight']
    df.drop(columns = ['weight'], inplace=True)
    print(df['label'].value_counts())
    
    X = df.drop(columns=['label'])
    y = pd.DataFrame(df['label'])
    
    return X, y.values.ravel()



def get_sample_weight(y_train):
    """
    This function calculates the weigh_ratio for training imbalance dataset
    
    Args: Labels of train-set
    
    Returns: The weight ratio
    """
    weight_ratio = float(len(y_train[y_train == 0]))/float(len(y_train[y_train == 1]))
    
    return weight_ratio


def concating(df, embedding):
    """
    This function concats drugs and targets vectors
    
    Args: 
        df: A DataFrame of drug-target interactions
        embedding:A DataFrame of embedded vectors for each drug and target
        
    Returns: A DataFrame of drug-target interaction vectors
    """
    dataset = []
    for idx, row in df.iterrows():
        from_vector = embedding.loc[row['from']]
        to_vector = embedding.loc[row['to']]
        features = from_vector.append(to_vector).reset_index(drop=True)
        features = features.append(row)
        dataset.append(features)

    df_final = pd.DataFrame(dataset)
    df_final.drop(columns=['from', 'to'], inplace=True)
    return df_final


# ### 0 - downloading data

#Run it when you running DT2Vec for the first time
downloading_data()


# ### 1- Reading DTI


df_interaction = reading_interarction()
print(f'\n number of interactions: {df_interaction.shape[0]}')
print(df_interaction['weight'].value_counts())
df_interaction.head()

numGene_DTI = len(set(df_interaction['to']))
print(f'number of unige gene in DTI: {numGene_DTI}')
numDrug_DTI = len(set(df_interaction['from']))
print(f'number of unige drug in DTI: {numDrug_DTI}')


# ### 2- Reading PPS


PPI_file = ["e_simmat_dg","ic_simmat_dg","gpcr_simmat_dg","nr_simmat_dg"]

dic_protein_type = {}
# add type of proteins
for file in PPI_file:
    df_tmp = pd.read_csv(f'/{file}.csv').set_index(['Unnamed: 0'])
    df_tmp['type']= file
    dic_tmp = df_tmp[['type']].to_dict()['type']
    dic_protein_type.update(dic_tmp)

dic_protein_type

df_PPI = reading_similarity(PPI_file)
df_PPI_all = mergeing_sim_matrix(df_PPI)
df_PPI.head()


# ###  3- Reading DDS 
DDI_file = ["e_simmat_dc","ic_simmat_dc","gpcr_simmat_dc","nr_simmat_dc"]
df_DDI = reading_similarity(DDI_file)
df_DDI_all = mergeing_sim_matrix(df_DDI)
df_DDI.head()


# ### 4- Creating DTI adjacency matrix

all_interaction_matri = adjacency_matrix()

# ### 5- Changing Ids from str to int for Nod2vec

# all simalrities and interactions as one list (a format needed for N2V)
df_all_sim_inter = (df_DDI.append(df_PPI)).append(df_interaction)
df_all_sim_inter.head()

# number of unique gene and drug 989, 791 repectively
list_of_str = set(df_all_sim_inter['from'].append(df_all_sim_inter['to']))
len(list_of_str) 

# Creat dic of drug and genes to number (in N2V the name of node should save as number)
DrugID2NumericID = {k: v+1 for v, k in enumerate(sorted(list_of_str))}
NumericID2DrugID = dict((y,x) for x,y in DrugID2NumericID.items())


# all positive and negative interactions in matrix
all_interaction_matri.index = pd.Index(map(lambda x : x.lstrip('hsa'), all_interaction_matri.index))
# mapping genes to number
all_interaction_matri_newLabel = all_interaction_matri.rename(index=DrugID2NumericID) 
# mapping drugs to number
all_interaction_matri_newLabel = all_interaction_matri_newLabel.rename(columns=DrugID2NumericID) 
all_interaction_matri_newLabel.head()

# DDS (n = 791  --> n*n-1/2 -->791*790/2= 3312445 )
df_DDI_newID = df_DDI.replace(DrugID2NumericID)
print(f'\n all DDS: {df_DDI_newID.shape[0]}')
df_DDI_newID.head()


edgeList_dds = df_DDI_newID.values.tolist()
G = networkx.Graph()

for i in range(len(edgeList_dds)):
    G.add_edge(edgeList_dds[i][0], edgeList_dds[i][1], weight=edgeList_dds[i][2])
    
A = networkx.adjacency_matrix(G)
DDS_adj = A.copy()

louvain = Louvain()
labels = louvain.fit_transform(DDS_adj)

DDS_cluster_label= pd.DataFrame({'drug':list(G.nodes()), 'label':labels})
DDS_cluster_label['label']= DDS_cluster_label['label'].astype(str)
labels_unique, counts = np.unique(labels, return_counts=True)
print(labels_unique, counts)

weights = nx.get_edge_attributes(G,'weight').values()
print(len(weights))


louvain_hierarchy = LouvainHierarchy()
dendrogram = louvain_hierarchy.fit_transform(A)
dasgupta_score(A, dendrogram)


#tree_sampling_divergence(A, dendrogram)

drug2cluster = dict(zip(DDS_cluster_label.drug, DDS_cluster_label.label))


color_dict_drug = {k: v for k, v in enumerate( ['mediumvioletred','mediumblue','gold','green','violet','mediumturquoise','mediumvioletred','darkgoldenrod','pink','dimgray'])}
color_dict_drug= {str(key): value for key, value in color_dict_drug.items()}

modularity2 = community.modularity(drug2cluster, G, weight='weight')
print("The modularity Q based on networkx is {}".format(modularity2))


pos = nx.spring_layout(G, scale=2)

plt.figure(figsize=(8,6))
nx.draw(G, pos, node_color = [color_dict_drug[v] for v in drug2cluster.values()], edge_color=weights, node_size=[30]*len(G.nodes()))
plt.savefig('drug_graph_GSD.png')


# PPS (n = 989 --> 989*988/2 = 488566)
df_PPI_newID = df_PPI.replace(DrugID2NumericID)
print(f'all PPS: {df_PPI_newID.shape[0]}')
df_PPI_newID.head()

# clusters PPS network
edgeList_pps= df_PPI_newID.values.tolist()
G = networkx.Graph()

for i in range(len(edgeList_pps)):
    G.add_edge(edgeList_pps[i][0], edgeList_pps[i][1], weight=edgeList_pps[i][2])
    
A = networkx.adjacency_matrix(G).A
PPS_adj = A.copy()

louvain = Louvain()
labels = louvain.fit_transform(PPS_adj)

labels_unique, counts = np.unique(labels, return_counts=True)

PPS_cluster_label= pd.DataFrame({'target':list(G.nodes()), 'label':labels})
PPS_cluster_label['label']= PPS_cluster_label['label'].astype(str)
print(labels_unique, counts)


weights = nx.get_edge_attributes(G,'weight').values()
print(len(weights))

louvain_hierarchy = LouvainHierarchy()
dendrogram = louvain_hierarchy.fit_transform(A)
dasgupta_score(A, dendrogram)


target2cluster = dict(zip(PPS_cluster_label.target, PPS_cluster_label.label))


modularity2 = community.modularity(target2cluster, G, weight='weight')
print("The modularity Q based on networkx is {}".format(modularity2))



weight = [element * 1000 for element in weights]
weight = [500 if i>=500 else i for i in weight]


color_dict = {k: v for k, v in enumerate(['#58ACFA','#FF1493', 'yellow','orange', '#00CED1','#5F9EA0','#006400','#96bf65','#fcc808','#7b2b48',
 '#e96957','#e06000','#173679','#d2dd49','#684a6b','#096eb2','#ce482a', 'red', 'lime', 'lightslategray',
                                      'olive', 'rosybrown', 'sienna', 'darkmagenta','midnightblue','maroon',
                                      'lightcoral','gold','sandybrown','tomato','lawngreen','lightgreen','darkorchid',
                                      'lightskyblue','darkgreen'])}
color_dict= {str(key): value for key, value in color_dict.items()}


#partition = community_louvain.best_partition(G)
pos = nx.spring_layout(G, k=0.15)

plt.figure(figsize=(10,6))
nx.draw(G, pos, node_color=[color_dict[v] for v in target2cluster.values()], edge_color=weight, node_size=[20]*len(G.nodes()))
plt.savefig('target_graph_GSD.png')


# DTI
df_interaction_newID = df_interaction.replace(DrugID2NumericID)
print(f'all DTI: {df_interaction_newID.shape[0]}')
df_interaction_newID.head()


# ### 6- Developing ML model


def embedding(DDS, PPS_seq, mapping=True):
    """
    This function maps drug-drug similarities and protein-protein similarities to vectors
    
    Args: 
    PPS_seq: A DataFrame of protein-protein smilarities 
    DDS: A DataFrame of drug-drug smilarities 
    mapping: Binary (False/True), read from the saved files (mapped before)
    
    Returns: A dataframe vectors of drugs and proteins 
    """
    
    PPS_seq = PPS_seq[PPS_seq['weight']!=0]
    DDS = DDS[DDS['weight']!=0]
    
    DDS.to_csv('edglist_drugs_goldenData.edgelist', sep=' ', index=False, header=False)
    PPS_seq.to_csv('edglist_proteins_goldenData.edgelist', sep=' ', index=False, header=False)
    
    if mapping:
        # nod2vec (for drug)
        os.system(f'PYTHONHASHSEED=10 python2 node2vec/src/main.py --workers 8 --input edglist_drugs_goldenData.edgelist --output dim100_drugs_goldenData.emb --weighted --dimensions 100')
         # nod2vec (for proteins)
        os.system(f'PYTHONHASHSEED=10 python2 node2vec/src/main.py --workers 8 --input edglist_proteins_goldenData.edgelist --output dim100_proteins_goldenData.emb --weighted --dimensions 100')

    embeddings_seq_drug = pd.read_csv('dim100_drugs_goldenData.emb', sep=' ', skiprows=[0], header=None, index_col=0)  
    embeddings_seq_protein = pd.read_csv('/dim100_proteins_goldenData.emb', sep=' ', skiprows=[0], header=None, index_col=0)  

    embeddings_seq = embeddings_seq_drug.append(embeddings_seq_protein)
    embeddings_seq.index.name = 'ID'
    
    return embeddings_seq, embeddings_seq_drug, embeddings_seq_protein


embeddings_seq, embeddings_drug, embeddings_protein = embedding(df_PPI_newID, df_DDI_newID, mapping=False) 


def plot_pca_2d_with_clusters(df_total, DDS, PPS_seq, pca_t):
    
    """
    This function plot PCA of drugs and proteins vectors 
    
    Args: 
    PPS_seq: A DataFrame of protein-protein smilarities 
    DDS: A DataFrame of drug-drug smilarities 
    df_total: A dataframe vectors of drugs and proteins 
    
    """
    
    drug_name = list(set(list(DDS['from'])+list(DDS['to'])))
    protein_name = list(set(list(PPS_seq['from'])+list(PPS_seq['to'])))

    drug_vec = df_total.loc[drug_name]
    drug_vec['type'] = 'Drug'
            
    protein_vec = df_total.loc[protein_name]   
    
    if pca_t=='Drug':
        df = drug_vec.copy()
    elif pca_t== 'Target':
        protein_vec['type'] = protein_vec.index
        protein_vec['type'] = protein_vec['type'].replace(NumericID2DrugID)
        protein_vec['type'] = 'hsa' + protein_vec['type'].astype(str)
        protein_vec['type'] = protein_vec['type'].astype(str).replace(dic_protein_type).replace({"e_simmat_dg":'Enzymes',
                                                                                                 "ic_simmat_dg":'ion channels',
                                                                                                 "gpcr_simmat_dg":'G-protein-coupled receptors',
                                                                                                 "nr_simmat_dg": 'Nuclear receptors'})
        df = protein_vec.copy()
    
    pca = PCA(n_components=2)
    if pca_t=='Drug':
        pca = TSNE(n_components=2, random_state=42, perplexity=50, n_iter= 400 )
    pca_result = pca.fit_transform(df.drop(columns=['type']).values)
  
    
    df['TSNE-1'] = pca_result[:,0]
    df['TSNE-2'] = pca_result[:,1]
        
    chart = ggplot(df, aes(x='TSNE-1', y='TSNE-2',  color='factor(type)') )         + geom_point(size=120, alpha=0.8)         + scale_color_manual(values = ['#58ACFA','#FF1493', '#00BFFF','#00CED1','#5F9EA0','#006400','#006400','#96bf65','#fcc808','#7b2b48',
 '#e96957','#e06000','#173679','#e8a1a2','#d2dd49','#684a6b','#096eb2','#bde1e9','#d2dd49','#ce482a'])\
        #+ geom_text(aes(label='factor(name)'), size=6, color='black')
    
    
    chart.save(f'./{pca_t}_pca_GSD.png', width=12, height=8)  
   
    return chart


def plot_pca_2d_with_Louvain_clusters(embeddings_seq, drug2cluster, target2cluster, type_data= 'drug'):
    
    """
    This function plot PCA of drugs and proteins vectors 
    
    Args: 
    PPS_seq: A DataFrame of protein-protein smilarities 
    DDS: A DataFrame of drug-drug smilarities 
    df_total: A dataframe vectors of drugs and proteins 
    
    """
    df = embeddings_seq.copy()
    df['cluster'] = df.index
   # df['cluster'] = df['cluster'].astype(str).replace(NumericID2DrugID)
    
    if type_data== 'drug':
        df = df[df['cluster'].isin(list(drug2cluster.keys()))]
        df['cluster'] = df['cluster'].replace(drug2cluster)
    else:
        df = df[df['cluster'].isin(list(target2cluster.keys()))]
        df['cluster'] = df['cluster'].replace(target2cluster)

    
    pca = PCA(n_components=2)
    if type_data== 'drug':
        pca = TSNE(n_components=2, random_state=42, perplexity=50, n_iter= 400 )
    pca_result = pca.fit_transform(df.drop(columns=['cluster']).values)
  
    if type_data== 'target':
        color_ids = ['#58ACFA','#FF1493', 'yellow','orange', '#00CED1','#5F9EA0','#006400','#96bf65','#fcc808','#7b2b48',
 '#e96957','#e06000','#173679','#d2dd49','#684a6b','#096eb2','#ce482a', 'red', 'lime', 'lightslategray',
                                      'olive', 'rosybrown', 'sienna', 'darkmagenta','midnightblue','maroon',
                                      'lightcoral','gold','sandybrown','tomato','lawngreen','lightgreen','darkorchid',
                                      'lightskyblue','darkgreen']
    if type_data== 'drug':
        color_ids = ['mediumvioletred','mediumblue','gold','green','violet','mediumturquoise','mediumvioletred','darkgoldenrod','pink','dimgray']

    
    
    df['PCA-1'] = pca_result[:,0]
    df['PCA-2'] = pca_result[:,1]
        
    chart = ggplot(df, aes(x='PCA-1', y='PCA-2',  color='factor(cluster)') )         + geom_point(size=120, alpha=0.8)         + scale_color_manual(values = color_ids)    
    
    chart.save(f'./{type_data}_GSD_pca_cluster.png', width=12, height=8)  
        
   
    return chart


embeddings_seq, embeddings_seq_drug, embeddings_seq_protein = embedding(df_PPI_newID, df_DDI_newID, mapping=False)

plot_pca_2d_with_Louvain_clusters(embeddings_seq, drug2cluster, target2cluster, type_data= 'target')

plot_pca_2d_with_clusters(embeddings_seq, df_DDI_newID, df_PPI_newID, pca_t='Target')

plot_pca_2d_with_Louvain_clusters(embeddings_seq, drug2cluster, target2cluster, type_data= 'drug')

plot_pca_2d_with_clusters(embeddings_seq, df_DDI_newID, df_PPI_newID, pca_t='Drug')



embeddings_seq, embeddings_seq_drug, embeddings_seq_protein = embedding(df_DDI_newID, df_PPI_newID, mapping=False)


def run_all(df_interaction_newID, all_interaction_matri_newLabel, embeddings , n=10):
    """
    This function ceates the machine learning model (train and test the model)
    
    Args: 
        df_interaction_newID: A DataFrame of drug-target interactions
        df_PPI_newID: A DataFrame of protein-protein smilarities 
        df_DDI_newID: A DataFrame of drug-drug smilarities 
        all_interaction_matri_newLabel: DataFrame of all positive and negative interactions in matrix
        n: Number of folds
        
    Returns: 
        all_train_results: Traning results for n folds
        all_test_results: Testing results for n folds
    """
    ## spilit positive DTIs to external and CV
    DTI_external_test = df_interaction_newID.sample(frac = 0.1)
    DTI_new = df_interaction_newID.drop(DTI_external_test.index)

    num_crossVal = 0 
    all_train_results, all_test_results = [], []
    kf = KFold(n_splits=n, random_state=42, shuffle=True)
    kf.get_n_splits(DTI_new)
    for train_index, test_index in kf.split(DTI_new):
        df_interaction_newID_train, df_interaction_newID_test = DTI_new.iloc[train_index], DTI_new.iloc[test_index]
        num_crossVal = num_crossVal + 1 
        print(f'\n\n\n  KFold: {num_crossVal}')
        print(f'# all DTI: {DTI_new.shape[0]}')
        print(f'# DTI train-set: {df_interaction_newID_train.shape[0]}')
        print(f'# DTI test-set: {df_interaction_newID_test.shape[0]}')

        # Embeddings: embedding file for all drugs and genes based on DTI, DDS, PPS
        # read from saved file

        all_interaction_edgeList = all_interaction_matri_newLabel.stack().reset_index().rename(columns={'level_0':'to','level_1':'from',0:'weight'})
        all_negative_edgeList = all_interaction_edgeList[all_interaction_edgeList['weight']==0]

        ## spilit negative DTIs to external and CV
        DTI_external_negative = all_negative_edgeList.sample(frac = 0.1)
        DTI_new_negative = all_negative_edgeList.drop(DTI_external_negative.index)

        # selecting n negative sample where n is number of train and test
        negative_edgeList_sampel = DTI_new_negative.sample(n = DTI_new.shape[0])

        # splite them to train and test 
        p_n = (100 - n)/100
        msk = np.random.rand(len(negative_edgeList_sampel)) < p_n
        df_negative_train = negative_edgeList_sampel[msk]
        df_negative_test = negative_edgeList_sampel[~msk]

        print(f'# negative interaction: {negative_edgeList_sampel.shape[0]}')
        print(f'# negative train: {df_negative_train.shape[0]}')
        print(f'# negative test: {df_negative_test.shape[0]}')

        df_train_edj = df_negative_train.append(df_interaction_newID_train)
        df_test_edj = df_negative_test.append(df_interaction_newID_test)
        df_train = concating(df_train_edj, embeddings)
        df_test = concating(df_test_edj, embeddings)

        print('=============================================================')
        print(f'# train: {df_train.shape[0]}')
        print(f'# test: {df_test.shape[0]}')
        print('=============================================================')
        print(f'training...')

        X, y = get_data(df_train.rename(columns={"label": "weight"}))
        print(f'X_train.shape: {X.shape}')

        if SCALE_POS_RATIO == True:
            weight_ratio = get_sample_weight(y)
            model = XGBClassifier(scale_pos_weight=weight_ratio)
        else:
            model = XGBClassifier(learning_rate= 0.4, max_depth= 4, min_child_weight=2) #defult gamma= 0, min_child_weight=1
        model.fit(X, y)

        pred_train = model.predict(X)
        train_results = calculate_metrics(y, pred_train)
        train_results['Kfold'] = num_crossVal
        train_results['model'] = model
        all_train_results.append(train_results)

        print('\n')
        print(f'testing...')

        X, y = get_data(df_test.rename(columns={"label": "weight"}))
        print(f'X_test.shape: {X.shape}')

        pred_test = model.predict(X)
        test_results = calculate_metrics(y, pred_test)
        test_results['model'] = model
        all_test_results.append(test_results)
        best_model = pd.DataFrame(all_test_results).sort_values(by=['average_precision_score'], ascending=False)['model'][0]

        del X
        del y

        ##########################################
    
        # external testset
        DTI_new_negative_sample = DTI_external_negative.sample(n = DTI_external_test.shape[0])
        df_all_external = DTI_external_test.append(DTI_new_negative_sample)


        df_all_external = concating(df_all_external, embeddings)
        X_test_ex, y_test_ex = get_data(df_all_external.rename(columns={"label": "weight"}))
        pred_test_ex = best_model.predict(X_test_ex)
        test_results_ex = calculate_metrics(y_test_ex, pred_test_ex)
        
    return all_train_results, all_test_results, test_results_ex


resuts_all_runs_list = []
resuts_external_list = []

for i in range(0,5):
    all_train_results, all_test_results, test_results_ex = run_all(df_interaction_newID, all_interaction_matri_newLabel, embeddings_seq , n=10)
    resuts_all_runs_list.append(all_test_results) 
    resuts_external_list.append(test_results_ex)



all_result = pd.DataFrame({'accuracy':[0], 'precision':[0], 'recall':[0],
                           'f1_score':[0],'f2_score':[0], 'average_precision_score':[0],
                           'ROC':[0]})
for i in range(0,5):
    all_result = all_result.append(pd.DataFrame(resuts_all_runs_list[i]))


pd.DataFrame(resuts_external_list)




