import random
from collections import defaultdict
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import itertools
from sklearn.manifold import TSNE
from sknetwork.data import karate_club, painters, movie_actor
from sknetwork.clustering import Louvain, BiLouvain, modularity, bimodularity
from sknetwork.linalg import normalize
from sknetwork.utils import bipartite2undirected, membership_matrix
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph
from sknetwork.hierarchy import LouvainHierarchy, BiLouvainHierarchy
from sknetwork.hierarchy import dasgupta_score, dasgupta_cost
from sklearn.cluster import MiniBatchKMeans, KMeans, FeatureAgglomeration, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, AgglomerativeClustering, DBSCAN, Birch
import graphviz
from xgboost import plot_tree
import csv
import os
import json
import gensim
import numpy as np
from pprint import pprint
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from ggplot import *
from scipy.stats import chi2_contingency
from matplotlib import rc
import scipy
import pickle
import shap
import requests
import json
import re
from os.path import isfile, join
from os import listdir
from bs4 import BeautifulSoup
from collections import Counter
from ggplot import *
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import networkx as nx
import networkx
import community
from yellowbrick.cluster import KElbowVisualizer
from community import community_louvain
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    auc,
    average_precision_score,
    fbeta_score)

# for an imbalance data change it to True
np.random.seed(42)
SCALE_POS_RATIO = False
HIGHLY_POS = 5.5 #positive threshold
HIGHLY_NEG = 0
MAIN_DIR = '.'

# ## Functions

def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


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
    
    DDS.to_csv('edglist_drugs.edgelist', sep=' ', index=False, header=False)
    PPS_seq.to_csv('edglist_proteins.edgelist', sep=' ', index=False, header=False)
    
    if mapping:
        # nod2vec (for drug)
        os.system(f'PYTHONHASHSEED=10 python2 node2vec/src/main.py --workers 8 --input edglist_drugs.edgelist --output dim100_drugs.emb --weighted --dimensions 100')
         # nod2vec (for proteins)
        os.system(f'PYTHONHASHSEED=10 python2 node2vec/src/main.py --workers 8 --input edglist_proteins.edgelist --output dim100_proteins.emb --weighted --dimensions 100')

    embeddings_seq_drug = pd.read_csv('dim100_drugs.emb', sep=' ', skiprows=[0], header=None, index_col=0)  
    embeddings_seq_protein = pd.read_csv('dim100_proteins.emb', sep=' ', skiprows=[0], header=None, index_col=0)  

    embeddings_seq = embeddings_seq_drug.append(embeddings_seq_protein)
    embeddings_seq.index.name = 'ID'
    
    return embeddings_seq, embeddings_seq_drug, embeddings_seq_protein


def calculate_metrics(y, y_pred):
    """
    This function calculates machine learning metrics
    
    Args: Real lables and predicted labels

    Returns: A dictionary containing all the results
    """
    
    result = {
        'accuracy': metrics.accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='binary'),
        'recall': recall_score(y, y_pred, average='binary'),
        'f1_score': f1_score(y, y_pred, average='binary'),
        'average_precision_score': average_precision_score(y, y_pred),
        'f2_score':fbeta_score(y, y_pred, beta=2)
    }
    
    try:
        result['ROC'] = metrics.roc_auc_score(y, y_pred)
    except:
        pass

    return result


def calculate_metrics_multiclass(y, y_pred):
    """
    This function calculates machine learning metrics
    
    Args: Real lables and predicted labels

    Returns: A dictionary containing all the results
    """
    
    result = {
        'accuracy': metrics.accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='samples'),
        'recall': recall_score(y, y_pred, average='samples'),
        'f1_score': f1_score(y, y_pred, average='samples'),
    }
    

    return result


def concating(df, embedding):
    """
    This function concats drugs and targets vectors
    
    Args: 
        df: A DataFrame of drug-target interactions
        embedding: A DataFrame of embedded vectors for each drug and target
        
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


def normalized_data(df):
    df_n = df.copy()
    mean_weight = df_n['weight'].mean()
    std_weight = df_n['weight'].std()
    df_n['weight_new'] = (df_n['weight']-mean_weight)/std_weight
    df_n = df_n.drop(columns=["weight"])
    df_n = df_n.rename(columns={'weight_new':'weight'})
    return df_n


def scale_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = df[['weight']]
    scaler.fit(data)
    data = scaler.transform(data)
    df['weight'] = data
    return df



with open('all_nod2vec_new.pkl', 'rb') as f:
    all_nod2vec_new = pickle.load(f)
all_nod2vec_new


# ### 1- Reading input data 

# DTI
DTI= pd.read_csv(f'{MAIN_DIR}edgelis/DTI.csv', index_col=0)
num_drug = len(set(list(DTI['from'])))
num_protein = len(set(list(DTI['to'])))

print(f'Number of drug: {num_drug} and number of protein: {num_protein}')
DTI.head()

# PPS
PPS_seq = pd.read_csv(f'{MAIN_DIR}edgelis/PPS_seq.csv', index_col=0)
#PPS_seq = normalized_data(PPS_seq)
#PPS_seq = scale_data(PPS_seq)
PPS_seq.head()


with open('protein2num.pkl', 'rb') as f:
    protein2num = pickle.load(f)
num2protein = dict((str(int(y)),x) for x,y in protein2num.items())


PPS_seq['weight'].mean()

PPS_seq_new = PPS_seq[PPS_seq['weight']>=0.001]
PPS_seq_new = PPS_seq.copy()
PPS_seq_new['weight'].mean()

# clusters PPS network
edgeList_pps = PPS_seq_new.values.tolist()
G = networkx.Graph()
weights = []

for i in range(len(edgeList_pps)):
    G.add_edge(edgeList_pps[i][0], edgeList_pps[i][1], weight=edgeList_pps[i][2])
    weights.append(edgeList_pps[i][2])
    
A = networkx.adjacency_matrix(G).A
PPS_adj = A.copy()

louvain = Louvain()
labels = louvain.fit_transform(PPS_adj)

labels_unique, counts = np.unique(labels, return_counts=True)

PPS_cluster_label= pd.DataFrame({'target':list(G.nodes()), 'label':labels})
PPS_cluster_label['label']= PPS_cluster_label['label'].astype(str)
print(labels_unique, counts)

target2cluster = dict(zip(PPS_cluster_label.target, PPS_cluster_label.label))

modularity2 = community.modularity(target2cluster, G, weight='weight')
print("The modularity Q based on networkx is {}".format(modularity2))


color_dict = {k: v for k, v in enumerate(['#58ACFA','#FF1493', 'yellow','orange', '#00CED1','#5F9EA0','#006400','#96bf65','#fcc808','#7b2b48',
 '#e96957','#e06000','#173679','#d2dd49','#684a6b','#096eb2','#ce482a', 'red', 'lime', 'lightslategray',
                                      'olive', 'rosybrown', 'sienna', 'darkmagenta','midnightblue','maroon',
                                      'lightcoral','gold','sandybrown','tomato','lawngreen','lightgreen','darkorchid',
                                      'lightskyblue','darkgreen'])}
color_dict= {str(key): value for key, value in color_dict.items()}

with open('color_dict_target.pkl', 'wb') as fp:
    pickle.dump(color_dict, fp)

partition = community_louvain.best_partition(G)
pos = community_layout(G, target2cluster)
plt.figure(figsize=(9,5))
nx.draw(G, pos, node_color=[color_dict[v] for v in target2cluster.values()], edge_color=weights, node_size=[15]*len(G.nodes()))

weight = [element * 1000 for element in weights]
weight = [40 if i>=40 else i for i in weight]

#partition = community_louvain.best_partition(G)
pos = nx.spring_layout(G, scale=2)

plt.figure(figsize=(10,6))
nx.draw(G, pos, node_color=[color_dict[v] for v in target2cluster.values()], edge_color=weight, node_size=[20]*len(G.nodes()))
plt.savefig('target_graph.png')

# DDS
DDS = pd.read_csv(f'{MAIN_DIR}DDS_known_ChEMBLid_T{num_drug}.csv')
DDS.head()

DDS_new = DDS.copy()

edgeList_dds = DDS_new.values.tolist()
G = networkx.Graph()
weights = []
for i in range(len(edgeList_dds)):
    G.add_edge(edgeList_dds[i][0], edgeList_dds[i][1], weight=edgeList_dds[i][2])
    weights.append(edgeList_dds[i][2])
    
A = networkx.adjacency_matrix(G)
DDS_adj = A.copy()

louvain = Louvain()
labels = louvain.fit_transform(DDS_adj)

DDS_cluster_label= pd.DataFrame({'drug':list(G.nodes()), 'label':labels})
DDS_cluster_label['label']= DDS_cluster_label['label'].astype(str)
labels_unique, counts = np.unique(labels, return_counts=True)
print(labels_unique, counts)

drug2cluster = dict(zip(DDS_cluster_label.drug, DDS_cluster_label.label))
color_dict_drug = {k: v for k, v in enumerate( ['mediumvioletred','mediumblue','gold','green','violet','mediumturquoise','mediumvioletred','darkgoldenrod','pink','dimgray'])}
color_dict_drug= {str(key): value for key, value in color_dict_drug.items()}

with open('color_dict_drug.pkl', 'wb') as fp:
    pickle.dump(color_dict_drug, fp)

modularity2 = community.modularity(drug2cluster, G, weight='weight')
print("The modularity Q based on networkx is {}".format(modularity2))

partition = community_louvain.best_partition(G)
pos = community_layout(G, drug2cluster)

plt.figure(figsize=(9,5))
nx.draw(G, pos, node_color=[color_dict_drug[v] for v in drug2cluster.values()], edge_color=weights, node_size=[15]*len(G.nodes()))

pos = nx.spring_layout(G, scale=2)

plt.figure(figsize=(8,6))
nx.draw(G, pos, node_color=[color_dict_drug[v] for v in drug2cluster.values()], edge_color=weights, node_size=[30]*len(G.nodes()))
plt.savefig('drug_graph.png')

DDS_boxplot = DDS[['weight']]
DDS_boxplot['type'] = 'DDS'

PPS_boxplot = PPS_seq[['weight']]
PPS_boxplot['type'] = 'PPS'

df_boxplot = DDS_boxplot.append(PPS_boxplot)

boxplot = df_boxplot.boxplot(by='type',fontsize=15, figsize=(6,7))

vals, names, xs = [],[],[]
for i, col in enumerate(DTI[["weight"]]):
    vals.append(DTI[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, DTI[col].values.shape[0]))

plt.boxplot(vals, labels=["Drug-target interaction"])
palette = ['y']
for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.2, color='#72BEB7')
    
    plt.ylabel("pChEMBL Value", fontweight='normal', fontsize=14)
    sns.despine(bottom=True) # removes right and top axis lines
    plt.axhline(y=5.50, color='b', linestyle='--', linewidth=1, label='Activity thresholds')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Negative interaction')

    plt.legend(bbox_to_anchor=(0.31, 1.06), loc=2, borderaxespad=0, framealpha=1, facecolor ='white', frameon=True)
    plt.legend(loc='upper right')
    plt.savefig('scatter.png')
plt.show()


# ### 3- Creating test/train intearctions in ML
# define postive/negative interactions (positive--> higher than threshold)
df_high_pos = DTI[DTI['weight']>= HIGHLY_POS]
df_high_pos['L'] = 1

df_high_neg = DTI[DTI['weight']< HIGHLY_NEG]
df_high_neg['L'] = 0

# neutral DTI (positive but less than threshold)
df_neutral= DTI[(DTI['weight']<HIGHLY_POS) & (DTI['weight']>HIGHLY_NEG)]
df_neutral['L'] = -1

print(f'Total number of DTI for creating graph: {len(DTI)}')
print(f'Num of gene/protein: {num_protein}')
print(f'Num of drug: {num_drug}')
print(f'Num of +/- DTI for developing ML-model: {len(df_high_pos)+len(df_high_neg)}')
print(f'Num of highly + interactions >= {HIGHLY_POS}: {len(df_high_pos)}')
print(f'Num of - interactions = {HIGHLY_NEG}: {len(df_high_neg)}')
print(f'Num of positive interactions: {len(df_neutral)}')


# dataframe with real value of interactions as "weight" and binary value of interactions as "label"
DTI_for_ML = df_high_pos.append(df_high_neg)
print(DTI_for_ML['L'].value_counts())
DTI_for_ML.head()

# without normalization and scaling
# Normalizing based on DDS, PPS, DTI_neutral
# positive and negative DTI for machine learning are binaray and do not need scaling and normalization

DTI_boxplot = DTI[(DTI["weight"]<HIGHLY_POS) & (DTI["weight"]>HIGHLY_NEG)][['weight']]
DTI_boxplot['type'] = 'DTI'

DDS_boxplot = DDS[['weight']]
DDS_boxplot['type'] = 'DDS'

PPS_boxplot = PPS_seq[['weight']]
PPS_boxplot['type'] = 'PPS'


# #  RUN ALL

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
    
    if type_data== 'drug':
        df = df[df['cluster'].isin(list(drug2cluster.keys()))]
        df['cluster'] = df['cluster'].replace(drug2cluster)
    else:
        df = df[df['cluster'].isin(list(target2cluster.keys()))]
        df['cluster'] = df['cluster'].replace(target2cluster)

    
    pca = PCA(n_components=2)
    #pca = TSNE(n_components=2, random_state=42, perplexity=50, n_iter= 400 )
    pca_result = pca.fit_transform(df.drop(columns=['cluster']).values)
  
    
    df['PCA-1'] = pca_result[:,0]
    df['PCA-2'] = pca_result[:,1]
    
    if type_data== 'target':
        color_ids = ['#58ACFA','#FF1493', 'yellow','orange', '#00CED1','#5F9EA0','#006400','#96bf65','#fcc808','#7b2b48',
 '#e96957','#e06000','#173679','#d2dd49','#684a6b','#096eb2','#ce482a', 'red', 'lime', 'lightslategray',
                                      'olive', 'rosybrown', 'sienna', 'darkmagenta','midnightblue','maroon',
                                      'lightcoral','gold','sandybrown','tomato','lawngreen','lightgreen','darkorchid',
                                      'lightskyblue','darkgreen']
    if type_data== 'drug':
        color_ids = ['mediumvioletred','mediumblue','gold','green','violet','mediumturquoise','mediumvioletred','darkgoldenrod','pink','dimgray']

    chart = ggplot(df, aes(x='PCA-1', y='PCA-2',  color='factor(cluster)') )         + geom_point(size=120, alpha=0.8)         + scale_color_manual(values= color_ids)    
    
    chart.save(f'{type_data}_pca_cluster.png', width=12, height=8)  
        
   
    return chart, df


def plot_pca_2d_with_clusters(df, DDS, PPS_seq, pca_t='both'):
    
    """
    This function plot PCA of drugs and proteins vectors 
    
    Args: 
    PPS_seq: A DataFrame of protein-protein smilarities 
    DDS: A DataFrame of drug-drug smilarities 
    df_total: A dataframe vectors of drugs and proteins 
    
    """
    df_ChEMBL2DrugBank = pd.read_csv('ChEMBL2DrugBank.csv')
    ChEMBL2DrugBank = pd.Series(df_ChEMBL2DrugBank['DrugBank'].values, index=df_ChEMBL2DrugBank['ChEMBL']).to_dict()
    DrugBank2ChEMBL = dict((y,x) for x,y in ChEMBL2DrugBank.items())

    list_protein = ['unclassified_protein', 'Trascription_factor', 'Transporter', 'Secreted_protein',
                   'other_categories', 'membrance_receptor', 'Ion_Channel', 'Epigenetic_regulation', 
                    'Enzyme_Cytochrome P450', 'Cytosolic_protein', 'Enzyme_Hydrolase', 'Enzyme_Kinase',
                    'Enzyme_Ligase','Enzyme_Lyase','Enzyme_Oxidoreductase','Enzyme_Phosphatase',
                    'Enzyme_Protease', 'Enzyme_rest', 'Enzyme_Transferase']
    

    protein_type = {}
    for p in list_protein:
        df_p = pd.read_csv(f'Protein_class/{p}.csv')
        df_p['type_protein'] = p

        dict_protein = pd.Series(df_p['type_protein'].values, index = df_p['ChEMBL ID']).to_dict()
        dict_protein['CHEMBL2055'] = 'Trascription_factor'
        protein_type.update(dict_protein)

    df_total = df.copy()
    df_total['name'] = df_total.index
    
    drug_name = list(set(list(DDS['from'])+list(DDS['to'])))
    protein_name = list(set(list(PPS_seq['from'])+list(PPS_seq['to'])))

    drug_vec = df_total.loc[drug_name]
    drug_vec['type'] = 'Drug'
    drug_vec['name'] = 'CHEMBL' + drug_vec['name'].astype(str)
    
    with open('protein2num.pkl', 'rb') as f:
            protein2num = pickle.load(f)
    num2protein = dict((str(int(y)),x) for x,y in protein2num.items())
   
            
    protein_vec = df_total.loc[protein_name]
    protein_vec['name'] = protein_vec['name'].astype(str).replace(num2protein)
    protein_vec['type']= protein_vec['name'].replace(protein_type)
    
    
    if pca_t=='drug':
        df = drug_vec.copy()
    elif pca_t== 'target':
        df = protein_vec.copy()
    
    pca = PCA(n_components=2)
   # pca = TSNE(n_components=2, random_state=42)
    pca_result = pca.fit_transform(df.drop(columns=['type', 'name']).values)
  
    
    df['PCA-1'] = pca_result[:,0]
    df['PCA-2'] = pca_result[:,1]
        
    chart = ggplot(df, aes(x='PCA-1', y='PCA-2',  color='factor(type)', label = 'factor(name)') )         + geom_point(size=100, alpha=0.8)         + scale_color_manual(values = ['#58ACFA','#FF1493', '#00BFFF','#00CED1','#5F9EA0','#006400','#96bf65','#fcc808','#7b2b48',
 '#e96957','#e06000','#173679','#e8a1a2','#d2dd49','#684a6b','#096eb2','#bde1e9','#d2dd49','#ce482a'])\
      #  + geom_text(aes(label='factor(name)'), size=10, color='#464546')
    
    
    chart.save(f'{pca_t}_pca.png', width=12, height=8)  
        
   
    return chart, df

def plot_pca(df):
    
    """
    This function plot PCA of drugs and proteins vectors 
    
    Args: 
    PPS_seq: A DataFrame of protein-protein smilarities 
    DDS: A DataFrame of drug-drug smilarities 
    df_total: A dataframe vectors of drugs and proteins 
    
    """
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.drop(columns=['label']).values)
  
    
    df['PCA-1'] = pca_result[:,0]
    df['PCA-2'] = pca_result[:,1]
        
    chart = ggplot(df, aes(x='PCA-1', y='PCA-2',  color='factor(label)') )         + geom_point(size=120, alpha=0.8)         
    
   chart.save(f'{pca_t}_pca_GSD.png', width=12, height=8)  
   
    return chart


def DT2Vec(DTI, embeddings_seq, HIGHLY_POS, HIGHLY_NEG, n=10, multiclass=False):
    """
    This function ceates the machine learning model (train and test the model) 
    
    Args: 
        DTI: A DataFrame of drug-target interactions
        embeddings_seq: A dataframe vectors of drugs and proteins 
        HIGHLY_POS: threshold for positive interactions 
        HIGHLY_NEG: threshold for negative interactions
        n: Number of folds
        
    Returns: 
        all_train_results: Traning results for n folds
        all_test_results: Testing results for n folds
    """
    
    df_high_pos = DTI[DTI["weight"]>= HIGHLY_POS]
    df_high_pos["L"] = 1
    
    df_high_neg = DTI[DTI["weight"]< HIGHLY_NEG]
    df_high_neg["L"] = 0

    df_neutral = DTI[(DTI["weight"]<HIGHLY_POS) & (DTI["weight"]>HIGHLY_NEG)]
    df_neutral["L"] = 2
    
    if multiclass:
         DTI_for_ML = (df_high_pos.append(df_high_neg)).append(df_neutral)
    else:
        DTI_for_ML = df_high_pos.append(df_high_neg)
    
    
    num_crossVal = 0 
    all_train_results, all_test_results = [], []
    kf = KFold(n_splits=n, random_state=42, shuffle=True)
    kf.get_n_splits(DTI_for_ML)
    
    for train_index, test_index in kf.split(DTI_for_ML):
        DTI_for_ML_train, DTI_for_ML_test = DTI_for_ML.iloc[train_index], DTI_for_ML.iloc[test_index]
        num_crossVal = num_crossVal + 1 
        print(f'\n\n\n  KFold: {num_crossVal}')
        print(f'# all DTI: {DTI_for_ML.shape[0]}')
        print(f'# DTI train-set: {DTI_for_ML_train.shape[0]}')
        print(f'# DTI test-set: {DTI_for_ML_test.shape[0]}')

        df_train_seq = concating(DTI_for_ML_train, embeddings_seq).drop(columns=["weight"])
        df_train_seq["weight"] = df_train_seq["L"]
        final_train_seq = df_train_seq.drop(columns=["L"])

        df_test_seq = concating(DTI_for_ML_test , embeddings_seq).drop(columns=["weight"])
        df_test_seq ["weight"] = df_test_seq ["L"]
        final_test_seq = df_test_seq.drop(columns=["L"])
        final_test_seq.head()

        X, y = get_data(final_train_seq)

        if multiclass:
            model = XGBClassifier(objective='multi:softmax', learning_rate=0.9, max_depth=2, min_child_weight=2)
        else:
            model = XGBClassifier(learning_rate= 0.4, max_depth= 4, min_child_weight=2) 

        model.fit(X, y)

        pred_train = model.predict(X)
        
        if multiclass:
            train_results = calculate_metrics_multiclass(y, pred_train)
        else:
            train_results = calculate_metrics(y, pred_train)
            
        train_results['model'] = model
        train_results['embbeding_seq'] = embeddings_seq
        train_results['Kfold'] = num_crossVal
        all_train_results.append(train_results)
        del X
        del y

        print('\n')

        X, y = get_data(final_test_seq)
        pred_test = model.predict(X)
        test_results = calculate_metrics(y, pred_test)
        test_results['model'] = model
        test_results['Kfold'] = num_crossVal
        all_test_results.append(test_results)
        del X
        del y
            
    return pd.DataFrame(all_train_results), pd.DataFrame(all_test_results), final_train_seq


def DT2Vec_external(DTI_new, DTI_external_test, embeddings_seq, HIGHLY_POS, HIGHLY_NEG, multiclass=False):
    """
    This function ceates the machine learning model  
    
    Args: 
        DTI: A DataFrame of drug-target interactions
        embeddings_seq: A dataframe vectors of drugs and proteins 
        HIGHLY_POS: threshold for positive interactions 
        HIGHLY_NEG: threshold for negative interactions
        
    Returns: 
        all_train_results: Traning results for n folds
        all_test_results: Testing results for n folds
        df_DTI_proba: A dataframe of DTI with proba

    """
    
    df_high_pos = DTI_new[DTI_new["weight"]>= HIGHLY_POS]
    df_high_pos["L"] = 1
    
    df_high_neg = DTI_new[DTI_new["weight"]<= HIGHLY_NEG]
    df_high_neg["L"] = 0

    df_neutral = DTI_new[(DTI_new["weight"]<HIGHLY_POS) & (DTI_new["weight"]>HIGHLY_NEG)]
    df_neutral["L"] = 2
    
    if multiclass:
        DTI_for_ML = (df_high_pos.append(df_high_neg)).append(df_neutral)
    else:
        DTI_for_ML = df_high_pos.append(df_high_neg)

    df_seq = concating(DTI_for_ML, embeddings_seq).drop(columns=["weight"])
    df_seq["weight"] = df_seq["L"]
    final_seq = df_seq.drop(columns=["L"])
        
    X, y = get_data(final_seq)


    if multiclass:
        model = XGBClassifier(objective='multi:softmax', learning_rate=0.9, max_depth=2, min_child_weight=2)
    else:
        model = XGBClassifier(learning_rate=0.9, max_depth=2, min_child_weight=2)

    model.fit(X, y)
    
    ## for the test-set ##
    df_high_pos_test = DTI_external_test[DTI_external_test["weight"]>= HIGHLY_POS]
    df_high_pos_test["L"] = 1
    
    df_high_neg_test = DTI_external_test[DTI_external_test["weight"]<=HIGHLY_NEG]
    df_high_neg_test["L"] = 0

    df_neutral = DTI_external_test[(DTI_external_test["weight"]<HIGHLY_POS) & (DTI_external_test["weight"]>HIGHLY_NEG)]
    df_neutral["L"] = 2
    
    if multiclass:
         DTI_for_ML_test = (df_high_pos_test.append(df_high_neg_test)).append(df_neutral_test)
    else:
        DTI_for_ML_test = df_high_pos_test.append(df_high_neg_test)

    df_seq_test = concating(DTI_for_ML_test, embeddings_seq).drop(columns=["weight"])
    df_seq_test["weight"] = df_seq_test["L"]
    final_seq_test = df_seq_test.drop(columns=["L"])
    
    X_test, y_test = get_data(final_seq_test)
    pred_test = model.predict(X_test)
    test_results = calculate_metrics(y_test, pred_test)
    del X
    del y
    
    
    return model, test_results, final_seq_test


def DT2Vec_external_best_model(model, DTI_external_test, embeddings_seq, HIGHLY_POS, HIGHLY_NEG, multiclass=False):
    """
    This function ceates the machine learning model  
    
    Args: 
        DTI: A DataFrame of drug-target interactions
        embeddings_seq: A dataframe vectors of drugs and proteins 
        HIGHLY_POS: threshold for positive interactions 
        HIGHLY_NEG: threshold for negative interactions
        
    Returns: 
        all_train_results: Traning results for n folds
        all_test_results: Testing results for n folds
        df_DTI_proba: A dataframe of DTI with proba

    """
    
    ## for the test-set ##
    df_high_pos_test = DTI_external_test[DTI_external_test["weight"]>= HIGHLY_POS]
    df_high_pos_test["L"] = 1
    
    df_high_neg_test = DTI_external_test[DTI_external_test["weight"]<= HIGHLY_NEG]
    df_high_neg_test["L"] = 0

    df_neutral = DTI_external_test[(DTI_external_test["weight"]<HIGHLY_POS) & (DTI_external_test["weight"]>HIGHLY_NEG)]
    df_neutral["L"] = 2
    
    if multiclass:
         DTI_for_ML_test = (df_high_pos_test.append(df_high_neg_test)).append(df_neutral_test)
    else:
        DTI_for_ML_test = df_high_pos_test.append(df_high_neg_test)

    df_seq_test = concating(DTI_for_ML_test, embeddings_seq).drop(columns=["weight"])
    df_seq_test["weight"] = df_seq_test["L"]
    final_seq_test = df_seq_test.drop(columns=["L"])
    
    X_test, y_test = get_data(final_seq_test)
    pred_test = model.predict(X_test)
    test_results = calculate_metrics(y_test, pred_test)
    
    return test_results, final_seq_test


# ##  DT2Vec 


# Embed drugs and proteins to vector 
embeddings_seq, embeddings_seq_drug, embeddings_seq_protein = embedding(DDS, PPS_seq, mapping=False)

chart, df_target_seq = plot_pca_2d_with_clusters(embeddings_seq, DDS, PPS_seq, pca_t='target')

chart, df_Lou_target = plot_pca_2d_with_Louvain_clusters(embeddings_seq, drug2cluster, target2cluster, type_data= 'target')

chart, df_Lou_drug = plot_pca_2d_with_Louvain_clusters(embeddings_seq, drug2cluster, target2cluster, type_data= 'drug')


# ## K-means Clustering

cluster_value = {}
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score, roc_curve, auc


for n_clusters in range(4,5):
    clusterer = MiniBatchKMeans(n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(embeddings_seq_drug)
    

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(embeddings_seq_drug, cluster_labels, random_state=42)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    calinski_harabaz_avg = calinski_harabasz_score(embeddings_seq_drug, cluster_labels)
    print("The average calinski_harabaz is :", calinski_harabaz_avg)

    davies_bouldin_avg= davies_bouldin_score(embeddings_seq_drug, cluster_labels)
    print("The average davies_bouldin_score is :", davies_bouldin_avg)
    
    cluster_value[n_clusters] = {'S':silhouette_avg, 'CH':calinski_harabaz_avg, 'DB':davies_bouldin_avg, 'labels':cluster_labels}

len(cluster_value[4]['labels'])

df_drug_kmeans = embeddings_seq_drug.copy()
df_drug_kmeans['label'] = cluster_value[4]['labels']
df_drug_kmeans

chart, df_drug_seq = plot_pca_2d_with_clusters(embeddings_seq, DDS, PPS_seq, pca_t='drug')


# ### Adding clusters to DTI

protein_dict = pd.Series(df_target_seq['type'].values, index = df_target_seq.index).to_dict()

dict_Lou_target = df_Lou_target['cluster'].to_dict()
dict_Lou_drug = df_Lou_drug['cluster'].to_dict()

DTI_cluster = DTI.copy()
DTI_cluster['target_C']= DTI_cluster['to'].map(dict_Lou_target) 
DTI_cluster['drug_C']= DTI_cluster['from'].map(dict_Lou_drug)
DTI_cluster['target_type']= DTI_cluster['to'].map(protein_dict) 
DTI_cluster['label']= DTI_cluster['weight']
DTI_cluster.loc[DTI_cluster['label'] <= HIGHLY_NEG, 'label'] = -100
DTI_cluster.loc[DTI_cluster['label'] >= HIGHLY_POS, 'label'] = 100
DTI_cluster['label'] = np.where((DTI_cluster['label']<HIGHLY_POS )&(DTI_cluster['label']>HIGHLY_NEG), -1 , DTI_cluster['label'])
DTI_cluster['label'] = DTI_cluster['label'].map({-100:'Negative',100:'Positive', -1:'Weak'})

DTI_cluster['from_ChEMBL'] = 'CHEMBL' + DTI_cluster['from'].astype(str)
DTI_cluster['to_ChEMBL'] = DTI_cluster['to'].astype(str).map(num2protein)


DTI_cluster.to_csv('DTI_cluster.csv')

with open('Lou_target.pkl', 'wb') as fp:
    pickle.dump(dict_Lou_target, fp)

with open('Lou_drug.pkl', 'wb') as fp:
    pickle.dump(dict_Lou_drug, fp)
    
with open('protein_dict.pkl', 'wb') as fp:
    pickle.dump(protein_dict, fp)


# split data to train-validation and external test 
DTI_external_test = DTI.sample(frac = 0.1)
DTI_new = DTI.drop(DTI_external_test.index)
print (f'DTI: {len(DTI)}')
print (f'DTI_external_test: {len(DTI_external_test)}')
print (f'DTI_train_internal_test: {len(DTI_new)}')
DTI_external_test.head()


# XGBoost
resuts_all_runs_list = []
resuts_external_list = []

for i in range(0,1):
    DTI_external_test = DTI.sample(frac = 0.1)
    DTI_new = DTI.drop(DTI_external_test.index)
    
    all_train_results, all_test_results, final_train_seq = DT2Vec(DTI_new, embeddings_seq, HIGHLY_POS, HIGHLY_NEG, n=10, multiclass=False)
    resuts_all_runs_list.append(all_test_results) 
    # select the best model based on "average_precision_score"
    model = all_test_results.sort_values(by=['average_precision_score'], ascending=False)['model'][0]
    test_results, final_seq_test = DT2Vec_external_best_model(model, DTI_external_test, embeddings_seq, HIGHLY_POS, HIGHLY_NEG, multiclass=False)
    resuts_external_list.append(test_results) 

resuts_all_runs = pd.concat(resuts_all_runs_list, ignore_index=True)
resuts_external = pd.DataFrame(resuts_external_list)

# results on external-testset (5 runs)--> best model (of 10 fold CV) was selected and was applied on external testsets in each run
resuts_external




