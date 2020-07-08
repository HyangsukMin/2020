# %% 
########################################
## Setup and Import
########################################
from datetime import datetime
from itertools import combinations
from tqdm import tqdm
import random
import math

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm 
from sklearn.metrics import auc, precision_recall_curve, roc_curve, accuracy_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


#%% 
################################################################################
## Utilities
################################################################################
def hedge_to_pg_nodes(hedge, p):
    nodes = []
    if p == 2:
        nodes = list(hedge)
    else:
        for combination in combinations(hedge, p - 1):
            nodes.append(combination)

    return nodes

def hedges_to_pg(edgelist, p):
    print(datetime.now(),"Generating {}-projected graph".format(p))

    assert (p >= 2)
    w = 1
    pg = nx.Graph()
    node_size = p - 1
    n_edges = 0
    for idx, edge in enumerate(edgelist):
        if len(edge) < p:
            continue
        
        if p == 2:
            nodes = edge
        else :
            nodes = []
            for node in combinations(edge,node_size):
                nodes.append(node)

        for node1, node2 in combinations(nodes,2):
            if p > 2:
                if len(set(node1 + node2)) > 2:
                    continue
            if pg.has_edge(node1,node2):
                pg[node1][node2]['weight'] += w
            else:
                pg.add_edge(node1, node2, weight=1)
    return pg

def get_edgelist(FilePath = "./data/paper_author.txt"):
    edges = []
    with open(FilePath,'r') as f:
        line = f.readline()
        for line in f:
            if len(line) == 0:
                break
            edges.append(list(map(int,line.rstrip().split())))
    return edges

def get_answers(FilePath ='data/answer_public.txt'):
    answers = []    
    with open(FilePath,'r') as f:
        for i, line in enumerate(f):
            answers.append(int(eval(line.strip())))
    answers = np.asarray(answers)
    return answers

################################################################################
## Features_1
################################################################################    
'''
Jaccard Similarity
Common Neighbors
Ademic Adar
Resource Allocation
Hub Promoted Index
'''
def _get_neighbors(pg, node):
    if node in pg:
        neighbors = set(pg.neighbors(node))
    else:
        neighbors = set([])

    return neighbors
def _get_degree(pg,node):
    if node in pg:
        degree = pg.degree[node]
    else:
        degree = 1
    return degree

def neighbors(hedge, pg, p):
    nodes = hedge_to_pg_nodes(hedge, p)
    cns = {}
    uns = {}
    degree = []
    first = True
    for node in nodes:
        if first:
            prev_n = _get_neighbors(pg, node)
            first = False
        neighbors = _get_neighbors(pg, node)
        degree.append(_get_degree(pg,node))
        cns = prev_n.intersection(neighbors)
        uns = prev_n.union(neighbors)

    # CN
    cn = len(cns)
    
    # HPI
    if len(degree) == 0:
        hpi = 0
    else :
        min_degree = sum(degree)/(len(degree))
        hpi = cn/min_degree

    if len(uns) == 0:
        jc = 0
    else:
        jc = len(cns) / float(len(uns))

    aa = 0
    ra = 0
    for node in cns:
        aa += 1 / float(math.log(len(_get_neighbors(pg, node))))
        ra += 1 / float(len(_get_neighbors(pg,node)))
    return cn, jc, aa, ra, hpi

################################################################################
## Features_2
################################################################################    
'''
Arithmetic Mean
Harmonic Mean
'''
def means(hedge, pg, p):
    w_sum = 0  # arithmetic
    w_mul = 1  # geometric
    w_invsum = 0  # harmonic
    N = 0
    for combination in combinations(hedge, p):
        if p == 2:
            node1 = combination[0]
            node2 = combination[1]
        if p == 3:
            node1 = (combination[0], combination[1])
            node2 = (combination[0], combination[2])
        if p == 4:
            node1 = (combination[0], combination[1], combination[3])
            node2 = (combination[0], combination[2], combination[3])
        if pg.has_edge(node1, node2):
            w = pg[node1][node2]['weight']
        else:
            w = 0
        w_sum += w
        if w != 0:
            w_invsum += 1 / float(w)
        N += 1
    N += 0.0001
    am = w_sum / float(N)
    if w_invsum != 0:
        hm = float(N) / w_invsum
    else:
        hm = 0
    return hm, am

################################################################################
## Features_2
################################################################################   
'''
number of nodes
'''
def number_of_nodes(hedge):
    nn = len(hedge)
    return nn

################################################################################
## Features_Concat
################################################################################    
def _append_features(vector_dict, hedge, pg, p):
    hm, am = means(hedge, pg, p)
    vector_dict["hm"].append(hm)
    vector_dict["am"].append(am)
    cn, jc, aa, ra, hpi = neighbors(hedge, pg, p)
    vector_dict["cn"].append(cn)
    vector_dict["jc"].append(jc)
    vector_dict["aa"].append(aa)
    vector_dict['ra'].append(ra)
    vector_dict['hpi'].append(hpi)
    nn = number_of_nodes(hedge)
    vector_dict['nn'].append(nn)
    return vector_dict

def get_feature_vector_dict(hedge, pg,p):
    vector_dict = {"hm": [],
                   "am": [],
                   "cn": [],
                   "jc": [],
                   "aa": [],
                   "ra": [],
                   'hpi': [],
                   'nn': []}    
    
    vector_dict = _append_features(vector_dict, hedge, pg,p)

    for key in vector_dict:
        vector_dict[key] = np.asarray(vector_dict[key])

    return vector_dict

def get_feature_vectors_dict(hedges, pg,p):
    vectors_dict = {"hm": [],
                    "am": [],
                    "cn": [],
                    "jc": [],
                    "aa": [],
                    "ra": [],
                    'hpi': [],
                    'nn': []}

    for hedge in tqdm(hedges):
        for key in vectors_dict:
            vector = get_feature_vector_dict(hedge, pg,p)[key]
            vectors_dict[key].append(vector)

    for key in vectors_dict:
        vectors_dict[key] = np.asarray(vectors_dict[key])

    return vectors_dict

################################################################################
## Evaluation & Draw Curves
################################################################################
def print_evaluation(y_true, y_pred):
    # precision, recall, fpr, tpr
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    auc_pr = auc(recall, precision)
    auc_roc = auc(fpr, tpr)
    print("---------------------------------------------------------")
    print("auc_pr  : {}".format(auc_pr))
    print("auc_roc : {}".format(auc_roc))

#%%
################################################################################
## Data Preprocess
################################################################################    
# Get Hyper-EdgeList and Construct Projection Graph
def get_pg(dim):
    edgelist = get_edgelist()
    pg2 = hedges_to_pg(edgelist,2)
    pg3 = hedges_to_pg(edgelist,3)
    pg4 = hedges_to_pg(edgelist,4)
    return pg2, pg3, pg4

def get_data(pg2,pg3,pg4,dim,features=["hm",  "am", "cn", "jc", "aa", "ra", 'hpi', 'nn'], test = False):
    if test :
        hedges = get_edgelist("./data/query_private.txt")
    else :
        hedges = get_edgelist("./data/query_public.txt")
    answers = get_answers()
    # features = ["hm",  "am", "cn", "jc", "aa", "ra",'hpi','nn']
    print("got {}".format(features))
    features = features
    print(datetime.now(),"Extracting features from 2-projected graph.")
    vectors_dict = get_feature_vectors_dict(hedges, pg2,2)
    vectors2 = np.concatenate([vectors_dict[feature] for feature in features], axis=1)
    vectors = vectors2
    if dim > 1 :
        print(datetime.now(),"Extracting features from 3-projected graph.")
        vectors_dict = get_feature_vectors_dict(hedges, pg3,3)
        vectors3 = np.concatenate([vectors_dict[feature] for feature in features], axis=1)  
        vectors = np.concatenate([vectors, vectors3],axis=1)
    if dim > 2:
        print(datetime.now(),"Extracting features from 4-projected graph.")
        vectors_dict = get_feature_vectors_dict(hedges, pg4,4)
        vectors4 = np.concatenate([vectors_dict[feature] for feature in features], axis=1)  
        vectors = np.concatenate([vectors, vectors4],axis=1)
    print(vectors.shape)
    if test :
        return vectors
    else :
        return vectors, answers

#%%
def train_test(vectors, answers, test_size = 0.2, clf_types=['lr','rf','lgbm']):
    
    X_train, X_test, y_train, y_test = train_test_split(vectors, answers, 
                                                            test_size = test_size, 
                                                            random_state = 1,
                                                            stratify = answers)
    print(X_train.shape, y_train.shape)
    #%%
    clf_lr = LogisticRegression(random_state=0,
                                penalty= 'l2',
                                multi_class='ovr',
                                solver='lbfgs',
                                C=1.0,
                                max_iter=100000)

    clf_rf = RandomForestClassifier(random_state=0,
                                    n_estimators = 1000,
                                    max_depth = 13,
                                    max_leaf_nodes = 280)

    clf_lgbm = lgbm.LGBMClassifier(random_state = 0,
                                    learning_rate = 0.005,
                                    n_estimators = 1000,
                                    num_leaves = 25,
                                    max_depth = 50,
                                    max_bin = 300,
                                    colsample_bytree = 0.8,
                                    subsample = 0.8,
                                    reg_alpha = 1.2,
                                    reg_lambda = 1.4)   

    models = []
    ################################################################################
    ## Training & Testing : Logistic Regression
    ################################################################################   
    if 'lr' in clf_types:
        clf_type = "Logistic Regression"
        print("========== Train/testing..",clf_type,"==========")

        clf_lr.fit(X_train, y_train)
        y_pred = clf_lr.predict_proba(X_test)

        print("Random Guessing : ", sum(answers)/len(answers))
        print("Test",len(y_pred)," Accuracy : ",accuracy_score(y_pred[:,1].round(0), y_test))
        print("Train",len(y_train)," Accuracy : ",accuracy_score(clf_lr.predict(X_train),y_train))
        print("Total",len(answers)," Accuracy : ",accuracy_score(clf_lr.predict(vectors),answers))

        print_evaluation(y_test, y_pred[:,1])
        print('=' * 60)
        models.append(clf_lr)

        ################################################################################
        ## Training & Testing : RandomForest
        ################################################################################
    if 'rf' in clf_types:
        clf_type = 'RandomForest'
        print("========== train/testing..",clf_type,"==========")

        clf_rf.fit(X_train, y_train)

        y_pred = clf_rf.predict_proba(X_test)
        print("Random Guessing : ", sum(answers)/len(answers))
        print("Test",len(y_pred)," Accuracy : ",accuracy_score(y_pred[:,1].round(0), y_test))
        print("Train",len(y_train)," Accuracy : ",accuracy_score(clf_rf.predict(X_train),y_train))
        print("Total",len(answers)," Accuracy : ",accuracy_score(clf_rf.predict(vectors),answers))

        print_evaluation(y_test, y_pred[:,1])
        print('=' * 60)
        models.append(clf_rf)

        ################################################################################
        ## Training & Testing : LGBMClassifier
        ################################################################################   
    if 'lgbm' in clf_types:
        clf_type = 'lgbm'
        print("========== train/testing..",clf_type,"==========")
        clf_lgbm.fit(X_train, y_train)

        y_pred = clf_lgbm.predict_proba(X_test)
        print("Random Guessing : ", sum(answers)/len(answers))
        print("Test",len(y_pred)," Accuracy : ",accuracy_score(y_pred[:,1].round(0), y_test))
        print("Train",len(y_train)," Accuracy : ",accuracy_score(clf_lgbm.predict(X_train),y_train))
        print("Total",len(answers)," Accuracy : ",accuracy_score(clf_lgbm.predict(vectors),answers))

        print_evaluation(y_test, y_pred[:,1])
        print('=' * 60)
        models.append(clf_lgbm)
    print(clf_types)
    return models
# %%
################################################################################
## Ablation Study
################################################################################
pg2,pg3,pg4 = get_pg(3)

# # ## One Feature with 2-pg, 3-pg ,4-pg
# features=["hm",  "am", "cn", "jc", "aa", "ra", 'hpi', 'nn']
# for feature in features:
#     print("\n######################################################")
#     print("========================={}==========================".format(feature))
#     print("######################################################")
#     vectors, answers = get_data(pg2,pg3,pg4,3, features = [feature])
#     train_test(vectors, answers, clf_types=['lr','rf','lgbm']) # 'stack',

# d = 1
# print("\n######################################################")
# print("=========================# of pg {}==========================".format(d))
# print("######################################################")
# vectors, answers = get_data(pg2,pg3,pg4,d, features=["hm",  "am", "cn", "jc", "aa", "ra","hpi","nn"])
# train_test(vectors, answers, clf_types=['lr','rf','lgbm'])#,'rf','lgbm','svm','xgb','stack','voting']) # ,

# d = 2
# print("\n######################################################")
# print("=========================# of pg {}==========================".format(d))
# print("######################################################")
# vectors, answers = get_data(pg2,pg3,pg4,d, features=["hm",  "am", "cn", "jc", "aa", "ra","hpi","nn"])
# train_test(vectors, answers, clf_types=['lr','rf','lgbm']) # ,

# d = 3
# print("\n######################################################")
# print("=========================# of pg {}==========================".format(d))
# print("######################################################")
# vectors, answers = get_data(pg2,pg3,pg4,d, features=["hm",  "am", "cn", "jc", "aa", "ra","hpi","nn"])
# train_test(vectors, answers, clf_types=['lr','rf','lgbm']) # ,

#%%
################################################################################
# FINAL
################################################################################
d = 2
print("\n######################################################")
print("=========================# of pg {}==========================".format(d))
print("######################################################")
vectors, answers = get_data(pg2,pg3,pg4,d, features=["hm",  "am", "cn", "jc", "aa", "ra", "hpi", "nn"])
models = train_test(vectors, answers, clf_types=['rf']) # ,

test_vectors = get_data(pg2,pg3,pg4, d, features=["hm",  "am", "cn", "jc", "aa", "ra", "hpi", "nn"],test=True)
models[0].predict(test_vectors)
#%%
submission = models[0].predict(test_vectors)
submission2 = [True if x==1 else False for x in submission ]
submission2 = list(map(str,submission2))

# with open("answer_private.txt","w") as f:
#     f.writelines(["%s\n" % item  for item in submission])
with open("answer_private.txt","w") as f:
    f.writelines(["%s\n" % item  for item in submission2])
