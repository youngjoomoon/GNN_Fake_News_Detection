# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:25:01 2022

@author: myj
"""

import torch
import pandas as pd
import dgl
import torch 
import torch.nn as nn
import dgl.function as fn
import numpy as np
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, GATConv, HeteroGraphConv
from transformers import BertTokenizer
from transformers import BertConfig, BertModel
from sentence_transformers import SentenceTransformer
import numpy as np 
from scipy.sparse import coo_matrix

nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('train.csv')


# =============================================================================
# 유튜브 문장 임베딩 
# =============================================================================
youtube_embeddings = []
for idx, txt in df.iterrows():
    embeddings = torch.zeros(384,)
    embeddings = embeddings.reshape(1, -1)
    i = 0
    while i<10:
        sentence = txt[f'youtube{i}']
        i += 1
        #print('==================',i, '==================')
        embedding = nlp_model.encode(sentence)
        embedding = embedding.reshape(1, -1)
        embeddings += embedding
    youtube_embeddings.append(embeddings)
###  youtube_embeddings

# =============================================================================
# 약 4600개의 단어를 바탕으로 Frequency top 150 단어들만 추려서 원핫인코딩  
# =============================================================================
# Remove Unnecessary Characters  
df['keybert_keywords'] = df['keybert_keywords'].str.replace('[', '').str.replace(']','')\
    .str.replace("'",'').str.replace(' ','')

# bert 단어들을 나누기 (One-hot encoding 작업 )
keyword_list = []
for idx, keywords in df.iterrows():
    for keyword in keywords['keybert_keywords'].split(','):
        keyword_list.append(keyword)

keyword_df = pd.DataFrame({'keyword': keyword_list})
keyword_list_freq = keyword_df.value_counts().head(150).reset_index()['keyword'].to_list()

# word indexing 
words_to_index = {word: index for index, word in enumerate(keyword_list_freq)}

# one hot encdoing function 
def one_hot_encoding(word, words_to_index):
  one_hot_vector = [0]*(len(words_to_index))
  if word in words_to_index:
      index = words_to_index[word]
      one_hot_vector[index] = 1
  else: 
      one_hot_vector = torch.zeros(len(words_to_index),)
  one_hot_vector = torch.tensor(one_hot_vector)
  one_hot_vector = one_hot_vector.reshape(1, -1)
  return one_hot_vector
    
# keyword one-hot
keyword_embeddings = []
for idx, keywords in df.iterrows():
    key_embeddings = torch.zeros(1, 150)
    for keyword in keywords['keybert_keywords'].split(','):
        key_embedding = one_hot_encoding(keyword, words_to_index)
        #print(key_embedding.size())
        key_embeddings += key_embedding
    keyword_embeddings.append(key_embeddings)
    
### keyword_embeddings

# =============================================================================
# Claim embedding   
# =============================================================================
claim_embeddings_list = []
for idx, row in df.iterrows():
    claim_embeddings = torch.zeros(384,)
    claim_embeddings = claim_embeddings.reshape(1, -1)
    sentence = row['claim']
    claim_embedding = nlp_model.encode(sentence)
    claim_embedding = claim_embedding.reshape(1,-1)
    claim_embeddings += claim_embedding
    claim_embeddings_list.append(claim_embeddings)

claim_embeddings_list
#claim_embeddings_list




# =============================================================================
# Graph
# =============================================================================
dic_array = np.zeros((len(df),len(df)))
print(dic_array.shape)
print(dic_array.sum())



for i, i_keyword in enumerate(df['keybert_keywords']):
    i_index = 0
    imax_index = len(i_keyword.split(','))
    while i_index < imax_index:
        i_split = i_keyword.split(',')
        for j, j_keyword in enumerate(df['keybert_keywords']):
            j_index = 0
            jmax_index = len(j_keyword.split(','))
            while j_index < jmax_index:
                j_split = j_keyword.split(',')
                if i_split[i_index] == j_split[j_index]:
                    dic_array[i][j] += 1
                #print(j_index)
                j_index += 1
        i_index +=1 



claim_embeddings_tensor = torch.stack(claim_embeddings_list, 0).squeeze(1)
keyword_embeddings_tensor = torch.stack(keyword_embeddings, 0).squeeze(1)
youtube_embeddings_tensor = torch.stack(youtube_embeddings, 0).squeeze(1)

features = torch.cat((claim_embeddings_tensor, keyword_embeddings_tensor, youtube_embeddings_tensor), 1)
label = torch.from_numpy(df['label'].astype('category').to_numpy())


# graph
edge_index = torch.nonzero(torch.tensor(dic_array), as_tuple=False).T

g = dgl.graph((edge_index[0], edge_index[1]))
#g = dgl.add_self_loop(g)  #?? 
g.ndata['features'] = features
g.ndata['label'] = label
g.edata['features'] = torch.tensor(coo_matrix(torch.tensor(dic_array)).data)
g = dgl.add_self_loop(g) 

g.ndata
g.edata

# =============================================================================
# MASKING
# =============================================================================
n_nodes = df.shape[0]
n_train = int(n_nodes * 0.6)
n_val = int(n_nodes * 0.2)
train_mask = torch.zeros(n_nodes, dtype=torch.bool)
val_mask = torch.zeros(n_nodes, dtype=torch.bool)
test_mask = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[:n_train]  = True
val_mask[n_train:n_train+n_val] = True
test_mask[n_train+n_val:] = True

g.ndata['train_mask'] = train_mask
g.ndata['val_mask'] = val_mask
g.ndata['test_mask'] = test_mask



# Contruct a two-layer GNN model
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
        self.dropout = dropout

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.dropout(h, self.dropout, training = self.training)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
    

# Create the model with given dimensions
model = SAGE(g.ndata['features'].shape[1], 64, 2, 0.5)

def train(g, model):
    # 학습에 필요한 요소 정의 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    best_val_acc = 0
    best_test_acc = 0 
    
    features = g.ndata['features']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    #val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    for e in range(150):
        # Forward
        model.train(True)
        logits = model(g, features)
    
        # Compute prediction
        pred = logits.argmax(1) #값이 가장 높은 것 추출
        
        # Compute loss 
        #print(label[train_mask])
        loss = F.cross_entropy(logits[train_mask], label[train_mask])
    
        # Compute accuracy on training/validation/test
        model.train(False)
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        #val_acc = (pred[val_mask] == labels[val_mask]).float().mean() # 정석은 model.eval() 해줘야함
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    
        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < test_acc:
            best_val_acc = test_acc
            best_test_acc = test_acc
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, test acc: {:.3f} (best {:.3f})'.format(
                e, loss, test_acc, best_test_acc))
    #torch.save(model.state_dict(), 'gcn_model')

train(g, model)


torch.save(model.state_dict(), 'graphsage_model_v5_h64_early')
