# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 19:10:29 2022

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


# =============================================================================
# GAT
# =============================================================================

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h

import time
import numpy as np


features = g.ndata['features']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask'] 
test_mask = g.ndata['test_mask'] 

# create the model, 2 heads, each head has hidden size 8
net = GAT(g,
          in_dim=features.size()[1],
          hidden_dim=32,
          out_dim=2,
          num_heads=2)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
best_val_acc = 0
best_test_acc = 0
# main loop
dur = []
for epoch in range(20):
    if epoch >= 3:
        t0 = time.time()
    net.train(True)

    logits = net(features)
    pred = logits.argmax(1)
    logp = F.log_softmax(logits, 1)
    loss = F.cross_entropy(logits[train_mask], label[train_mask])
    
    #compute accuracy 
    net.train(False)
    train_acc = (pred[train_mask]==labels[train_mask]).float().mean()
    val_acc = (pred[val_mask]==labels[val_mask]).float().mean()
    test_acc = (pred[test_mask]==labels[test_mask]).float().mean()
    
    # save the best one
    if best_val_acc <= val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)
    if epoch % 5 == 0:
        print(("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | val acc {:.3f}  | test acc: {:.3f}".format(
            epoch, loss.item(), np.mean(dur), val_acc, test_acc)))

    #print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
      #  epoch, loss.item(), np.mean(dur)))

