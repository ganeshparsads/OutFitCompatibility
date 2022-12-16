import itertools
import os
import warnings

import cv2
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn import metrics
from IPython.display import display

def loadimg_from_id(ID, root_dir=test_dataset.root_dir):
    """load image from pre-defined id.
    
    Args:
        ID: List of ids of 5 items.
        
    Return:
        imgs: torch.tensor of shape (1, 5, 3, 224, 224)
    """
    imgs = []

    count = 1
    for id in ID:
        if 'mean' in id:
            img_path = os.path.join(test_dataset.data_dir, id.split('_')[0]) + '.png'
        else:
            img_path = os.path.join(root_dir, *id.split('_')) + '.jpg'

        img = Image.open(img_path).convert('RGB')
        print("------------------------ Layer " + str(count) + "------------------------")
        display(img)
        print("-------------------------------------------------")
        img = test_dataset.transform(img)
        imgs.append(img)
        count += 1
    imgs = torch.stack(imgs)
    imgs = imgs.unsqueeze(0)
    return imgs

def defect_detect(img, model, normalize=True):
    """ Compute the gradients of each element in the comparison matrices to 
    approximate the problem of each input.
    
    Args:
        img: images of shape (N, 3, 224, 224).
        model: the model to compute the compatibility score.
        normalize: whether to normalize the relation results.
        
    Return:
        relation: gradients on comparison matrix.
        out: prediction score.
    """
    # Register hook for comparison matrix
    relation = None
    def func_r(module, grad_in, grad_out):
        nonlocal relation
        relation = grad_in[1].detach()

    for name, module in model.named_modules():
        if name == 'predictor.0':
            module.register_backward_hook(func_r)

    # Forward
    out, *_ = model._compute_score(img)
    one_hot = torch.FloatTensor([[-1]]).to(device)

    # Backward
    model.zero_grad()
    out.backward(gradient=one_hot, retain_graph=True)
    
    if normalize:
        relation = relation / (relation.max() - relation.min())
    relation += 1e-3
    return relation, out.item()

def vec2mat(relation, select):
    """ Convert relation vector to 4 matrix, which is corresponding to 4 layers
    in the backend CNN.
    
    Args:
        relation: (np.array | torch.tensor) of shpae (60,)
        select: List of select item indices, e.g. (0, 2, 3) means select 3 items
            in total 5 items in the outfit.
        
    Return:
        mats: List of matrix
    """
    mats = []
    for idx in range(4):
        mat = torch.zeros(5, 5)
        mat[np.triu_indices(5)] = relation[15*idx:15*(idx+1)]
        mat += torch.triu(mat, 1).transpose(0, 1)
        mat = mat[select, :]
        mat = mat[:, select]
        mats.append(mat)
    return mats

def show_rela_diagnosis(relation, select, cmap=plt.cm.Blues):
    """ Visualize diagnosis on relationships of 4 scales
    
    Args:
        relation: (np.array | torch.tensor) of shpae (60,)
        select: List of select item indices
    """
    mats = vec2mat(relation , select)
        
    fig = plt.figure(figsize=(20, 5))
    all_names = {0:'Top', 1:'Bottom', 2:'Shoe', 3:'Bag', 4:'Accssory'}
    node_names = {i:all_names[s] for i, s in enumerate(select)}
    
    edge_vmax = max(m.max() for m in mats)
    edge_vmin = min(m.min() for m in mats)
    
    container = []
    for idx in range(4):
        A = mats[idx]
        if isinstance(A, torch.Tensor):
            A = A.cpu().data.numpy()
                   
        A = np.triu(A, k=1)
        A = np.round(A, decimals=2)
        container.append(A)
    container = np.stack(container)
    sorted_vedge = sorted(container.ravel(), reverse=True)
        
    for idx in range(4):
        plt.subplot(1, 4, idx+1)
        plt.title("Layer {}".format(idx+1), fontsize=28)
        A = mats[idx]
        if isinstance(A, torch.Tensor):
            A = A.cpu().data.numpy()
                   
        A = np.triu(A, k=1)
        A = np.round(A, decimals=2)
        indices = np.triu_indices(A.shape[0], k=1)
        weights = A[indices[0], indices[1]]
        # Generate graph
        G = nx.Graph()
        for i, j, weight in zip(*indices, weights):
            G.add_edge(node_names[i], node_names[j], weight=weight)
        
        elarge, esmall, filtered_weights = [], [], []
        for e in G.edges(data=True):
            if e[2]['weight'] in sorted_vedge[:3]:
                elarge.append((e[0], e[1]))
            else:
                esmall.append((e[0], e[1]))
                filtered_weights.append(e[2]['weight'])
        pos=nx.circular_layout(G) # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes()], node_size=1600, node_color='#A0CBE2')

        # edges
        nx.draw_networkx_edges(G,pos,edgelist=esmall, width=6, alpha=0.5, edge_color=filtered_weights, edge_cmap=cmap,
                               edge_vmax=edge_vmax, edge_vmin=edge_vmin)
        nx.draw_networkx_edges(G,pos,edgelist=elarge, width=6, alpha=0.5, edge_color='red', style='dashed')

        # labels
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_labels(G,pos, font_size=18, font_family='Times New Roman')
        if len(select) == 4:
            nx.draw_networkx_edge_labels(G, pos, font_size=18, font_family='Times New Roman', edge_labels=labels, label_pos=0.33)
        else:
            nx.draw_networkx_edge_labels(G, pos, font_size=18, font_family='Times New Roman', edge_labels=labels)
        
        plt.axis('off')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    # plt.show()
    plt.savefig("rela_diagnosis.pdf")
    print("Save diagnosis result to rela_diagnosis.pdf")
    
def item_diagnosis(relation, select):
    """ Output the most incompatible item in the outfit
    
    Return:
        result (list): Diagnosis value of each item 
        order (list): The indices of items ordered by its importance
    """
    mats = vec2mat(relation, select)
    for m in mats:
        mask = torch.eye(*m.shape).byte()
        m.masked_fill_(mask, 0)
    result = torch.cat(mats).sum(dim=0)
    order = [i for i, j in sorted(enumerate(result), key=lambda x:x[1], reverse=True)]
    return result, order
