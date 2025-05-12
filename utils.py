import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from datetime import datetime
from torch_geometric.utils import to_dense_adj
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from sklearn.metrics.pairwise import pairwise_distances
from metrics import calculate_accuracy_and_f1
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
try:
    from sklearnex import patch_sklearn
except:
    def patch_sklearn(): return
from functools import partial
from sklearn.metrics import cluster
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
    

def add_sim_edge(x, dataset_name):
    cosine_sim = {"cora": 0.4, "citeseer": 0.4, "wikics": 0.3, "pubmed": 0.5}
    threshold = cosine_sim[dataset_name]
    x_norm = F.normalize(x, p=2, dim=1)
    sim_full = torch.mm(x_norm,x_norm.T)
    sim_full[torch.eye(sim_full.shape[0], dtype=torch.bool)] = -torch.inf
    edge_mask = (sim_full > threshold)
    src, dst = torch.where(edge_mask)
    edge_index = to_undirected(torch.stack([src, dst], dim=0))
    return edge_index


def clustering(feature, n_clusters, true_labels=None, kmeans_device='cpu',device=torch.device('cuda:0'), seed=42, batch_size=100000, tol=1e-3, spectral_clustering=False, smooth=False, alpha=10):
    
    if spectral_clustering:
        if isinstance(feature, torch.Tensor):
            feature = feature.numpy()
        print("spectral clustering on cpu...")
        patch_sklearn()
        Cluster = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', random_state=seed)
        f_adj = np.matmul(feature, np.transpose(feature))
        predict_labels = Cluster.fit_predict(f_adj)
    else:
        if kmeans_device == 'cuda':
            if isinstance(feature, np.ndarray):
                feature = torch.tensor(feature)
            if not smooth:
                print("kmeans on gpu...")
                predict_labels, _ = kmeans(X=feature, num_clusters=n_clusters, batch_size=batch_size, tol=tol, device=device, seed=seed)
            else:
                #alpha = 4 * feature.shape[0] / torch.sum(feature ** 2)
                print("smooth kmeans on gpu...")
                predict_labels, _, _ = smooth_kmeans(X=feature, num_clusters=n_clusters, alpha=alpha, batch_size=batch_size, tol=tol, device=device, seed=seed)
                
            predict_labels = predict_labels.numpy()
        else:
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
            print("kmeans on cpu...")
            patch_sklearn()
            Cluster = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=20)
            predict_labels = Cluster.fit_predict(feature)
    if true_labels is None:
        return predict_labels
    else:
        acc, f1, caa = calculate_accuracy_and_f1(true_labels, predict_labels)
        nmi = normalized_mutual_info_score(true_labels, predict_labels)
        ari = adjusted_rand_score(true_labels, predict_labels)
    return predict_labels, acc, caa, ari, nmi, f1


def set_seed_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def log(*args):
    print(f'[{datetime.now()}]', *args)


def delete_non_tensor_attributes(data):
    for attr_name in data.keys:
        if not isinstance(data[attr_name], torch.Tensor):
            delattr(data, attr_name)
    return data

def scale(z: torch.Tensor):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / ((zmax - zmin) + 1e-20)
    z_scaled = z_std
    return z_scaled

def mean_var(z: torch.Tensor):
    zmean = z.mean(dim=0)
    zstd = z.std(dim=0)
    z_mean_var = (z - zmean) / (zstd + 1e-20)
    return z_mean_var

def initialize(X, num_clusters, seed):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param seed: (int) seed for kmeans
    :return: (np.array) initial state
    """
    num_samples = len(X)
    if seed == None:          
        indices = np.random.choice(num_samples, num_clusters, replace=False)
    else:
        np.random.seed(seed)            
        indices = np.random.choice(num_samples, num_clusters, replace=False)            
    initial_state = X[indices]                      
    
    return initial_state

def boltzmann_weights(D, alpha=1, device=torch.device('cpu')):
    """
    calculate boltzmann_weights
    :param D: (torch.tensor) distance matrix, N*K 
    """

    exp_terms = torch.exp(-alpha * D)
    sum_exp = torch.sum(exp_terms, axis=1, keepdim=True)
    J = torch.sum(D * exp_terms, axis=1, keepdim=True) / (sum_exp + 1e-10)
    W = exp_terms / (sum_exp + 1e-10) * (1 - alpha * ( D - J)) #N*K 
    #print(W.device, device)
    zero_mask = torch.isclose(torch.sum(W, 1), torch.tensor(0.0).to(device))
    if torch.any(zero_mask):
        min_indices = torch.argmin(D[zero_mask], dim=1)
        W[zero_mask] = 0
        W[zero_mask, min_indices] = 1
    
    return W

def smooth_kmeans(
        X,
        num_clusters,
        alpha=10,
        distance='euclidean',
        batch_size=100000,
        cluster_centers=[],
        tol=1e-3,
        tqdm_flag=False,
        iter_limit=500,
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        seed=None
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if tqdm_flag:
        print(f'running smooth k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(smooth_pairwise_distance, batch_size=batch_size, device=device, tqdm_flag=tqdm_flag, alpha=alpha)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)
    if type(cluster_centers) == list:
        initial_state = initialize(X, num_clusters, seed=seed)
        initial_state = torch.tensor(initial_state).to(device)
    else:
        if tqdm_flag:
            print('resuming')
        # calculate centroids with soft weighted
        initial_state = cluster_centers       
        smooth_weight = pairwise_distance_function(X, initial_state) #N*K
        numerator = torch.mm(smooth_weight.T, X)
        initial_state = numerator / smooth_weight.sum(dim=0).unsqueeze(1)
        initial_state = initial_state.to(device)
    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running smooth kmeans]')
    while True:
        smooth_weight, choice_cluster = pairwise_distance_function(X, initial_state)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            smooth_weight_index = smooth_weight[:,index].unsqueeze(-1)
            sum_w_index = torch.sum(smooth_weight_index)
            if sum_w_index < 1e-16:
                initial_state[index] = X[torch.randint(0, X.shape[0], (1,))].to(device)
            else:
                initial_state[index] = (X * smooth_weight_index).sum(0) / sum_w_index 

        center_shift = torch.norm(initial_state - initial_state_pre) / (torch.norm(initial_state_pre) + 1e-16)

        # increment iteration
        iteration = iteration + 1        

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster.cpu(), initial_state.cpu(), smooth_weight.cpu() #assignment, centroids

def smooth_pairwise_distance(data1, data2, alpha=1, batch_size=100000, device=torch.device('cpu'), tqdm_flag=True):
    # return smooth weights, N*K
    if tqdm_flag:
        print(f'device is :{device}')

    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*K*M
    B = data2.unsqueeze(dim=0)
    if batch_size == -1:
        # full batch kmeans
        dis_ = 0.5 * (A - B) ** 2.0
        # return N*N matrix for pairwise distance
        dis_ = dis_.sum(dim=-1).squeeze()
        smooth_weight = boltzmann_weights(dis_, alpha)
        choice_cluster = torch.argmin(dis_, dim=1)
        return smooth_weight, choice_cluster
    else:
        # mini-batch kmeans
        dis = torch.zeros(data1.shape[0], data2.shape[0]).to(device)
        for batch_idx in range(int(np.ceil(data1.shape[0] / batch_size))):
            dis_ = 0.5 * (A[batch_idx * batch_size: (batch_idx + 1) * batch_size] - B) ** 2.0
            dis_ = dis_.sum(dim=-1).squeeze()
            dis[batch_idx * batch_size: (batch_idx + 1) * batch_size] = dis_
        choice_cluster = torch.argmin(dis, dim=1).long().to('cpu')
        smooth_weight = boltzmann_weights(dis, alpha, device)
        return smooth_weight, choice_cluster
                  

                  
                  
def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        batch_size=100000,
        cluster_centers=[],
        tol=1e-4,
        tqdm_flag=False,
        iter_limit=0,
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        seed=None
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if tqdm_flag:
        print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, batch_size=batch_size, device=device, tqdm_flag=tqdm_flag)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)
    if type(cluster_centers) == list:
        initial_state = initialize(X, num_clusters, seed=seed)
    else:
        if tqdm_flag:
            print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        choice_cluster = pairwise_distance_function(X, initial_state)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            # selected = idx[choice_cluster == index].to(device)
            # selected = X[selected]
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
            selected = torch.index_select(X, 0, selected)


            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster.cpu(), initial_state.cpu() #assignment, centroids


def kmeans_predict(
        X,
        cluster_centers,
        batch_size=100000,
        distance='euclidean',
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        tqdm_flag=True
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor) cluster ids
    """
    if tqdm_flag:
        print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, batch_size=batch_size, device=device, tqdm_flag=tqdm_flag)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    choice_cluster = pairwise_distance_function(X, cluster_centers, batch_size=batch_size)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, batch_size=100000, device=torch.device('cpu'), tqdm_flag=True):
    if tqdm_flag:
        print(f'device is :{device}')

    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)
    if batch_size == -1:
        # full batch kmeans
        dis_ = (A - B) ** 2.0
        # return N*N matrix for pairwise distance
        dis_ = dis_.sum(dim=-1).squeeze()
        return torch.argmin(dis_, dim=1)
    else:
        # mini-batch kmeans
        choice_cluster = torch.zeros(data1.shape[0])
        for batch_idx in range(int(np.ceil(data1.shape[0] / batch_size))):
            dis = (A[batch_idx * batch_size: (batch_idx + 1) * batch_size] - B) ** 2.0
            dis = dis.sum(dim=-1).squeeze()
            choice_ = torch.argmin(dis, dim=1)
            choice_cluster[batch_idx * batch_size: (batch_idx + 1) * batch_size] = choice_
        choice_cluster = choice_cluster.long()
        return choice_cluster

    
    

def read_jsonl(file_path):
    """
    Read a .jsonl file and return the contents as a list of dictionaries.

    Parameters:
    file_path (str): The path to the .jsonl file to be read.

    Returns:
    list: A list of dictionaries, each representing a JSON object.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def read_json(file_path):
    """
    Read a .jsonl file and return the contents as a list of dictionaries.

    Parameters:
    file_path (str): The path to the .jsonl file to be read.

    Returns:
    list: A list of dictionaries, each representing a JSON object.
    """
    data = []
    with open(file_path) as file:
        data=json.load(file)
    return data

def generate_chat_input_file(input_text, model_name = 'gpt-4o-mini'):
    jobs = []
    for i, text in enumerate(input_text):
        obj = {}
        obj['input'] = text
        jobs.append(obj)
    return jobs 

def get_k_hop_neighbors(data,node_idx,hop=2):
    edge_index = data.edge_index
    visited = set([node_idx]) 
    current_level = set([node_idx])
    for _ in range(hop):
        next_level = set()
        for u in current_level:
            neighbors = edge_index[1, edge_index[0] == u].tolist()
            for v in neighbors:
                if v not in visited:
                    visited.add(v)
                    next_level.add(v)
        current_level = next_level
    
    visited=list(visited)
    visited.remove(node_idx)
    
    for idx,i in enumerate(visited):
        if isinstance(i,torch.Tensor):
            visited[idx]=i.item()
    return visited

def get_top_k_neighbor_simcse(data,sampled_node_idxs,g_feat,k=2,hop=2):
    neighbor_dict = {}
    for i in tqdm(sampled_node_idxs):
        
        neighbors = get_k_hop_neighbors(data,i,hop)
        if len(neighbors) == 0:
            neighbor_dict[i] = []
        elif len(neighbors) <= k:
            neighbor_dict[i] = neighbors
        else:
            sim_score = []
            for j in neighbors:
                sim_score.append((j,1-cosine(g_feat[i],g_feat[j])))
            sorted_score = sorted(sim_score, key=lambda item: item[1], reverse=True)
            neighbor_dict[i] = [sorted_score[m][0] for m in range(k)]
    return neighbor_dict                       
                   
        


def get_one_hop_neighbors(data,sampled_node_idxs):
    neighbor_dict = {}
    for center_node_idx in sampled_node_idxs:
        #center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = set(neighbors_in_mask(data.edge_index,center_node_idx))
    return neighbor_dict
    
def get_one_hop_neighbors(data,sampled_node_idxs):
    neighbor_dict = {}
    for center_node_idx in sampled_node_idxs:
        #center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = set(neighbors_in_mask(data.edge_index,center_node_idx))
    return neighbor_dict
    
def neighbors_in_mask(edge_index, node_id):
    row, col = edge_index 
    match_idx = torch.where(row == node_id)[0]
    neigh_nodes = col[match_idx]
    return neigh_nodes.tolist()
    
def get_modul_mask(data,args):
    x, edge_index, y = data.x, data.edge_index, data.y
    #N, E = data.x.shape[0], int(data.edge_index.shape[1]/2)
    N = int(edge_index.max().item()) + 1
    edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])
    adj = SparseTensor(row=edge_index[0],col=edge_index[1], sparse_sizes=(N, N))
    adj.fill_value_(1.)
    batch = torch.LongTensor(list(range(N)))
    batch, adj_batch = get_sim(batch, adj, wt=args.wt, wl=args.wl)
    mask = get_mask(adj_batch)
    return mask
    
def top_k_samples(feature,n_cluster,cluster_centers,predict_labels,top_k=1):
    distances = pairwise_distances(feature, cluster_centers)
    if top_k == 1:
        top_confidence_samples = []
        for i in range(n_cluster):
            cluster_indices = np.where(predict_labels == i)[0]
            cluster_distances = distances[cluster_indices, i]

            # Get the indices of the top N samples with the smallest distances
            top_n_indices = cluster_indices[np.argmin(cluster_distances)]
            top_confidence_samples.append(top_n_indices)
    else:
        top_confidence_samples = {}
        for i in range(n_cluster):
            cluster_indices = np.where(predict_labels == i)[0]
            cluster_distances = distances[cluster_indices, i]

            # Get the indices of the top N samples with the smallest distances
            top_n_indices = cluster_indices[np.argsort(cluster_distances)[:top_k]]
            top_confidence_samples[i] = top_n_indices
    return top_confidence_samples
