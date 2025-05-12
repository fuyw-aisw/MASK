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
from sklearn.metrics.pairwise import pairwise_distances

def inv_sqrt(mat, eps):
    eig_vals, eig_vecs = torch.linalg.eigh(mat)
    return eig_vecs @ (eig_vals.clamp(min=eps).rsqrt().diag_embed()) @ eig_vecs.mT


def correlation_multi_view(x1: torch.Tensor, x2: torch.Tensor, eps=1e-5): #x1:N*d_1; x2:N*d_2
    x1_centered = x1 - x1.mean(dim=0, keepdim=True)
    x2_centered = x2 - x2.mean(dim=0, keepdim=True)
    N = x1.size(0)
    C_12 = x1_centered.T @ x2_centered / (N - 1) #d_1*d_2
    C_11 = x1_centered.T @ x1_centered / (N - 1) + eps * torch.eye(x1.size(1), device=x1.device) #d_1*d_1
    C_22 = x2_centered.T @ x2_centered / (N - 1) + eps * torch.eye(x2.size(1), device=x2.device)#d_2*d_2
    
    C_22_inv_sqrt = inv_sqrt(C_22, eps) #d_2*d_2
    C_11_inv = torch.linalg.inv(C_11)
    
    loss = -torch.trace(C_22_inv_sqrt @ C_12.T @ C_11_inv @ C_12 @ C_22_inv_sqrt)
    
    return loss
       

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def nt_xent(x1, x2=None, t=0.2):

    if x2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x1.shape[0]
        out_1 = F.normalize(x1, dim=-1)
        out_2 = F.normalize(x2, dim=-1)
        # out_1 = x
        # out_2 = features2

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    # print("temperature is {}".format(t))
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    mask = get_negative_mask(batch_size).to(x1.device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    Ng = neg.sum(dim=-1)


    loss = (- torch.log(pos / (pos + Ng)))

    return loss.mean()

class MultipleNegativesRankingLoss(torch.nn.Module):

    def __init__(self, scale: float = 0.05):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def cos_sim(self,a: torch.Tensor, b: torch.Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def forward(self, embeddings_a, embeddings_b, labels):

        scores = self.cos_sim(embeddings_a, embeddings_b) / self.scale
        
        return self.cross_entropy_loss(scores, labels)