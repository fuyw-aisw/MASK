import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GCNConv
import torch.nn.functional as F
from utils import scale, clustering, mean_var, smooth_kmeans
from loss import correlation_multi_view, nt_xent, MultipleNegativesRankingLoss
from metrics import true_map_cluster
import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels, base_model=SAGEConv, dropout: float = 0.5, ns: float = 0.5):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.dropout = dropout
        self.k = len(hidden_channels)
        self.ns = ns

        self.convs = nn.ModuleList()
        self.convs.extend([base_model(in_channels, hidden_channels[0])])

        for i in range(1, self.k):
            self.convs.extend(
                [base_model(hidden_channels[i-1], hidden_channels[i])])

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.k):
            self.convs[i].reset_parameters()

    def forward(self, x: torch.Tensor, edge_index=None, adjs=None, dropout=True):
        if not adjs:
            for i in range(self.k):
                if dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.convs[i](x, edge_index)
                x = F.leaky_relu(x, self.ns)
        else:
            for i, (edge_index, _, size) in enumerate(adjs):
                if dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                x = F.leaky_relu(x, self.ns)
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, in_channels: int, project_hidden, activation=nn.PReLU):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.in_channels = in_channels
        self.project_hidden = project_hidden
        self.activation = activation

        self.project = None
        if self.project_hidden is not None:
            self.project = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.project.extend(
                [nn.Linear(self.in_channels, self.project_hidden[0])])
            self.activations.extend([nn.PReLU(project_hidden[0])])
            for i in range(1, len(self.project_hidden)):
                self.project.extend(
                    [nn.Linear(self.project_hidden[i-1], self.project_hidden[i])])
                self.activations.extend([nn.PReLU(project_hidden[i])])

    def forward(self, x: torch.Tensor, edge_index=None, adjs=None) -> torch.Tensor:
        x = self.encoder(x, edge_index, adjs)
        if self.project is not None:
            for i in range(len(self.project_hidden)):
                x = self.project[i](x)
                x = self.activations[i](x)
        return x

    
class Graph_embed(torch.nn.Module):
    def __init__(self, args, num_features=384):
        super(Graph_embed, self).__init__()
        hidden = list(map(int, args.hidden.split(',')))
        self.encoder = Encoder(num_features, hidden, base_model=GCNConv,dropout=args.dropout, ns=args.ns)
        self.model = Model(self.encoder, in_channels=hidden[-1], project_hidden=None)
        self.cluster_size = args.num_classes

    def forward(self, data, aug=True):
        if aug:
            x_view1, x_view2, edge_index = data.x_view1, data.x_view2, data.edge_index
            #multi-view texts
            g_feat1 = self.model(x_view1, edge_index) 
            g_feat2 = self.model(x_view2, edge_index)
            g_feat1 = F.normalize(g_feat1, p=2, dim=1)
            g_feat2 = F.normalize(g_feat2, p=2, dim=1)
            return g_feat1, g_feat2
        else:
            x, edge_index = data.x, data.edge_index
            g_feat = self.model(x, edge_index)
            g_feat = F.normalize(g_feat, p=2, dim=1)
            return g_feat
    
    def cluster(self, g_feat1, g_feat2=None, y_true=None, seed=1, aug=True):
        if aug:
            y_pred1 = clustering(g_feat1.detach().cpu().numpy(), self.cluster_size, None, "cuda", g_feat1.device, seed, smooth=True)
            y_pred2 = clustering(g_feat2.detach().cpu().numpy(), self.cluster_size, None, "cuda", g_feat2.device, seed, smooth=True)
            mis_mask = self.find_mismatch_nodes(y_pred1,y_pred2)

            return mis_mask
        else:
            y_pred, acc, caa, ari, nmi, f1 = clustering(g_feat1.detach().cpu().numpy(), self.cluster_size, y_true, "cuda", g_feat1.device, seed, smooth=True)
            return y_pred, acc, caa, ari, nmi, f1
            

    def corr_loss(self, g_feat1, g_feat2):
        loss = correlation_multi_view(g_feat1, g_feat2)
        return loss
    
    def mixup_loss(self, g_feat1, g_feat2, tau1):
        loss = nt_xent(g_feat1, g_feat2, tau1)
        return loss
       
    
    def cl_loss(self, g_feat, p_feat, labels, tau2):
        criterion = MultipleNegativesRankingLoss(scale=tau2)
        loss = criterion(g_feat,p_feat,labels)
        return loss
    
    '''
    def cluster_loss(self, g_feat1, g_feat2, seed):
        _, cluster_center1, bol_weights1 = smooth_kmeans(X=g_feat1, num_clusters=self.cluster_size, device=g_feat1.device, seed=seed)
        _, cluster_center2, bol_weights2 = smooth_kmeans(X=g_feat2, num_clusters=self.cluster_size, device=g_feat1.device, seed=seed)
        loss = torch.sum((g_feat1 - torch.matmul(bol_weights1.to(g_feat1.device), cluster_center1.to(g_feat1.device))) ** 2) + torch.sum((g_feat2 - torch.matmul(bol_weights2.to(g_feat2.device), cluster_center2.to(g_feat2.device))) ** 2)
        return loss
    '''
    
    @staticmethod
    def find_mismatch_nodes(y_pred1,y_pred2):
        #y_pred_label1 = torch.argmax(y_pred1,dim=1)
        #y_pred_label2 = torch.argmax(y_pred2,dim=1)
        #print(y_pred_label1)
        mapping = true_map_cluster(y_pred2,y_pred1) #y_true,y_pred mapping cluster
        y_pred_label2 = [mapping[int(label)] for label in y_pred2]
        y_pred_label1 = y_pred1
        mis_mask = y_pred_label1 != y_pred_label2
        mis_mask = np.nonzero(mis_mask)[0]

        return mis_mask 