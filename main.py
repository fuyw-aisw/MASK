import torch
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import os
from utils import set_seed_config
from network import Graph_embed
from data import get_sentence_embeddings
from metrics import true_map_cluster
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix
from utils import log, add_sim_edge
from torch_geometric.utils import to_dense_adj, to_undirected
import prompt
import warnings
warnings.filterwarnings("ignore")
#from sklearn.cluster import KMeans
import json
from datetime import datetime


def cosine_sim(X1, X2):
    X_norm1 = X1 / torch.norm(X1, p=2, dim=1, keepdim=True)
    X_norm2 = X2 / torch.norm(X2, p=2, dim=1, keepdim=True)    

    sim_matrix = torch.mm(X_norm1, X_norm2.T)
    
    return sim_matrix

def train(args,data,seeds):
    best_acc = 0
    model = Graph_embed(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr1, weight_decay=args.wd)
    N = data.x.shape[0]
    data.edge_index = to_undirected(data.edge_index) 
    data_obj = Data(x = data.x, edge_index = data.edge_index)
    data_obj = data_obj.to(args.device)
    n_clusters = data.y.max().item()+1
    y_true = data.y.numpy()
    
    #multi-view texts generation; correlation loss
    if not os.path.exists(f"jsons/{args.dataset}_raw_texts_multi_view.json"):
        data_mask = torch.load(f"preprocessed_data/{args.dataset}_sbert_imb.pt")
        node_idx = data_mask[10].node_map.numpy()
        raw_texts = data_mask[10].raw_texts
        prompts1, prompts2 = prompt.prompt_texts_view_aug(raw_texts=raw_texts, dataset_name=args.dataset)
        texts_view1, texts_view1_reason = prompt.efficient_gpt_text_ge(prompts1)
        texts_view2, texts_view2_reason = prompt.efficient_gpt_text_ge(prompts2)
        texts_multi_view = {str(idx): [texts_view1[i],texts_view2[i]] for i, idx in enumerate(node_idx)}
        with open(f"jsons/{args.dataset}_raw_texts_view1_reason.json", 'w', encoding='utf-8') as f:
            json.dump(texts_view1_reason, f, ensure_ascii=False, indent=4) 
        with open(f"jsons/{args.dataset}_raw_texts_view2_reason.json", 'w', encoding='utf-8') as f:
            json.dump(texts_view2_reason, f, ensure_ascii=False, indent=4)
        with open(f"jsons/{args.dataset}_raw_texts_multi_view.json", 'w', encoding='utf-8') as f:
            json.dump(texts_multi_view, f, ensure_ascii=False, indent=4)
    else:
        with open(f"jsons/{args.dataset}_raw_texts_multi_view.json", 'r', encoding='utf-8') as f:
            texts_multi_view = json.load(f)
    #generate multi-view texts at once, and then extract the corresponding texts varying imbalance ratio 

    texts_view1, texts_view2 = zip(*[(texts_multi_view[str(node)][0], texts_multi_view[str(node)][1]) for node in data.node_map.numpy()])
    texts_view1, texts_view2 = list(texts_view1), list(texts_view2)
    
    texts_view1_embed = get_sentence_embeddings(texts_view1, embed_type=args.embed_type, device=args.device).to(args.device)
    texts_view2_embed = get_sentence_embeddings(texts_view2, embed_type=args.embed_type, device=args.device).to(args.device)
    data_obj.x_view1, data_obj.x_view2 = texts_view1_embed, texts_view2_embed
    
    

    # pretrain over corr_loss + mixup_loss

    for epoch in range(args.pretrain_epochs):
        model.train()
        optimizer.zero_grad()

        if epoch % args.num_epoch == 0:
        
            # generate M synthesized representations; M:args.tag_num; contrastive loss to eliminate imbalance, contribution and confidence scores are learned by llm
            ### for better adjustment
            if not os.path.exists(f"jsons/{args.dataset}_raw_texts_scores_com_im_{args.im_ratio}_tag_{args.tag_num}_pretrain_{epoch}.json"):
                prompts_view1, prompts_view2, idx_grouped = prompt.prompt_texts_syn_sim(texts_view1, texts_view2, args.tag_num, args.dataset)
                idx_grouped_lens = [len(idx) for idx in idx_grouped]
                scores_view1, scores_view1_reason = prompt.efficient_gpt_text_score(prompts_view1, idx_grouped_lens)                
                with open(f"jsons/{args.dataset}_raw_texts_view1_scores_reason_im_{args.im_ratio}_tag_{args.tag_num}_pretrain_{epoch}.json", 'w', encoding='utf-8') as f:
                    json.dump(scores_view1_reason, f, ensure_ascii=False, indent=4)
                    
                scores_view2, scores_view2_reason = prompt.efficient_gpt_text_score(prompts_view2, idx_grouped_lens)
                with open(f"jsons/{args.dataset}_raw_texts_view2_scores_reason_im_{args.im_ratio}_tag_{args.tag_num}_pretrain_{epoch}.json", 'w', encoding='utf-8') as f:
                    json.dump(scores_view2_reason, f, ensure_ascii=False, indent=4) 
                with open(f"jsons/{args.dataset}_raw_texts_scores_com_im_{args.im_ratio}_tag_{args.tag_num}_pretrain_{epoch}.json", 'w', encoding='utf-8') as f:
                    json.dump([idx_grouped, scores_view1, scores_view2], f, ensure_ascii=False, indent=4)
            else:
                with open(f"jsons/{args.dataset}_raw_texts_scores_com_im_{args.im_ratio}_tag_{args.tag_num}_pretrain_{epoch}.json", 'r', encoding='utf-8') as f:
                    idx_grouped, scores_view1, scores_view2 = json.load(f)

            weight_view1 = torch.zeros([args.tag_num, N]).to(args.device)
            weight_view2 = torch.zeros([args.tag_num, N]).to(args.device)
            for i in range(args.tag_num):
                weight_view1[i, np.array(idx_grouped[i])] = torch.softmax((1-torch.tensor(scores_view1['cont_scores'][i])) * torch.tensor(scores_view1['conf_scores'][i]), dim = 0).to(args.device)
                weight_view2[i, np.array(idx_grouped[i])] = torch.softmax((1-torch.tensor(scores_view2['cont_scores'][i])) * torch.tensor(scores_view2['conf_scores'][i]), dim = 0).to(args.device)           
            

        g_feat1, g_feat2 = model(data_obj)
        #print(cosine_sim(g_feat_syn1, g_feat_syn2))
        g_feat_syn1 = F.normalize(torch.mm(weight_view1, g_feat1), dim=-1)
        g_feat_syn2 = F.normalize(torch.mm(weight_view2, g_feat2), dim=-1)

        #print(cosine_sim(g_feat1, g_feat2))
        #print(cosine_sim(g_feat_syn1, g_feat_syn2))
        cor_loss = model.corr_loss(g_feat1, g_feat2)
        mix_loss = model.mixup_loss(g_feat_syn1, g_feat_syn2, args.tau1)
        #print(g_feat_syn1.requires_grad)
        if args.dataset == "pubmed":
            loss = cor_loss 
        else:
            loss = args.alpha * cor_loss + (1 - args.alpha) * mix_loss
        print(f'[{datetime.now()}] (Pretrain) | Epoch={epoch+1}, loss={float(loss):.4f}, corr_loss={float(cor_loss):.4f}, mixup_loss={float(mix_loss):.4f}')
        
        loss.backward()            
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                g_feat = model(data_obj, False)
                ACC, CAA, ARI, NMI, F1 = [], [], [], [], []
                for seed in seeds:
                    y_pred, acc, caa, ari, nmi, f1 = model.cluster(g_feat, None, data.y.numpy(), seed, False)
                    ACC.append(acc)
                    CAA.append(caa)
                    ARI.append(ari)
                    NMI.append(nmi)
                    F1.append(f1)
                acc = np.mean(ACC)
                caa = np.mean(CAA)
                if best_acc < acc:
                    best_acc = acc
                    torch.save(model.state_dict(), f"models/best_model_{args.dataset}_im_{args.im_ratio}_tag_{args.tag_num}_pretrain.pt")
                print(f'[{datetime.now()}] Pretrain mean | Epoch={epoch+1:03d}, ACC={np.mean(ACC):.4f}, CAA={np.mean(CAA):.4f}, ARI={np.mean(ARI):.4f}, NMI={np.mean(NMI):.4f}, F1={np.mean(F1):.4f}')
                print(f'[{datetime.now()}] Pretrain std | Epoch={epoch+1:03d}, ACC={np.std(ACC):.4f}, CAA={np.std(CAA):.4f}, ARI={np.std(ARI):.4f}, NMI={np.std(NMI):.4f}, F1={np.std(F1):.4f}')
                

    #finetuning
    #breakpoint()

    best_acc=0
    model.load_state_dict(torch.load(f"models/best_model_{args.dataset}_im_{args.im_ratio}_tag_{args.tag_num}_pretrain.pt"))
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr2, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    model.eval()
    with torch.no_grad():
        g_feat1, g_feat2 = model(data_obj)
        g_feat = model(data_obj, False)
        mis_mask = model.cluster(g_feat1, g_feat2, None, seeds[0])
        y_assignments, _, _, _, _, _= model.cluster(g_feat, None, data.y.numpy(), seeds[0], False)
    labels_llm_ge = {}
    uncertain_mask = mis_mask.copy()
    g_feat_confi = torch.stack([g_feat[y_assignments == i].detach().mean(axis=0) for i in range(args.num_classes)]).to(args.device)
    confi_mask = {}
    
    for i in range(args.num_classes):
        idx_in_cluster = np.nonzero(y_assignments == i)[0]
        dist = torch.norm(g_feat[y_assignments == i] - g_feat_confi[i], p=2, dim=1) 
        num_top = min(len(dist), args.top)
        print(len(dist))
        _, idx = torch.topk(dist, num_top, largest=False)
        confi_mask[i] = idx_in_cluster[idx.cpu().numpy()].tolist()
    mapping = true_map_cluster(y_true,y_assignments) 
    print(len(uncertain_mask))
    print(mapping)
    print('----')
    print(confi_mask)
    ###reuse

    if not os.path.exists(f'jsons/{args.dataset}_labels_im_{args.im_ratio}_tag_{args.tag_num}_we_{args.alpha}_tau_{args.tau1}.json'):
        prompts_topics = prompt.prompt_cluster_sum(data_obj=data, confi_mask=confi_mask, dataset_name=args.dataset)
        topics_generate, topics_reason = prompt.efficient_gpt_text_ind(prompts_topics)
        print(topics_generate)

        with open(f'jsons/{args.dataset}_topics_im_{args.im_ratio}_tag_{args.tag_num}_we_{args.alpha}_tau_{args.tau1}.json', 'w', encoding='utf-8') as f:
            json.dump(topics_generate, f, ensure_ascii=False, indent=4)
        with open(f'jsons/{args.dataset}_topics_reason_im_{args.im_ratio}_tag_{args.tag_num}_we_{args.alpha}_tau_{args.tau1}.json', 'w', encoding='utf-8') as f:
            json.dump(topics_reason, f, ensure_ascii=False, indent=4)

        prompts_classify1, prompts_classify2 = prompt.prompt_aug_classifier(texts_view1=texts_view1, texts_view2=texts_view2, uncertain_mask=uncertain_mask, topic_clusters=topics_generate, dataset_name=args.dataset)
        label_classify1, label1_reason = prompt.efficient_gpt_text_cls(prompts_classify1, n_clusters)
        label_classify2, label2_reason = prompt.efficient_gpt_text_cls(prompts_classify2, n_clusters)

        with open(f'jsons/{args.dataset}_label1_reason_im_{args.im_ratio}_tag_{args.tag_num}_we_{args.alpha}_tau_{args.tau1}.json', 'w', encoding='utf-8') as f:
            json.dump(label1_reason, f, ensure_ascii=False, indent=4)
        
        with open(f'jsons/{args.dataset}_label2_reason_im_{args.im_ratio}_tag_{args.tag_num}_we_{args.alpha}_tau_{args.tau1}.json', 'w', encoding='utf-8') as f:
            json.dump(label2_reason, f, ensure_ascii=False, indent=4) 

        for idx in range(len(uncertain_mask)):
            labels_llm_ge[int(uncertain_mask[idx])] = [int(label_classify1[idx]),int(label_classify2[idx])] # first label ; second consistence
        with open(f'jsons/{args.dataset}_labels_im_{args.im_ratio}_tag_{args.tag_num}_we_{args.alpha}_tau_{args.tau1}.json', 'w', encoding='utf-8') as f:
            json.dump(labels_llm_ge, f, ensure_ascii=False, indent=4)
    else:
        with open(f'jsons/{args.dataset}_labels_im_{args.im_ratio}_tag_{args.tag_num}_we_{args.alpha}_tau_{args.tau1}.json', 'r', encoding='utf-8') as f:
            labels_llm_ge = json.load(f)
        label_classify1, label_classify2 = np.array(list(labels_llm_ge.values()))[:,0], np.array(list(labels_llm_ge.values()))[:,1]
        uncertain_mask = np.array(list([int(key) for key in labels_llm_ge.keys()]))

    ground_truth = np.array([mapping[int(label)] for label in data.y[uncertain_mask]])
    llm_idx = np.array([i for i in range(len(label_classify1)) if label_classify1[i] == label_classify2[i]])
    llm_labels = np.array([label_classify1[i] for i in range(len(label_classify1)) if label_classify1[i] == label_classify2[i]])
    acc_gen = np.array(label_classify1==ground_truth).mean()
    print(uncertain_mask)
    print(len(llm_labels), len(uncertain_mask), len(llm_labels)/len(uncertain_mask), acc_gen)
    print('----')
    
    for idx in llm_idx:
        data_obj.x_view1[uncertain_mask[idx]] = (data_obj.x_view1[uncertain_mask[idx]] + data_obj.x_view2[uncertain_mask[idx]])/2
        #texts_view1[uncertain_mask[idx]] += texts_view2[uncertain_mask[idx]]
       
    # finetune over corr_loss + mixup_loss + rank_loss 
    for epoch in range(args.finetune_epochs):
        model.train()
        optimizer.zero_grad()
        
        if epoch % args.num_epoch == 0:
            # generate M synthesized representations; M:args.tag_num; contrastive loss to eliminate imbalance
            if not os.path.exists(f"jsons/{args.dataset}_raw_texts_scores_com_im_{args.im_ratio}_tag_{args.tag_num}_finetune_{epoch}.json"):
                prompts_view1, prompts_view2, idx_grouped = prompt.prompt_texts_syn_sim(texts_view1, texts_view2, args.tag_num, args.dataset)
                idx_grouped_lens = [len(idx) for idx in idx_grouped]
                scores_view1, scores_view1_reason = prompt.efficient_gpt_text_score(prompts_view1, idx_grouped_lens)                
                with open(f"jsons/{args.dataset}_raw_texts_view1_scores_reason_im_{args.im_ratio}_tag_{args.tag_num}_finetune_{epoch}.json", 'w', encoding='utf-8') as f:
                    json.dump(scores_view1_reason, f, ensure_ascii=False, indent=4) 
                
                scores_view2, scores_view2_reason = prompt.efficient_gpt_text_score(prompts_view2, idx_grouped_lens)
                with open(f"jsons/{args.dataset}_raw_texts_view2_scores_reason_im_{args.im_ratio}_tag_{args.tag_num}_finetune_{epoch}.json", 'w', encoding='utf-8') as f:
                    json.dump(scores_view2_reason, f, ensure_ascii=False, indent=4) 
                with open(f"jsons/{args.dataset}_raw_texts_scores_com_im_{args.im_ratio}_tag_{args.tag_num}_finetune_{epoch}.json", 'w', encoding='utf-8') as f:
                    json.dump([idx_grouped, scores_view1, scores_view2], f, ensure_ascii=False, indent=4)
            else:
                with open(f"jsons/{args.dataset}_raw_texts_scores_com_im_{args.im_ratio}_tag_{args.tag_num}_finetune_{epoch}.json", 'r', encoding='utf-8') as f:
                    idx_grouped, scores_view1, scores_view2 = json.load(f)


            weight_view1 = torch.zeros([args.tag_num, N]).to(args.device)
            weight_view2 = torch.zeros([args.tag_num, N]).to(args.device)
            for i in range(args.tag_num):
                weight_view1[i, np.array(idx_grouped[i])] = torch.softmax((1-torch.tensor(scores_view1['cont_scores'][i])) * torch.tensor(scores_view1['conf_scores'][i]), dim = 0).to(args.device)
                weight_view2[i, np.array(idx_grouped[i])] = torch.softmax((1-torch.tensor(scores_view2['cont_scores'][i])) * torch.tensor(scores_view2['conf_scores'][i]), dim = 0).to(args.device)   
        g_feat1, g_feat2 = model(data_obj)
        g_feat_syn1 = torch.nn.functional.normalize(torch.mm(weight_view1, g_feat1), dim=-1)
        g_feat_syn2 = torch.nn.functional.normalize(torch.mm(weight_view2, g_feat2), dim=-1)
 
        assert g_feat1[uncertain_mask[llm_idx]].shape[0] == len(llm_labels)
        #with torch.no_grad():
        g_feat = model(data_obj, False)
        
        y_assignments, _, _, _, _, _ = model.cluster(g_feat1, None, data.y.numpy(), seeds[0], False) 
        _, counts = np.unique(y_assignments, return_counts=True)
        if len(counts) == args.num_classes:
            g_feat_confi = torch.stack([g_feat1[y_assignments == i].detach().mean(axis=0) for i in range(args.num_classes)]).to(args.device)
            
        cor_loss = model.corr_loss(g_feat1, g_feat2)
        mix_loss = model.mixup_loss(g_feat_syn1, g_feat_syn2, args.tau1)
        rank_loss = model.cl_loss(g_feat[uncertain_mask[llm_idx]], g_feat_confi, torch.tensor(llm_labels, dtype=torch.long).to(args.device), args.tau2)

        if args.dataset == "wikics":
            loss = args.beta * mix_loss + (1 - args.beta) * rank_loss
        else:
            loss = args.alpha * cor_loss + (1 - args.alpha) * (args.beta * mix_loss + (1 - args.beta) * rank_loss)
            
        
        print(f'[{datetime.now()}] (Finetune) | Epoch={epoch+1}, loss={float(loss):.4f}, corr_loss={float(cor_loss):.4f}, mixup_loss={float(mix_loss):.4f}, rank_loss={float(rank_loss):.4f}')
        
        loss.backward()
        optimizer.step() 
        scheduler.step()
        
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                g_feat = model(data_obj, False)
                ACC, CAA, NMI, ARI, F1 = [], [], [], [], []
                for seed in seeds:
                    y_pred, acc, caa, ari, nmi, f1 = model.cluster(g_feat, None, data.y.numpy(), seed, False)
                    ACC.append(acc)
                    CAA.append(caa)
                    ARI.append(ari)
                    NMI.append(nmi)
                    F1.append(f1)
                acc = np.mean(ACC)
                if best_acc < acc:
                    best_acc = acc
                    torch.save(model.state_dict(), f"models/best_model_{args.dataset}_im_{args.im_ratio}_tag_{args.tag_num}_finetune.pt")
                print(f'[{datetime.now()}] Finetune mean | Epoch={epoch+1:03d}, ACC={np.mean(ACC):.4f}, CAA={np.mean(CAA):.4f}, ARI={np.mean(ARI):.4f}, NMI={np.mean(NMI):.4f}, F1={np.mean(F1):.4f}')
                print(f'[{datetime.now()}] Finetune std | Epoch={epoch+1:03d}, ACC={np.std(ACC):.4f}, CAA={np.std(CAA):.4f}, ARI={np.std(ARI):.4f}, NMI={np.std(NMI):.4f}, F1={np.std(F1):.4f}')
   

            
    #testing final


    
    model.load_state_dict(torch.load(f"models/best_model_{args.dataset}_im_{args.im_ratio}_tag_{args.tag_num}_finetune.pt"))
    model.eval()
    with torch.no_grad():
        g_feat = model(data_obj, False)
        ACC, CAA, NMI, ARI, F1 = [], [], [], [], []
        for seed in [seeds[0]]:
            y_pred, acc, caa, ari, nmi, f1 = model.cluster(g_feat, None, data.y.numpy(), seed, False)
            torch.save(y_pred,f"results/{args.dataset}_pred_score.pt")
            ACC.append(acc)
            CAA.append(caa)
            ARI.append(ari)
            NMI.append(nmi)
            F1.append(f1)  
        print(f'[{datetime.now()}] Final evaluation mean | ACC={np.mean(ACC):.4f}, CAA={np.mean(CAA):.4f}, ARI={np.mean(ARI):.4f}, NMI={np.mean(NMI):.4f}, F1={np.mean(F1):.4f}')
        print(f'[{datetime.now()}] Final evaluation std |  ACC={np.std(ACC):.4f}, CAA={np.std(CAA):.4f}, ARI={np.std(ARI):.4f}, NMI={np.std(NMI):.4f}, F1={np.std(F1):.4f}')


                
            
                

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Graph clustering enhanced by LLM")
    parser.add_argument('--dataset',default='cora',type=str)
    parser.add_argument('--embed_type',default='sbert',type=str)
    parser.add_argument('--pretrain_epochs', type=int, default=200) 
    parser.add_argument('--finetune_epochs', type=int, default=200)
    parser.add_argument('--seed_num', type=int, default=5)
    parser.add_argument('--optim',type=str, default='adam')
    parser.add_argument('--alpha',help='hyper-parameter to control the weights',type=float, default=0.1)
    parser.add_argument('--beta',help='hyper-parameter to control the weights',type=float, default=0.9)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--top', type=int, default=50, help='hyper-parameter for the number of top high confidence samples')
    parser.add_argument('--tau1', type=float, default=0.5, help='temperature for cl1')
    parser.add_argument('--tau2', type=float, default=0.01, help='temperature for cl2')
    parser.add_argument('--hidden', type=str, default='64', help='GNN encoder')
    parser.add_argument('--projection', type=str, default='', help='Projection')

    # learning para
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--lr1', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr2', type=float, default=0.0001, help='learning rate')    
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--ns', type=float, default=0.5, help='') 
    
    parser.add_argument('--im_ratio', type=int, default=10, help='')    
    
    parser.add_argument('--tag_num', type=int, default=100, help='')
    parser.add_argument('--num_epoch', type=int, default=100, help='')
    
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}"
    data = torch.load(f"preprocessed_data/{args.dataset}_{args.embed_type}_imb.pt")[args.im_ratio]
    args.input_dim = data.x.shape[1]
    args.num_classes = data.y.max().item()+1
    set_seed_config(42)
    seeds = [random.randint(1, 10000) for _ in range(args.seed_num)]
    print(seeds)
    print(args)
    train(args,data,seeds)
        
      