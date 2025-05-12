# TRACI
PyTorch implementation for "Text-guided Group Mixup with Canonical Mining for Imbalanced Graph Clustering"

# Overview

TRACI is a novel framework for imbalanced text-attributed graph clustering, which leverages large language models to generate balanced, mixed groups with an emphasis on minority classes.

# Installation

Start by following this source codes:
```bash
git clone https://github.com/fuyw-aisw/MARK.git
cd MARK
pip -r requirements.txt
## or install the following dependencies
## step1: install PyTorch’s CUDA support on Linux
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
## step2: install pyg package
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.0.0%2Bcu118.html ### GPU
```
# Reproduction

* Cora
```
python main.py --dataset cora --im_ratio 10 --tag_num 100 --hidden '64' --lr1 0.0005 --tau1 0.5
```

* CiteSeer
```
python main.py --dataset citeseer --im_ratio 10 --tag_num 100 --hidden '128,64' --lr1 0.0001 --tau1 0.9 --alpha 0.2
```
* WikiCS
```
python main.py --dataset wikics --im_ratio 10 --tag_num 500 --hidden '256,128' --lr1 0.0001 --tau1 0.9 --tau2 0.09
```
* PubMed
```
python main.py --dataset pubmed --im_ratio 10 --tag_num 500 --hidden '128,64' --lr1 0.0005 --tau1 0.5
```

