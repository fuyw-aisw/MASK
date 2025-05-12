import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch_geometric.utils import index_to_mask
from utils import set_seed_config
from tqdm import tqdm

#embeddings: data.x
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def batched_data(inputs, batch_size):
    return [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]

def get_sentence_embeddings(texts,embed_type="sbert",device="cuda:0",batch_size=64):
    if embed_type == "sbert":
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2').to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        model = AutoModel.from_pretrained("microsoft/deberta-base").to(device)
    output = []
    with torch.no_grad():
        for batch in tqdm(batched_data(texts, batch_size)):
            batch_dict = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**batch_dict)
            embeddings = mean_pooling(outputs, batch_dict['attention_mask'])
            output.append(embeddings.cpu())
            del batch_dict
    output = torch.cat(output, dim = 0) 
    return output


if __name__ == '__main__':
    dataset = ['cora','pubmed','citeseer','wikics']#,'arxiv','products']
    embedding = ['sbert']
    for name in dataset:
        for embed_type in embedding:
            print(name,embed_type,"prepare")
            data = torch.load(f"Graph-LLM-master/preprocessed_data/new/{name}_fixed_sbert.pt")
            data.x = get_sentence_embeddings(data.raw_texts,embed_type=embed_type)
            del data.train_masks
            del data.val_masks
            del data.test_masks
            
            torch.save(data,f"./preprocessed_data/{name}_{embed_type}.pt")
