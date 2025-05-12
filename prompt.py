import torch
import random
from utils import get_one_hop_neighbors, get_top_k_neighbor_simcse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import torch
from conversation import conv_v1, conv_v2, conv_v3, conv_v4
import utils
import json
from call_api import call_api
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import re

def add_output_requirements(dataset_name):
    if dataset_name == "wikics":
        prompt = "\n The newly generated article should be no more than 350 words. Output the newly generated article as a dictionary, with key \"answer\" and \"reason\"."
    else:
        prompt = "\n The newly generated article should have a title of no more than 15 words and an abstract limited to 300 words. Output the newly generated article as a dictionary, with key \"answer\" and \"reason\"."
    return prompt

def prompt_texts_view_aug(raw_texts, dataset_name="cora"):
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    prompts1, prompts2 = [], []
    for text in raw_texts:
        #formal
        prompt1 = f"Please rigorously rewrite the following {typ} article while preserving its core ideas:\n\n{text}\n\n"
        prompt1 += "Please use formal academic language with domain-specific terminology, while maintaining strict factual consistency with original content when rewriting the article."
        prompt1 += add_output_requirements(dataset_name)
        prompts1.append(prompt1)
        
        #easy to understand
        prompt2 = f"Please simplify this {typ} article for non-experts while retaining key information:\n\n{text}\n\n"
        prompt2 += "Please avoid technical jargon and use short sentences and everyday vocabulary when rewriting the article."
        prompt2 += add_output_requirements(dataset_name)
        prompts2.append(prompt2)
    return prompts1, prompts2



def prompt_texts_syn(texts_view1, texts_view2, tag_num, dataset_name="cora"):
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    N = len(texts_view1)
    random_labels = torch.randint(low=0, high=tag_num, size=(N,), dtype=torch.int64)
    texts_view1_grouped = [[] for _ in range(tag_num)]
    texts_view2_grouped = [[] for _ in range(tag_num)]
    for idx, (text1, text2, label) in enumerate(zip(texts_view1, texts_view2, random_labels)):
        label_idx = label.item()
        if 0 <= label_idx < tag_num:
            texts_view1_grouped[label_idx].append(text1)
            texts_view2_grouped[label_idx].append(text2)
        else:
            raise ValueError("Invalid label index")
    prompts1, prompts2 = [], []
    for idx in range(tag_num):
        prompt_a = f"Below are the important samples related to {typ}. Please comprehensively mixup texts in each cluster and then synthesize a novel text considering the commonalities and core content of these samples together."
        prompt_a += "\n The significant samples are stated in the following." + "{\n"
        
        prompt1 = prompt_a
        prompt1 += "\n".join(texts_view1_grouped[idx]) + "}\n"
        prompt1 += add_output_requirements(dataset_name)
        prompts1.append(prompt1)
        
        prompt2 = prompt_a
        prompt2 += "\n".join(texts_view2_grouped[idx]) + "}\n"
        prompt2 += add_output_requirements(dataset_name)
        prompts2.append(prompt2)
    return prompts1, prompts2            

def prompt_texts_syn_sim(texts_view1, texts_view2, tag_num, dataset_name="cora"):
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    N = len(texts_view1)
    random_labels = torch.randint(low=0, high=tag_num, size=(N,), dtype=torch.int64)
    texts_view1_grouped = [[] for _ in range(tag_num)]
    texts_view2_grouped = [[] for _ in range(tag_num)]
    idx_view_grouped = [[] for _ in range(tag_num)]
    
    for idx, (text1, text2, label) in enumerate(zip(texts_view1, texts_view2, random_labels)):
        label_idx = label.item()
        if 0 <= label_idx < tag_num:
            
            texts_view1_grouped[label_idx].append(text1)
            texts_view2_grouped[label_idx].append(text2)
            idx_view_grouped[label_idx].append(idx)
        else:
            raise ValueError("Invalid label index")
    prompts1, prompts2 = [], []
    for idx in range(tag_num):
        prompt_a = f"Below are a cluster of texts related to {typ}. Please evaluate the contribution and confidence scores of each text in this cluster."
        prompt_a += "\n The contribution score should range from 0.00 to 1.00. A lowest score of 0.00 indicates the lowest contributio while 1.00 reflects the highest contribution. When assessing the contribution score of a text, several aspects need to be taken into account seperately. These include its semantic relevance to the cluster, the density and diversity of information it contains, its conceptual representativeness, and its contextual coherence with other texts. After evaluating each aspect independently, an overall contribution score is then derived by synthesizing these individual evaluations."
        prompt_a += "\n After determining a contribution score, please also asign a confidence score to it. This confidence score should also fall within the range of 0.00 to 1.00. It serves to evaluate the accuracy and credibility of the contribution score."
        prompt_a += "\n Texts in the cluster are stated in the following.\n"   
        
        prompt1 = prompt_a + "{\n"
        for i, text in enumerate(texts_view1_grouped[idx]):
            prompt1 += f"Text {i+1}: {text} \n"
        prompt1 += "\n}"
        prompt1 += "\n Output the dictionary with key \"contribution_scores\" and \"confidence_scores\" in JSON format: {\"contribution_scores\": [{\"text_id\": 1, \"score\": 0.XX, \"reason\": XX},...], \"confidence_scores\": [{\"text_id\": 1, \"score\": 0.XX, \"reason\": XX},...]}. The reason should be no more than 30 words."
        prompts1.append(prompt1)
        
        prompt2 = prompt_a + "{\n"
        for i, text in enumerate(texts_view2_grouped[idx]):
            prompt2 += f"Text {i+1}: {text} \n"
        prompt2 += "\n}"
        prompt2 += "\n Output the dictionary with key \"contribution_scores\" and \"confidence_scores\" in JSON format : {\"contribution_scores\": [{\"text_id\": 1, \"score\": 0.XX, \"reason\": XX},...], \"confidence_scores\": [{\"text_id\": 1, \"score\": 0.XX, \"reason\": XX},...]}. The reason should be no more than 30 words."
        prompts2.append(prompt2)
        
    return prompts1, prompts2, idx_view_grouped
    
def prompt_cluster_sum(data_obj,confi_mask,dataset_name="cora",memory_limit = 10000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
    else:
        raw_texts = data_obj.raw_texts
    prompts = []
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    for key in confi_mask.keys():
        prompt = f"Below are the high-confidence samples related to {typ} from a specific cluster. Analyze the commonalities and core content of these samples and provide a concise summary of the cluster's theme. Output the theme as a short name without adding any extra explanations."
        prompt += f"\n The cluster has the following high confidence samples:\n"
        prompt += f"High confidence sample in cluster{int(key)+1}:"+" {{\n"

        for idx in confi_mask[key]:
            prompt += f"{raw_texts[idx]}"+"}}\n"
        prompt += "Please comprehensively consider the commonalities between these samples and then conclude the topic of the cluster concisely. Output the newly generated topic name as a dictionary, with key \"answer\" and \"reason\"."
        prompts.append(prompt)
    return prompts
    

def prompt_neighbor_generate(data_obj,sampled_node_idxs,g_feat,topic_clusters,hop=2,sample_num=2,dataset_name="cora",memory_limit = 10000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
    else:
        raw_texts = data_obj.raw_texts
    prompts = []    
    neighbor_dict = get_top_k_neighbor_simcse(data_obj,sampled_node_idxs,g_feat,sample_num,hop)
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    
    for idx,sam_idx in enumerate(sampled_node_idxs):           
        #form = infer_text_form(raw_texts[sam_idx])    
        prompt = f"Given a target article related to {typ}: \n {raw_texts[sam_idx]}."
        prompt += f"\n The topic of this article may fall under one of the following clusters: {topic_clusters}."
        if len(neighbor_dict[sam_idx])>0:
            prompt += f"\n It has following important neighbors which has citation relationship to this {typ}, from most related to least related:\n"
            prompt += f"Neighbors of {typ} {idx}:"+" {\n"
            for nei_idx,nei in enumerate(neighbor_dict[sam_idx]):
                prompt += f"{raw_texts[nei]}"+"}\n"                
                
            if dataset_name == "wikics":
                prompt += f"Please consider the information from the target article and its neighbors, and generate a novel article similar to the given one, with no more than 350 words. Output the newly generated article as a dictionary, with key \"answer\" and \"reason\"."
            else:
                prompt += f"Please consider the information from the target article and its neighbors, and generate a novel article similar to the given one, with a title of no more than 15 words and an abstract limited to 300 words.Output the newly generated article as a dictionary, with key \"answer\" and \"reason\"."                
        else:
            if dataset_name == "wikics":
                prompt += f"Please consider the information from the target article, and generate a novel article similar to the given one, with no more than 350 words. Output the newly generated article as a dictionary, with key \"answer\" and \"reason\"." 
            else:
                prompt += f"Please consider the information from the target article, and generate a novel article similar to the given one, with a title of no more than 15 words and an abstract limited to 300 words. Output the newly generated article as a dictionary, with key \"answer\" and \"reason\"."                                 
        prompts.append(prompt)
    return prompts


def prompt_aug_classifier(texts_view1, texts_view2, uncertain_mask, topic_clusters, dataset_name="cora"):
    
    prompts1, prompts2 = [], []
    n_clusters = len(topic_clusters)
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    #anchor nodes
    for idx in uncertain_mask:
        text = texts_view1[idx]
        prompt = f"Given a target article related to {typ}: \n {text}."
        prompt += f"\n Please determine which cluster this {typ} most likely belongs to only from {n_clusters} clusters.\n These optional clusters are: "
        contexts = []
        for num_cluster in range(n_clusters):
            this_context = {f"Cluster {num_cluster+1}": topic_clusters[num_cluster]}
            contexts.append(this_context)
        prompt += str(contexts)
        prompt += f"\n Please comprehensively consider which cluster this article most likely belongs to, only answer the cluster number directly as a dictionary, with key \"answer\" and \"reason\"."
        prompt += f"\n The cluster number should be a interger and range from 1 to {n_clusters}."
        prompts1.append(prompt)

    #augmentation nodes    
    for idx in uncertain_mask:
        text = texts_view2[idx]
        prompt = f"Given a target article related to {typ}: \n {text}."
        prompt += f"\n Please determine which cluster this article most likely belongs to only from {n_clusters} clusters.\n These optional clusters are: "
        contexts = []
        for num_cluster in range(n_clusters):
            this_context = {f"Cluster {num_cluster+1}": topic_clusters[num_cluster]}
            contexts.append(this_context)
        prompt += str(contexts)
        prompt += f"\n Please comprehensively consider which cluster this article most likely belongs to, only answer the cluster number directly as a dictionary, with key \"answer\" and \"reason\"."
        prompt += f"\n The cluster number should be a interger and range from 1 to {n_clusters}."
        prompts2.append(prompt)
        
    return prompts1,prompts2

    #return prompts1
        
def prompt_aug_classifier_equ(data_obj,sampled_node_idxs,aug_node_texts,topic_clusters,dataset_name="cora",memory_limit = 1000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
    else:
        raw_texts = data_obj.raw_texts
    prompts = []
    n_clusters = data_obj.y.max().item()+1
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    for idx,sam_idx in enumerate(sampled_node_idxs):
        prompt = f"Given two target articles related to {typ}.\n The first article is:{raw_texts[sam_idx]}.\n The second article is {aug_node_texts[idx]}."
        prompt += f"\n The topic of these articles may fall under one of the following clusters: {topic_clusters}."
        prompt += f"\n Please comprehensively consider the two articles above and the cluster they are most likely to belong to separately, and then determine whether they belong to the same cluster. Output 1 if they belong to the same cluster, or 0 if they belong to different clusters. Please answer as a dictionary, with key \"answer\" and \"reason\"."
        prompts.append(prompt)
    return prompts    


def efficient_gpt_text_ind(input_text):    
    gpt_result = []

    prompts = []
    for idx, text in enumerate(input_text):
        conv = conv_v1.copy()
        conv.append_message(conv.roles[0], text)
        final_prompt = conv.get_prompt()
        prompts.append({'idx': idx, "prompt": final_prompt})

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(call_api, prompt, temperature=0, max_tokens=300): prompt['idx'] for prompt in prompts}
        
        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future]
            try:
                result = future.result()
                gpt_result.append([result, idx])
            except Exception as e:
                print(f"Error occurred for prompt index {idx}: {e}")
    
    gpt_result = sorted(gpt_result, key=lambda x: x[-1])
    final_result = []
    #print(gpt_result)
    for i in range(len(gpt_result)):
        result = None
        try:
            match = re.search(r'"answer"\s*:\s*"(.*?)"', gpt_result[i][0], re.DOTALL)
            if match:
                result = match.group(1)
        except ValueError:
            pass
        if result is None:
            final_result.append("Error occured")
        else:
            final_result.append(result)
        
    return final_result, gpt_result


def efficient_gpt_text_ge(input_text):    
    gpt_result = []

    prompts = []
    for idx, text in enumerate(input_text):
        conv = conv_v2.copy()
        conv.append_message(conv.roles[0], text)
        final_prompt = conv.get_prompt()
        prompts.append({'idx': idx, "prompt": final_prompt})

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(call_api, prompt, temperature=0, max_tokens=1000): prompt['idx'] for prompt in prompts}
        
        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future]
            try:
                result = future.result()
                gpt_result.append([result, idx])
            except Exception as e:
                print(f"Error occurred for prompt index {idx}: {e}")

    #breakpoint()
    gpt_result = sorted(gpt_result, key=lambda x: x[-1])
    #print(gpt_result)
    final_result = []
    for i in range(len(gpt_result)):
        result = None
        try:
            match = re.search(r'"answer":\s*(\{.*?\})', gpt_result[i][0], re.DOTALL)
            #print(match)
            if match:
                result = match.group(1)
            else:
                match = re.search(r'"answer":\s*(.*)', gpt_result[i][0], re.DOTALL)
                if match:
                    result = match.group(1)+"\n }"             
        except ValueError:
            pass
        if result is None:
            #print(gpt_result[i])
            #breakpoint()
            final_result.append("Error occured")
        else:
            final_result.append(result)
            
    return final_result, gpt_result

def efficient_gpt_text_score(input_text, idx_grouped_lens):    
    gpt_result = []

    prompts = []
    for idx, text in enumerate(input_text):
        conv = conv_v4.copy()
        conv.append_message(conv.roles[0], text)
        final_prompt = conv.get_prompt()
        prompts.append({'idx': idx, "prompt": final_prompt})

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(call_api, prompt, temperature=0, max_tokens=15000): prompt['idx'] for prompt in prompts}
        
        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future]
            try:
                result = future.result()
                gpt_result.append([result, idx])
            except Exception as e:
                print(f"Error occurred for prompt index {idx}: {e}")

    #breakpoint()
    gpt_result = sorted(gpt_result, key=lambda x: x[-1])
    #print(gpt_result)
    final_cont_scores, final_conf_scores = [], []
    random.seed(42)
    for i in range(len(gpt_result)):
        try:
            json_str = gpt_result[i][0]

            if json_str.startswith("```json"):
                json_str = json_str[7:-3].strip()
       
            data = json.loads(json_str)
            #print(json_str)
            if data:
                #cont_data = {int(entry["text_id"]): float(entry["score"]) for entry in data["contribution scores"]}
                #cont_data = {int(entry["text_id"]): (float(entry["score"]) if entry["score"] is not None else random.uniform(0,1)) for entry in data["contribution scores"]}
                #conf_data = {int(entry["text_id"]): float(entry["score"]) for entry in data["confidence scores"]}
                #conf_data = {int(entry["text_id"]): (float(entry["score"]) if entry["score"] is not None else random.uniform(0,1)) for entry in data["confidence scores"]}
                cont_data = {}
                for entry in data["contribution_scores"]:
                    try:
                        text_id = int(entry["text_id"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    
                    score = entry.get("score")
                    try:
                        value = float(score) if score is not None else random.uniform(0, 1)
                    except (TypeError, ValueError):
                        value = random.uniform(0, 1)
                    cont_data[text_id] = value
                conf_data = {}
                for entry in data["confidence_scores"]:
                    try:
                        text_id = int(entry["text_id"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    score = entry.get("score")
                    try:
                        value = float(score) if score is not None else random.uniform(0, 1)
                    except (TypeError, ValueError):
                        value = random.uniform(0, 1)
    
                    conf_data[text_id] = value        
                #all_text_ids = sorted(set(cont_data.keys()) | set(conf_data.keys()))
        except json.JSONDecodeError:
            cont_match = re.search(r'"contribution_scores":\s*\[(.*?)\]', gpt_result[i][0], re.DOTALL)
            conf_match = re.search(r'"confidence_scores":\s*\[(.*?)\]', gpt_result[i][0], re.DOTALL)
            if cont_match:
                cont_scores = re.findall(r'"text_id":\s*(\d+)\s*,\s*"score":\s*([\d.]+)', cont_match.group(1))
                #cont_data = {int(text_id): float(score) for text_id, score in cont_scores}
                cont_data = {int(text_id): (float(score) if score is not None else random.uniform(0,1)) for text_id, score in cont_scores}

            if conf_match:
                conf_scores = re.findall(r'"text_id":\s*(\d+)\s*,\s*"score":\s*([\d.]+)', conf_match.group(1))
                #conf_data = {int(text_id): float(score) for text_id, score in conf_scores}
                conf_data = {int(text_id): (float(score) if score is not None else random.uniform(0,1)) for text_id, score in conf_scores}
            #all_text_ids = sorted(set(cont_data.keys()) | set(conf_data.keys()))
             
        all_text_ids = np.arange(idx_grouped_lens[i])+1
        cont_score = []
        conf_score = []
        for text_id in all_text_ids:
            cont_score.append(cont_data.get(text_id, np.random.uniform(0, 1)))
            conf_score.append(conf_data.get(text_id, np.random.uniform(0, 1)))
        final_cont_scores.append(cont_score)
        final_conf_scores.append(conf_score)
    final_result = {"cont_scores": final_cont_scores, "conf_scores": final_conf_scores}
            
    return final_result, gpt_result

def efficient_gpt_text_cls(input_text, n_clusters):
    gpt_result = []
    prompts = []
    classes = np.arange(n_clusters)
    probs = [1 / len(classes)] * len(classes)

    for idx, text in enumerate(input_text):
        conv = conv_v3.copy()
        conv.append_message(conv.roles[0], text)
        final_prompt = conv.get_prompt()
        prompts.append({'idx': idx, "prompt": final_prompt})
    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = {executor.submit(call_api, prompt, temperature=0, max_tokens=100): prompt['idx'] for prompt in prompts}

        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future]
            try:
                result = future.result()
                gpt_result.append([result, idx])
            except Exception as e:
                print(f"Error occurred for prompt index {idx}: {e}")

                gpt_result.append([{"answer":str(random.choices(classes, probs)[0]+1),"reason":""}, idx])


    gpt_result = sorted(gpt_result, key=lambda x: x[-1])
    result_cls = []
    for i in range(len(gpt_result)):
        result = None
        try:
            match = re.search(r'"answer"\s*:\s*"?(\d+)"?', gpt_result[i][0])
            if match:
                result = int(match.group(1))
        except ValueError:
            pass
        if result is None:
            result = random.choices(classes, probs)[0] + 1
        if int(result - 1) in range(n_clusters):
            result_cls.append(int(result) - 1)
        else:
            result_cls.append(random.choices(classes, probs)[0])

    return result_cls, gpt_result

def efficient_gpt_text_cls_equ(input_text):
    gpt_result = []
    prompts = []


    for idx, text in enumerate(input_text):
        conv = conv_v3.copy()
        conv.append_message(conv.roles[0], text)
        final_prompt = conv.get_prompt()
        prompts.append({'idx': idx, "prompt": final_prompt})

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = {executor.submit(call_api, prompt, temperature=0, max_tokens=100): prompt['idx'] for prompt in prompts}
        
        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future]
            try:
                result = future.result()
                gpt_result.append((result, idx))
            except Exception as e:
                print(f"Error occurred for prompt index {idx}: {e}")

                gpt_result.append([{"answer":str(random.choice([0, 1])),"reason":""}, idx]) 


    gpt_result = sorted(gpt_result, key=lambda x: x[-1])
    result_cls = []
    for i in range(len(gpt_result)):
        try:
            match = re.search(r'"answer"\s*:\s*"?(\d+)"?', gpt_result[i][0])
            if match:
                result = int(match.group(1))
        except ValueError:
            pass                
        if result is None:
            result = random.choice([0,1])
        if result in [0,1]:
            result_cls.append(result)
        else:
            result_cls.append(random.choice([0,1]))
    #breakpoint()
    return result_cls, gpt_result
