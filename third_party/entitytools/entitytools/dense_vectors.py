import torch
from more_itertools import chunked
from tqdm.auto import tqdm
import faiss

def make_dense_vectors(model, tokenizer, texts):
    dense_vectors = []
    with torch.no_grad():
        for batch in chunked(tqdm(texts), 1000):
            tokenized = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors='pt')
            outputs = model(input_ids=tokenized['input_ids'].to(model.device), attention_mask=tokenized['attention_mask'].to(model.device))
            cls_vectors = outputs.last_hidden_state[:,0,:]
            dense_vectors.append(cls_vectors.cpu())
    
    dense_vectors = torch.vstack(dense_vectors).numpy()
    
    return dense_vectors

def make_dense_lookup(model, tokenizer, onto_vectors, anno_texts, top_k):
    
    query_vectors = make_dense_vectors(model, tokenizer, anno_texts)

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(onto_vectors.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(onto_vectors)

    eval_biencoder_distances, eval_biencoder_indices = gpu_index_flat.search(query_vectors, top_k)

    lookup_by_mention_text = {}
    for i,anno_text in enumerate(anno_texts):
        lookup_by_mention_text[anno_text] = eval_biencoder_indices[i].tolist()

    return lookup_by_mention_text
    