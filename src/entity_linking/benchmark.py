# we can test the core sapbert model vs. a finetuned version

# we generate a dense vector using the model, then calculate cosine similarity with all mentions
# get back top1, top5, or top 10
# to start let's just consider top 1
# if correct +1, if not 0

from utils import DATA_PROCESSED, get_mention_names_id_pairs, get_entity_name_id_pairs, get_entry_by_id

# get only names
# mention_names = get_mention_names(DATA_PROCESSED / "bc5cdr_train.bioc.xml.gz")
# entity_names = get_entity_names(DATA_PROCESSED / "mesh2015.json.gz")


# get name and id
# BC5CDR input  (mention names and ids)
bc5cdr_name_id_pairs = get_mention_names_id_pairs(DATA_PROCESSED / "bc5cdr_train.bioc.xml.gz")

# MeSH2015 input (entity names and ids)
mesh_name_id_pairs = get_entity_name_id_pairs(DATA_PROCESSED / "mesh2015.json.gz")


# Core SapBert model for outputing dense vector representations 
# this is the feature extraction pipeline so we can get the embeddings directly (we can only do inference with this, no fine-tuning)
from transformers import pipeline, AutoTokenizer, AutoModel


core_model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# core model
extractor = pipeline("feature-extraction", model=core_model_name)


# finetuned model
model_name = "Stevenf232/fine-tuned-SapBERT"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(core_model_name)




# ------------------ evaluating core model -------------------------

# input all names to model so it will create dense vectors of all the names (of the mentions and entities)
mention_name_features = [extractor(name) for (name, id) in bc5cdr_name_id_pairs]
entity_name_features = [extractor(name) for (name, id) in mesh_name_id_pairs]

import numpy as np

# mention name vectors
mention_name_context_vectors = [np.array(feature) for feature in mention_name_features]
mention_vectors = [context_vector[0,0,:] for context_vector in mention_name_context_vectors]

# entity name vectors
entity_name_context_vectors = [np.array(feature) for feature in entity_name_features]
entity_vectors = [context_vector[0,0,:] for context_vector in entity_name_context_vectors]




from sklearn.metrics.pairwise import cosine_similarity
for i in range(5):
    scores = cosine_similarity(mention_vectors[i].reshape(1,-1), entity_vectors).flatten().tolist()
    top_idx = np.argmax(scores)
    top_score = scores[top_idx]
    top_match = mesh_name_id_pairs[top_idx][0]
    correct_name = get_entry_by_id(bc5cdr_name_id_pairs, top_idx)[0]
    print(f"top_match: {top_match}, correct name: {correct_name}")
    



