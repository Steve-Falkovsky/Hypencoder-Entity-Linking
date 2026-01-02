# There are 2 inputs to the model which I need to embed and then compare:
# 1. The mention text from BC5CDR (probably better to concatenate title + abstract)
# 2. The name + definition (description) of all entities

# functions:
# function to extract the title+abstract from every passage in every doc in BC5CDR
# function to extract the name+definition of every entity in MeSH2015

# import model
# prepare both types of inputs
# construct positives and negatives 50/50
# train on the data constructed so that dot product/ cosine similarity is high when the entity is correctly matched and low when no match



# I want to make the first draft with only on 50 documents (50 title + abstract pairs)
# we still have to include all MeSH2015 entities so we don't miss links

# do this in jupyter notebook because this will need to scale anyway and I don't have the compute power for it


from utils import DATA_PROCESSED, extract_first_n_docs, read_first_n_from_json_gz

# BC5CDR input 
docs = extract_first_n_docs(DATA_PROCESSED / "bc5cdr_train.bioc.xml.gz", 50)

# concatenate title and abstract
texts = [doc['passages'][0]['text'] +  doc['passages'][0]['text'] for doc in docs]

# MeSH2015 input
# could try:
# 1. name + definition
# 2. name + aliases + definition
# (not sure how helpful it is to include aliases if we have 2 different entities that share an alias)

# for now do option 1 (name + definition)
entities = read_first_n_from_json_gz(DATA_PROCESSED / "mesh2015.json.gz")
entities = [entity['name'] + entity['definition'] for entity in entities]


# constructing positive pairs
# randomly sample mentions from the documents - we know the correct entity by the m_id
# need to keep track which document they relate to (so we can encode the text)


# constructing negative pairs
# take the same texts - now sample random entities from MeSH2015
# check that they're actually negative by checking that the m_id doesn't match



# SapBert model for outputing dense vector representations
from transformers import pipeline

model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
extractor = pipeline("feature-extraction", model=model_name)






 