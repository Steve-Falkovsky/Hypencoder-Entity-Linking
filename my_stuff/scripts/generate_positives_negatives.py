from my_stuff.utils import *

# BC5CDR input  (mention names and ids)
bc5cdr_name_id_pairs = get_mention_names_id_pairs("processed_sources/bc5cdr_train.bioc.xml.gz")


# MeSH2015 input (entity names and ids)
mesh_name_id_pairs = get_entity_name_id_pairs("processed_sources/mesh2015.json.gz")

# constructing positive pairs
# take all mention names and get the corresponding entity name by matching the id

mesh_id_to_name = {entity_id: entity_name for entity_name, entity_id in mesh_name_id_pairs}

positive_pairs = [
    (mention_name, mesh_id_to_name[mention_id])
    for mention_name, mention_id in bc5cdr_name_id_pairs
    if mention_id in mesh_id_to_name
]

# constructing negative pairs
# I want to have 4 times as many negative pairs as positive pairs
# There are several negative sampling techinques
# for now take one mention name from the positives and one entity name from the positives that don't match
# make it sample randomly from the positives
import random
negative_pairs = []
while len(negative_pairs) < 4 * len(positive_pairs):
    mention_name, mention_id = random.choice(positive_pairs)
    entity_name, entity_id = random.choice(positive_pairs)
    if mention_id != entity_id:
        negative_pairs.append((mention_name, entity_name))


# We'll turn them from tuples into dictionaries with boolean label of whether they are positive or negative pairs:
training_data = []
for mention_name, entity_name in positive_pairs:
    training_data.append({'mention_name': mention_name, 'entity_name': entity_name, 'label': True})
for mention_name, entity_name in negative_pairs:
    training_data.append({'mention_name': mention_name, 'entity_name': entity_name, 'label': False})
    
    
# Now we'll split our dataset into training, validation and test splits:
from sklearn.model_selection import train_test_split

train_pairs, valtest_pairs = train_test_split(training_data, train_size=0.6, random_state=43)
val_pairs, test_pairs = train_test_split(valtest_pairs, train_size=0.5, random_state=43)


# store train, val, test splits as json.gz files
import json
import gzip

def save_json_gz(data, filepath):
    with gzip.open(filepath, 'wt', encoding='utf-8') as zipfile:
        json.dump(data, zipfile)
        
save_json_gz(train_pairs, "processed_sources/pairs/mention_entity_name_only_linking_train.json.gz")
save_json_gz(val_pairs, "processed_sources/pairs/mention_entity_name_only_linking_val.json.gz")
save_json_gz(test_pairs, "processed_sources/pairs/mention_entity_name_only_linking_test.json.gz")