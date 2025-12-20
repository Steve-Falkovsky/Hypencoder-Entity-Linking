"""
create JSONL files for contrastive loss training from BC5CDR and MeSH2015 datasets to be used with Hypencoder

Desired format for each line in the JSONL file:
{
  "query": {
    "id": query ID,
    "content": query text,
  },
  "items": [
    {
      "id": passage ID,
      "content": passage text,
      "score": Optional teacher score,
      "type": Sometimes used to specify type of item,
    },
  ]
}

Contrastive Loss without Hard Negatives: just include only a single positive i.e. relevant item to the query in the "items" for that query. 
The negatives will then be the other queries positives in the batch.


we already have JSONL files with format:
{"mention": "word1", "entity": "word2", "id": "unique}

so we need to convert them to the desired format
"""
import json
from utils import *

def convert_to_contrastive_format(input_jsonl_path: str, output_jsonl_path: str):
    output = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            mention = entry['mention']
            entity = entry['entity']
            id_ = entry['id']
            
            contrastive_entry = {
                "query": {
                    "id": id_,
                    "content": mention,
                },
                "items": [
                    {
                        "id": id_,
                        "content": entity,
                        # for the fields below the values are optional (for contrastive loss)
                        "score": None,
                        "type": None,
                    }
                ]
            }
            output.append(contrastive_entry)
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in output:
            json.dump(entry, f)
            f.write('\n')
            
            
convert_to_contrastive_format("my_stuff/processed_sources/name_only_train.jsonl", "my_stuff/processed_sources/bc5cdr_train_hypencoder_contrastive.jsonl")
convert_to_contrastive_format("my_stuff/processed_sources/name_only_val.jsonl", "my_stuff/processed_sources/bc5cdr_val_hypencoder_contrastive.jsonl")
convert_to_contrastive_format("my_stuff/processed_sources/name_only_test.jsonl", "my_stuff/processed_sources/bc5cdr_test_hypencoder_contrastive.jsonl")