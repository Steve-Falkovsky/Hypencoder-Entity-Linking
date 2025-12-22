from pathlib import Path
import json
from .utils import DATA_PROCESSED, get_mention_names_id_pairs, get_entity_name_id_pairs

"""
desired output:
{"mention": "word1", "entity": "word2", "id": "unique_id_A"}
{"mention": "word3", "entity": "word4", "id": "unique_id_B"}
{"mention": "word5", "entity": "word6", "id": "unique_id_C"}
"""


# convert to JSONL - we want to create 3 differnent datasets, one for train, val, test
def toJSONL(split_path: str | Path, output_path: str | Path):

    # get name and id
    # BC5CDR input  (mention name,id)
    bc5cdr_name_id_pairs = get_mention_names_id_pairs(split_path)

    # eliminate duplicates
    unique_mention_name_id_pairs = list(set(bc5cdr_name_id_pairs))

    # MeSH2015 input (entity name,id)
    mesh_name_id_pairs = get_entity_name_id_pairs(DATA_PROCESSED / "mesh2015.json.gz")

    # pair[1] is id, pair[0] is the name
    mesh_pairs_dict = {pair[1]: pair[0] for pair in mesh_name_id_pairs}

    # for entities input, only use ones that appear in the bc5cdr
    # create output - a list of dicts with mention, entity, id (corresponding to every line in the output JSONL)
    output = [{"mention": mention_name, "entity": mesh_pairs_dict[id], "id": id} for (mention_name, id) in unique_mention_name_id_pairs]
    
    # write to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in output:
            json.dump(entry, f)
            f.write('\n') # add newline for JSONL format


toJSONL(DATA_PROCESSED / "bc5cdr_train.bioc.xml.gz", DATA_PROCESSED / "bc5cdr_train.jsonl")
toJSONL(DATA_PROCESSED / "bc5cdr_test.bioc.xml.gz", DATA_PROCESSED / "bc5cdr_test.jsonl")
toJSONL(DATA_PROCESSED / "bc5cdr_val.bioc.xml.gz", DATA_PROCESSED / "bc5cdr_val.jsonl")

