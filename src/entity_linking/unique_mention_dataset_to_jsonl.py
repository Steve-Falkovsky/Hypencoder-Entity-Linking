from pathlib import Path
import json
from utils import *

"""
desired output:
{
    "id": entity_id,
    "mention": mention_name,
    "mention_text": mention_text (the passage)
    "entity": entity_name, 
    "aliases": aliases,
    "definition": entity definition (what is it actually)
}
"""

# convert to JSONL - we want to create 3 differnent datasets, one for train, val, test
def toJSONL(split_path: str | Path, output_path: str | Path):
    # BC5CDR input (mention name, id, passage text)
    bc5cdr_triples = get_mention_names_id_passage_pairs(split_path)

    # eliminate duplicates (preserve first-seen order)
    unique_triples = list(dict.fromkeys(bc5cdr_triples))

    # MeSH2015 input (entity id -> (name, aliases, definition))
    id_to_entry = get_entity_id_to_name_aliases_definition(
        DATA_PROCESSED / "mesh2015.json.gz"
    )

    output = []
    for mention_name, entity_id, mention_text in unique_triples:
        if entity_id not in id_to_entry:
            continue

        entity_name, aliases, definition = id_to_entry[entity_id]
        aliases = filter_aliases(aliases)

        output.append(
            {
                "mention": mention_name,
                "mention_text": mention_text,
                "entity": entity_name,
                "aliases": aliases,
                "definition": definition,
                "id": entity_id,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in output:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")  # add newline for JSONL format

if __name__ == "__main__":
    out_dir = DATA_PROCESSED / "full_entry"
    out_dir.mkdir(parents=True, exist_ok=True)

    toJSONL(DATA_PROCESSED / "bc5cdr_train.bioc.xml.gz", out_dir / "train.jsonl")
    toJSONL(DATA_PROCESSED / "bc5cdr_test.bioc.xml.gz", out_dir / "test.jsonl")
    toJSONL(DATA_PROCESSED / "bc5cdr_val.bioc.xml.gz", out_dir / "validation.jsonl")