from pathlib import Path
import json
from utils import *

"""
desired output:
{
    "mention": mention_name,
    "entity": entity_name, 
    "aliases": aliases,
    "id": entity_id,
}

The length of the alias list can be extremely long and there are many duplicate words.
In order for the aliases to fit in the context of the model, we'll do pre-processing.
(also, a shorter alias list is better for information distillation)

even after de-duplication, some aliases lists are too long.
We can either:
1. truncate (simple, but lose information)
2. maybe it's better to have the shorter aliases come first (if it's long then the context is clear anyway)
3. split the alias list into several parts and have:
"entity: {entity_name} \n aliases: {aliases_part1}"
"entity: {entity_name} \n aliases: {aliases_part2}"
"""

# convert to JSONL - we want to create 3 differnent datasets, one for train, val, test
def toJSONL(split_path: str | Path, output_path: str | Path):

    # BC5CDR input (mention name, id)
    bc5cdr_name_id_pairs = get_mention_names_id_pairs(split_path)

    # eliminate duplicates (preserve first-seen order)
    unique_mention_name_id_pairs = list(dict.fromkeys(bc5cdr_name_id_pairs))

    # MeSH2015 input (entity id -> (name, aliases))
    mesh_id_to_name_aliases = get_entity_id_to_name_and_aliases(
        DATA_PROCESSED / "mesh2015.json.gz"
    )

    output = []
    for mention_name, concept_id in unique_mention_name_id_pairs:
        if concept_id not in mesh_id_to_name_aliases:
            continue

        entity_name, aliases = mesh_id_to_name_aliases[concept_id]
        aliases = filter_aliases(aliases)
        
        output.append(
            {
                "mention": mention_name,
                "entity": entity_name, 
                "aliases": aliases,
                "id": concept_id,
            }
        )

    # write to JSONL file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in output:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")  # add newline for JSONL format

if __name__ == "__main__":
    out_dir = DATA_PROCESSED / "name+alias"
    out_dir.mkdir(parents=True, exist_ok=True)

    toJSONL(DATA_PROCESSED / "bc5cdr_train.bioc.xml.gz", out_dir / "train.jsonl")
    toJSONL(DATA_PROCESSED / "bc5cdr_test.bioc.xml.gz", out_dir / "test.jsonl")
    toJSONL(DATA_PROCESSED / "bc5cdr_val.bioc.xml.gz", out_dir / "validation.jsonl")