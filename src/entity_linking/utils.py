from pathlib import Path
import gzip
import random
import bioc
import json
import re

# Project directories (resolve paths independently of the current working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# apparently .gz and .tar.gz are different things
# this is a tar archive
# source = "example_bioc_files.tar.gz"
# this is a gzip-compressed (.gz) single file
# Example: DATA_PROCESSED / "bc5cdr_train.bioc.xml.gz"


def extract_first_n_docs(path: str | Path, n: int | None = None):
    """
    Read a gzipped BioC XML file at `path` and return a list with info for the first `n` documents.
    If n is None (default), return info for all documents.
    Each list item is a dict: {'id': ..., 'passages': [ {'text': ..., 'annotations': [ {...}, ... ]}, ... ] }
    """
    if n is not None and n < 0:
        raise ValueError("n must be non-negative")

    results = []
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        data = file.read()

    collection = bioc.biocxml.loads(data)
    docs = collection.documents if n is None else collection.documents[:n]
    for document in docs:
        doc_info = {'id': document.id, 'passages': []}
        for passage in document.passages:
            passage_info = {'text': passage.text, 'annotations': []}
            for anno in passage.annotations:
                # serialize common fields of a BioC annotation
                anno_info = {
                    'id': getattr(anno, 'id', None),
                    'text': getattr(anno, 'text', None),
                    'infons': dict(getattr(anno, 'infons', {})),
                    'locations': [
                        {'offset': getattr(loc, 'offset', None), 'length': getattr(loc, 'length', None)}
                        for loc in getattr(anno, 'locations', [])
                    ]
                }
                passage_info['annotations'].append(anno_info)
            doc_info['passages'].append(passage_info)
        results.append(doc_info)

    return results


# get names
def get_mention_names(path: str | Path):
    docs = extract_first_n_docs(path)
    names = []
    for doc in docs:
        for passage in doc['passages']:
            for anno in passage['annotations']:
                names.append(anno['text'])
    return names


def get_mention_names_id_pairs(path: str | Path):
    """
    Get list of tuples (mention name, id) pairs from BC5CDR BioC XML file at `path`.
    """
    docs = extract_first_n_docs(path)
    name_id_pairs = []
    for doc in docs:
        for passage in doc['passages']:
            for anno in passage['annotations']:
                name_id_pairs.append((anno['text'], anno['infons'].get('concept_id')))
    return name_id_pairs


# Example: DATA_PROCESSED / "mesh2015.json.gz"

def read_first_n_from_json_gz(path: str | Path, n: int | None = None) -> list:
    """
    Read a gzipped JSON file at `path` and return the first `n` elements if the top-level
    JSON value is a list. If n is None (default), return all entries.
    Raises ValueError for non-list top-level or invalid `n`.
    """
    if n is not None and n < 0:
        raise ValueError("n must be non-negative")

    with gzip.open(path, 'rt', encoding='utf-8') as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"JSON top-level is {type(data).__name__}, expected list")

    return data if n is None else data[:n]


# get entity names
def get_entity_names(path: str | Path, n: int | None = None):
    entities = read_first_n_from_json_gz(path, n)
    names = [entity['name'] for entity in entities]
    return names

# get entity aliases
def get_entity_aliases(path: str | Path, n: int | None = None):
    entities = read_first_n_from_json_gz(path, n)
    # aliases per entity
    aliases = [entity['aliases'] for entity in entities]
    return aliases


def get_entity_name_id_pairs(path: str | Path, n: int | None = None):
    entities = read_first_n_from_json_gz(path, n)
    name_id_pairs = [(entity['name'], entity['id']) for entity in entities]
    return name_id_pairs


# function to get an entry by id
def get_entry_by_id(data: list, entry_id: str) -> dict | None:
    """
    Get a JSON entry by its ID from a list of entries.
    """
    for item in data:
        if item.get("id") == entry_id:
            return item
    return None


def get_entity_id_to_name_and_aliases(path: str | Path, n: int | None = None) -> dict[str, tuple[str, list[str]]]:
    """
    Build a mapping: entity_id -> (entity_name, aliases_list)
    """
    entities = read_first_n_from_json_gz(path, n)

    id_to_name_aliases: dict[str, tuple[str, list[str]]] = {}
    for ent in entities:
        ent_id = ent.get("id")
        name = ent.get("name")
        aliases = ent.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = [str(aliases)]

        id_to_name_aliases[str(ent_id)] = (str(name), [str(a) for a in aliases])

    return id_to_name_aliases


def filter_aliases(aliases, max_length=500):
    """
    Remove duplicate words across all strings in the list.
    Words are separated by spaces or commas followed by spaces.
    Commas within words (like "2,6-Dimethylphenyl") are preserved.
    
    Truncate each resulting string to max_length.
    """
    seen_words = set()
    result = []
    
    for alias in aliases:
        # Split by comma+space or just spaces (preserves commas within words)
        words = [w.strip() for w in re.split(r',\s+|\s+', alias) if w.strip()]
        
        # Keep only words we haven't seen before
        unique_words = [w for w in words if w not in seen_words]
        
        # Add these words to our seen set
        seen_words.update(unique_words)
        
        # Join the unique words back together
        result_string = ' '.join(unique_words)
        
        # Only add non-empty strings to result
        if result_string:
            result.append(result_string)
            
            
    # turn result into a string and truncate to max_length
    result = ' '.join(result)[:max_length]
    return result


def get_mention_names_id_passage_pairs(path: str | Path):
    """
    Get list of tuples (mention name, concept_id, passage_text) from a gzipped
    BC5CDR BioC XML file at `path`.
    """
    path = Path(path)

    with gzip.open(path, "rt", encoding="utf-8") as file:
        data = file.read()

    collection = bioc.biocxml.loads(data)

    triples: list[tuple[str, str, str]] = []
    for document in collection.documents:
        for passage in document.passages:
            passage_text = getattr(passage, "text", "") or ""
            for anno in passage.annotations:
                mention = getattr(anno, "text", None)
                if not mention:
                    continue

                infons = dict(getattr(anno, "infons", {}) or {})
                concept_id = infons.get("concept_id")
                if not concept_id:
                    continue

                triples.append((str(mention), str(concept_id), passage_text))

    return triples


def get_entity_id_to_name_aliases_definition(
    path: str | Path, n: int | None = None
) -> dict[str, tuple[str, list[str], str | None]]:
    """
    entity_id -> (name, aliases, definition)
    Assumes the JSON objects use the literal field 'definition'.
    """
    entities = read_first_n_from_json_gz(path, n)

    out: dict[str, tuple[str, list[str], str | None]] = {}
    for ent in entities:
        ent_id = str(ent["id"])
        name = str(ent["name"])
        aliases = ent.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = [aliases]
        aliases = [str(a) for a in aliases]
        definition = ent.get("definition")
        out[ent_id] = (name, aliases, definition)

    return out

