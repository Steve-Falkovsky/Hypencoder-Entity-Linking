from pathlib import Path
import gzip
import random
import bioc
import json

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
def get_entity_names(path: str | Path):
    entities = read_first_n_from_json_gz(path)
    names = [entity['name'] for entity in entities]
    return names


def get_entity_name_id_pairs(path: str | Path):
    entities = read_first_n_from_json_gz(path)
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


