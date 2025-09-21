import gzip
import json
import gc

def load_ontology_tsv(filename, by_id=False):
    """
    Loads an ontology from a .tsv file.
    :param filename: the name of the file.
    :return: the ontology.
    """
    ontology = []
    with gzip.open(filename,'rt',encoding='utf8') as f:
        for line in f:
            split = line.rstrip('\n').split('\t')
            db_id = int(split[0])
            source_id = split[1]
            extra_ids = split[2].split('|')
            tags = split[3].split('|')
            name = split[4]
            description = split[5]
            aliases = [ alias for alias in split[6].split('|') if alias ]
            relations = split[7].split('|')
            
            assert len(name.strip()) > 0, f"Entity must have a name ({source_id=})"
            
            entity = {'db_id':int(db_id), 'id':source_id, 'extra_ids':extra_ids, 'tags':tags, 'name':name, 'description':description, 'aliases':aliases, 'relations':relations}
            
            ontology.append( entity )

    return ontology
            

def load_ontology(filename):
    """
    Loads an ontology from a file.
    :param filename: the name of the file (must be gzipped JSON or TSV file)
    :return: the ontology
    """
    assert filename.endswith('.json.gz') or filename.endswith('.tsv.gz'), "Ontology filename must be a gzipped JSON or TSV file"
    if filename.endswith('.json.gz'):
        with gzip.open(filename,'rt') as f:
            ontology = json.load(f)
    elif filename.endswith('.tsv.gz'):
        ontology = load_ontology_tsv(filename)

    return ontology


def load_ontology_by_id(filename):
    """
    Loads an ontology by identifier.
    :param filename: the name of the file containing the ontology (must be gzipped JSON or TSV file)
    :return: the ontology.
    """
    ontology = load_ontology(filename)
    ontology_by_id = { entity['id']:entity for entity in ontology }
    return ontology_by_id


def load_ontology_names(filename):
    """
    Loads the names in an ontology
    :param filename: the name of the file containing the ontology (must be gzipped JSON or TSV file)
    :return: the names in the ontology
    """
    ontology = load_ontology(filename)
    names = [ e['name'] for e in ontology ]
    
    del ontology
    gc.collect()

    return names
    