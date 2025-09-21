import argparse
import json
import gzip
from bioc import biocxml

def get_concept_ids(collection):
    concept_ids = []
    for doc in collection.documents:
        for passage in doc.passages:
            for anno in passage.annotations:
                concept_id = anno.infons['concept_id']
                concept_ids.append(concept_id)
    
    return concept_ids

def main():
    parser = argparse.ArgumentParser(description='Make the datasets much smaller for faster testing. Modifications are made in-place.')
    parser.add_argument('--train_corpus',required=True,type=str,help='Gzipped BioC XML with training corpus')
    parser.add_argument('--val_corpus',required=True,type=str,help='Gzipped BioC XML with validation corpus')
    parser.add_argument('--frequent_mentions',required=True,type=str,help='List of mentions to use for filtering')
    parser.add_argument('--umls',required=True,type=str,help='Gzipped JSON of UMLS')
    args = parser.parse_args()

    print("Loading corpora...")
    with gzip.open(args.train_corpus,'rt',encoding='utf8') as f:
        train_collection = biocxml.load(f)
    with gzip.open(args.val_corpus,'rt',encoding='utf8') as f:
        val_collection = biocxml.load(f)

    print("Loading UMLS...")
    with gzip.open(args.umls,'rt',encoding='utf8') as f:
        ontology = json.load(f)

    print("Loading frequent mentions...")
    with open(args.frequent_mentions) as f:
        frequent_mentions = set( line.strip() for line in f )

    print("Reducing number of documents in corpora...")
    train_collection.documents = train_collection.documents[:5]
    val_collection.documents = val_collection.documents[:5]
    
    print("Reducing mentions in documents to those identified as frequent...")
    for doc in train_collection.documents + val_collection.documents:
        for passage in doc.passages:
            passage.annotations = [ anno for anno in passage.annotations if anno.text in frequent_mentions ]

    print("Reducing ontology to only those terms found in trimmed training corpus...")
    train_concept_ids = set(get_concept_ids(train_collection))
    ontology = [ e for e in ontology if e['cui'] in train_concept_ids ]

    print("Reducing validation corpus accordingly...")
    for doc in val_collection.documents:
        for passage in doc.passages:
            passage.annotations = [ anno for anno in passage.annotations if anno.infons['concept_id'] in train_concept_ids ]

    train_annos = [ anno for doc in train_collection.documents for passage in doc.passages for anno in passage.annotations ]
    val_annos = [ anno for doc in val_collection.documents for passage in doc.passages for anno in passage.annotations ]
    
    print(f"{len(train_collection.documents)=}")
    print(f"{len(val_collection.documents)=}")
    print(f"{len(train_annos)=}")
    print(f"{len(val_annos)=}")

    print("Saving corpora...")
    with gzip.open(args.train_corpus, 'wt', encoding='utf8') as f:
        biocxml.dump(train_collection, f)
    with gzip.open(args.val_corpus, 'wt', encoding='utf8') as f:
        biocxml.dump(val_collection, f)
        
    print("Saving UMLS...")
    with gzip.open(args.umls,'wt',encoding='utf8') as f:
        json.dump(ontology,f)
    
    print("Done.")

if __name__ == '__main__':
    main()
    