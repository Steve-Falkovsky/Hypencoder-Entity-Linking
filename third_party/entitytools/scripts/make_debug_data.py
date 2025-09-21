import gzip
import argparse
from bioc import biocxml
import json

def main():
    parser = argparse.ArgumentParser('Shrink down a corpus and the JSONed ontology. WARNING: everything is done in-place.')
    parser.add_argument('--train_corpus',required=True,type=str,help='Gzipped BioC XML with training corpus')
    parser.add_argument('--val_corpus',required=True,type=str,help='Gzipped BioC XML with validation corpus')
    parser.add_argument('--test_corpus',required=True,type=str,help='Gzipped BioC XML with validation corpus')
    parser.add_argument('--ontology',required=True,type=str,help='Gzipped JSON with ontology information')
    parser.add_argument('--trim_count',type=int,default=10,help='Number of documents to trim down to')
    args = parser.parse_args()
    
    print("Loading docs...")
    with gzip.open(args.train_corpus,'rt',encoding='utf8') as f:
        train_collection = biocxml.load(f)
    with gzip.open(args.val_corpus,'rt',encoding='utf8') as f:
        val_collection = biocxml.load(f)
    with gzip.open(args.test_corpus,'rt',encoding='utf8') as f:
        test_collection = biocxml.load(f)

    print(f"Trimming down to {args.trim_count} documents in each split")
    train_collection.documents = train_collection.documents[:args.trim_count]
    val_collection.documents = val_collection.documents[:args.trim_count]
    test_collection.documents = test_collection.documents[:args.trim_count]

    print("Saving docs...")
    with gzip.open(args.train_corpus, 'wt') as f:
        biocxml.dump(train_collection, f)
    with gzip.open(args.val_corpus, 'wt') as f:
        biocxml.dump(val_collection, f)
    with gzip.open(args.test_corpus, 'wt') as f:
        biocxml.dump(test_collection, f)

    train_concept_ids = [ anno.infons['concept_id'] for doc in train_collection.documents for passage in doc.passages for anno in passage.annotations ]
    val_concept_ids = [ anno.infons['concept_id'] for doc in val_collection.documents for passage in doc.passages for anno in passage.annotations ]
    test_concept_ids = [ anno.infons['concept_id'] for doc in test_collection.documents for passage in doc.passages for anno in passage.annotations ]

    combined_ids = set(train_concept_ids + val_concept_ids + test_concept_ids)
    print(f"Found {len(combined_ids)} unique concept IDs")

    print("Loading ontology...")
    with gzip.open(args.ontology,'rt') as f:
        ontology = json.load(f)
    print(f"Loaded ontology with {len(ontology)} terms")

    ontology = [ e for e in ontology if e['id'] in combined_ids ]
    print(f"Trimmed ontology to {len(ontology)} terms")
    
    print("Saving ontology...")
    with gzip.open(args.ontology,'wt') as f:
        json.dump(ontology,f)

    print("Done.")
	
if __name__ == '__main__':
    main()
