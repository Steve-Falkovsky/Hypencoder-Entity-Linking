import gzip
from bioc import biocxml
import argparse

from entitytools.ontology import load_ontology
from entitytools.sklearn_linkers import apply_linkers

def main():
    parser = argparse.ArgumentParser('Apply sklearn linkers to a BioC XML file')
    parser.add_argument('--in_corpus',required=True,type=str,help='Gzipped BioC XML with training corpus')
    parser.add_argument('--out_corpus',required=True,type=str,help='Gzipped BioC XML with validation corpus')
    parser.add_argument('--linkers',required=True,type=str,help='Trained sklearn linkers')
    parser.add_argument('--ontology',required=True,type=str,help='GZipped ontology file')
    args = parser.parse_args()

    print("Loading documents...")
    with gzip.open(args.in_corpus,'rt',encoding='utf8') as f:
        collection = biocxml.load(f)
    print(f"Loaded {len(collection.documents)} training documents")

    print("Loading ontology...")
    ontology = load_ontology(args.ontology)

    apply_linkers(collection, args.linkers, ontology)
    
    print(f"Saving to {args.out_corpus}")
    with gzip.open(args.out_corpus, 'wt', encoding='utf8') as f:
        biocxml.dump(collection, f)

    print("Done.")


if __name__ == '__main__':
    main()



