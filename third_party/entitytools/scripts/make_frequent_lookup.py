import gzip
import json
from bioc import biocxml
import argparse
import os

from entitytools.ontology import load_ontology
from entitytools.sampling import make_frequent_lookup

def main():
    parser = argparse.ArgumentParser('Make a lookup using the most common mappings in a set of linked documents')
    parser.add_argument('--in_corpus',required=False,type=str,help='Single gzipped BioC XML file to load')
    parser.add_argument('--in_dir',required=False,type=str,help='Directory containing gzipped BioC XML files')
    parser.add_argument('--output_file',required=True,type=str,help='Gzipped JSON file that is a lookup for mentions into the ontology')
    parser.add_argument('--ontology',required=True,type=str,help='GZipped ontology file')
    parser.add_argument('--lowercase',required=True,type=bool,help='Whether to use lowercase')
    parser.add_argument('--top_k',required=False,type=int,help='Top_k to use for the lookup (or just get everything)')
    parser.add_argument('--score_threshold',required=False,type=float,help='Minimum score for counting a link')
    args = parser.parse_args()

    assert args.in_corpus or args.in_dir, "Must provide --in_corpus or --in_dir"

    print("Loading documents...")
    if args.in_corpus:
        with gzip.open(args.in_corpus,'rt',encoding='utf8') as f:
            collections = [biocxml.load(f)]
    elif args.in_dir:
        collections = []
        for filename in os.listdir(args.in_dir):
            if filename.endswith('.bioc.xml.gz'):
                with gzip.open(f'{args.in_dir}/{filename}','rt',encoding='utf8') as f:
                    collections.append( biocxml.load(f) )
                    
    print("Loading ontology...")
    ontology = load_ontology(args.ontology)
        
    lookup_by_mention_text = make_frequent_lookup(collections, ontology, args.lowercase, score_threshold=args.score_threshold, top_k=args.top_k)

    print(f"Created lookup for {len(lookup_by_mention_text)} mention texts")
    
    print(f"Outputting to {args.output_file}...")
    with gzip.open(args.output_file,'wt') as f:
        json.dump(lookup_by_mention_text, f)

    print("Done.")


if __name__ == '__main__':
    main()


