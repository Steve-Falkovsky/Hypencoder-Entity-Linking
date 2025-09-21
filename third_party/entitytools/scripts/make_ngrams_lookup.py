import gzip
import argparse
import json

from entitytools.ngrams import load_and_make_ngrams_lookup

def main():
    parser = argparse.ArgumentParser('Apply a character n-grams retrieval model to a list of mentions')
    parser.add_argument('--mention_texts',required=True,type=str,help='Gzipped txt file with unique mentions')
    parser.add_argument('--ngrams_model',required=True,type=str,help='Directory that contains the ngrams model')
    parser.add_argument('--top_k',required=False,type=int,default=10,help='Maximum number to retrieve for each mention')
    parser.add_argument('--output_file',required=True,type=str,help='Gzipped JSON file that is a lookup for mentions into the ontology of the ngrams model')
    args = parser.parse_args()

    print("Loading mentions...")
    with gzip.open(args.mention_texts,'rt') as f:
        mention_texts = [ line.lower().strip() for line in f ]
    mention_texts = sorted(set(mention_texts))
    print(f"Loaded {len(mention_texts)} unique mentions...")
        
    lookup_by_mention_text = load_and_make_ngrams_lookup(mention_texts, args.ngrams_model, args.top_k)
    
    print(f"Created lookup for {len(lookup_by_mention_text)} mention texts")
    
    print(f"Outputting to {args.output_file}...")
    with gzip.open(args.output_file,'wt') as f:
        json.dump(lookup_by_mention_text, f)

    print("Done.")


if __name__ == '__main__':
    main()


