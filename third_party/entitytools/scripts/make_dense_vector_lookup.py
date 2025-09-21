import argparse
import gzip
from bioc import biocxml
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json

from entitytools.utils import get_annotation_texts
from entitytools.dense_vectors import make_dense_lookup

def main():
    parser = argparse.ArgumentParser('Apply a character n-grams retrieval model to a list of mentions')
    parser.add_argument('--mention_texts',required=False,type=str,help='Gzipped txt file with unique mentions')
    parser.add_argument('--train_corpus',required=False,type=str,help='Gzipped BioC XML with training corpus')
    parser.add_argument('--val_corpus',required=False,type=str,help='Gzipped BioC XML with validation corpus')
    parser.add_argument('--vectors',required=True,type=str,help='Precalculated vectors to base the lookup on')
    parser.add_argument('--top_k',required=False,type=int,default=10,help='Maximum number to retrieve for each mention')
    parser.add_argument('--model_name',required=True,type=str,help='Transformer model to use')
    parser.add_argument('--output_file',required=True,type=str,help='Gzipped JSON file that is a lookup for mentions into the ontology of the ngrams model')
    args = parser.parse_args()

    assert args.mention_texts or (args.train_corpus and args.val_corpus)

    if args.mention_texts:
        with gzip.open(args.mention_texts,'rt') as f:
            anno_texts = [ line.lower().strip() for line in f ]
        anno_texts = sorted(set(anno_texts))
    else:
        with gzip.open(args.train_corpus,'rt',encoding='utf8') as f:
            train_collection = biocxml.load(f)
        with gzip.open(args.val_corpus,'rt',encoding='utf8') as f:
            val_collection = biocxml.load(f)

        anno_texts = get_annotation_texts([train_collection, val_collection])
        anno_texts = sorted(set( anno_text.lower() for anno_text in anno_texts ))

    print(f"Loaded {len(anno_texts)} unique mentions...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device=}")

    print(f"Loading {args.model_name} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    
    print("Loading ontology vectors...")
    onto_vectors = np.load(args.vectors)

    print("Creating lookup...")
    lookup_by_mention_text = make_dense_lookup(model, tokenizer, onto_vectors, anno_texts, args.top_k)
    
    print(f"Created lookup for {len(lookup_by_mention_text)} mention texts")
    
    print(f"Outputting to {args.output_file}...")
    with gzip.open(args.output_file,'wt') as f:
        json.dump(lookup_by_mention_text, f)

    print("Done.")


if __name__ == '__main__':
    main()


