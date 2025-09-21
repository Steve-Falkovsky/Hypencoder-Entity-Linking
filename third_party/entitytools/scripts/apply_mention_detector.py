import gzip
import argparse
from bioc import biocxml

from entitytools.file_formats import pubmed_to_bioc
from entitytools.sentences import mark_sentences
from entitytools.mention_detector import do_ner

def main():
    parser = argparse.ArgumentParser('Run a mention detection pipeline and output a gzipped BioC XML file')
    parser.add_argument('--input_bioc',required=False,type=str,help='Gzipped BioC XML file')
    parser.add_argument('--input_pubmed',required=False,type=str,help='Gzipped PubMed XML file')
    parser.add_argument('--mention_detector_model',required=True,type=str,help='Directory with mention detector model')
    parser.add_argument('--output_bioc',required=True,type=str,help='Gzipped BioC XML file')
    parser.add_argument('--mark_sentences',action='store_true',help='Whether to run spaCy to split text into sentences and store')
    args = parser.parse_args()

    assert args.input_pubmed or args.input_bioc, "Must provide --input_bioc or --input_pubmed"

    print("Loading docs...")
    if args.input_bioc:
        with gzip.open(args.in_corpus,'rt',encoding='utf8') as f:
            collection = biocxml.load(f)
    elif args.input_pubmed:
        collection = pubmed_to_bioc(args.input_pubmed)
    print(f"Loaded {len(collection.documents)} documents")

    if args.mark_sentences:
        print("Marking sentences...")
        mark_sentences(collection)
    
    print("Doing NER...")
    do_ner(args.mention_detector_model, collection)

    print(f"Saving to {args.output_bioc}...")
    with gzip.open(args.output_bioc, 'wt') as f:
        biocxml.dump(collection, f)
    
    print("Done.")
	
if __name__ == '__main__':
    main()
