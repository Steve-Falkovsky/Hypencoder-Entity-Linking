import gzip
from bioc import biocxml
import argparse

from entitytools.utils import get_annotation_texts
from entitytools.sampling import sample_annotations_from_bioc_dir

def main():
    parser = argparse.ArgumentParser("Script to sample mentions from a big directory of BioC files (i.e. mentions extracted across PubMed)")
    parser.add_argument('--mode',required=True,type=str,help='Whether to sample everything or only use mentions from the provided corpora (all/corpora)')
    parser.add_argument('--train_corpus',required=False,type=str,help='Gzipped BioC XML with training corpus (for --mode corpora)')
    parser.add_argument('--val_corpus',required=False,type=str,help='Gzipped BioC XML with validation corpus (for --mode corpora)')
    parser.add_argument('--input_dir',required=True,type=str,help='Directory with PubMed BioC XML files that have mentions')
    parser.add_argument('--sample_size',required=False,type=int,default=100,help='Max number to sample for each mention')
    parser.add_argument('--require_complete',action='store_true',help='Remove any mention samples that do not have complete set (set by --sample_size)')
    parser.add_argument('--out_dir',required=True,type=str,help='Directory to put Gzipped JSONL files with mentions & contexts')
    parser.add_argument('--out_mentions',required=False,type=str,help='Optional Gzipped TXT file with the unique mention texts')
    args = parser.parse_args()

    assert args.mode in ['all','corpora']
    assert args.sample_size > 0, "Must provide a positive value for the number of samples"

    annotation_texts = None
    if args.mode == 'corpora':
        assert args.train_corpus and args.val_corpus, "Must provide --train_corpus and --val_corpus when --mode corpora"
        
        with gzip.open(args.train_corpus,'rt',encoding='utf8') as f:
            train_collection = biocxml.load(f)
        with gzip.open(args.val_corpus,'rt',encoding='utf8') as f:
            val_collection = biocxml.load(f)

        annotation_texts = get_annotation_texts([train_collection, val_collection])

    sample_annotations_from_bioc_dir(args.input_dir, 
                                     args.sample_size, 
                                     args.require_complete, 
                                     args.out_dir, 
                                     out_mentions=args.out_mentions, 
                                     annotation_text_filter=annotation_texts)
    
    print("Done.")


if __name__ == '__main__':
    main()
    

