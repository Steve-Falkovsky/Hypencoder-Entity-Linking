import gzip
from bioc import biocxml
import argparse

from entitytools.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_corpus',required=True,type=str,help='Gzipped BioC XML with gold annotations')
    parser.add_argument('--pred_corpus',required=True,type=str,help='Gzipped BioC XML with predicted annotations')
    parser.add_argument('--metric_file',required=True,type=str,help='Evaluation file')
    args = parser.parse_args()

    with gzip.open(args.gold_corpus,'rt',encoding='utf8') as f:
        gold_collection = biocxml.load(f)
    with gzip.open(args.pred_corpus,'rt',encoding='utf8') as f:
        pred_collection = biocxml.load(f)

    eval_df = evaluate(gold_collection, pred_collection)
    eval_df.to_csv(args.metric_file, index=False)

if __name__ == '__main__':
    main()