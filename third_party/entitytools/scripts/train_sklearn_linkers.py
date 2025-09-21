import argparse

from entitytools.sklearn_linkers import train_sklearn_linkers

def main():
    parser = argparse.ArgumentParser('Train sklearn classifiers for entity linking')
    parser.add_argument('--linked_dir',required=True,type=str,help='Directory with PubMed samples that have been linked')
    parser.add_argument('--crossencode_threshold',required=False,type=float,default=0.9,help='Threshold for cross-encoder scores')
    parser.add_argument('--min_samples_for_clf',required=False,type=int,default=15,help='Minimum samples for training a sklearn classifier')
    parser.add_argument('--out_file',required=True,type=str,help='Output file to save decisions and classifiers')
    args = parser.parse_args()
    
    train_sklearn_linkers(args.linked_dir, args.crossencode_threshold, args.min_samples_for_clf, args.out_file)

    print("Done.")

if __name__ == '__main__':
    main()



