import gzip
from bioc import biocxml
import json
import argparse

from entitytools.mention_detector import train_mention_detector
from entitytools.wandb import tune_with_wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,type=str,help='Whether to train a single model or hyperparameter tune (train/tune)')
    parser.add_argument('--train_corpus',required=True,type=str,help='Gzipped BioC XML with training corpus')
    parser.add_argument('--val_corpus',required=True,type=str,help='Gzipped BioC XML with validation corpus')
    parser.add_argument('--sweep_id',required=False,type=str,help='sweep_id for joining wandb sweep')
    parser.add_argument('--params',required=False,type=str,help='JSON file with parameters to use for saving out model')
    parser.add_argument('--predict_variants',action='store_true',help='Classify entities that are have concept_id="variant"')
    parser.add_argument('--output_dir',required=False,type=str,help='Directory to save model')
    args = parser.parse_args()

    assert args.mode in ['train','tune'], "--mode must be train or tune"

    with gzip.open(args.train_corpus,'rt',encoding='utf8') as f:
        train_collection = biocxml.load(f)
    with gzip.open(args.val_corpus,'rt',encoding='utf8') as f:
        val_collection = biocxml.load(f)
        
    print(f"Loaded {len(train_collection.documents)} training documents")
    print(f"Loaded {len(val_collection.documents)} validation documents")

    MODEL_NAME = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

    if args.mode == 'train':
        assert args.params, "Must provide a parameter file with --params"
        assert args.output_dir, "Must provide an output directory with --output_dir"

        with open(args.params) as f:
            params = json.load(f)
        
        train_mention_detector(
            train_collection = train_collection,
            val_collection = val_collection,
            model_name = MODEL_NAME,
            learning_rate = params["mention_detector"]["learning_rate"],
            batch_size = params["mention_detector"]["batch_size"],
            weight_decay = params["mention_detector"]["weight_decay"],
            mask_neg_rate = params["mention_detector"]["mask_neg_rate"],
            predict_variants = args.predict_variants,
            output_dir = args.output_dir
        )

        print("Done.")
    else:

        tune_with_wandb(
            project = 'el_mention_detector',
            sweep_id = args.sweep_id,
            sweep_method = 'grid',
            metric_name = 'eval_macro_f1',
            metric_goal = 'maximize',
            training_func = train_mention_detector,
            fixed_args = {
                "train_collection": train_collection,
                "val_collection": val_collection,
                "model_name": MODEL_NAME,
                "predict_variants": args.predict_variants
            },
            tunable_args = {
                "learning_rate": [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5],
                "batch_size": [8, 16, 32],
                "weight_decay": [0.0, 0.01],
                "mask_neg_rate": [0.0, 0.2, 0.4, 0.8, 0.9],
            }
        )

    

if __name__ == '__main__':
    main()
    



