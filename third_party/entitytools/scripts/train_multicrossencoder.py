import gzip
from bioc import biocxml
import torch
import json
import argparse
import time

from entitytools.ontology import load_ontology
from entitytools.multicrossencoder import train_multicrossencoder
from entitytools.wandb import tune_with_wandb
from entitytools.ngrams import get_unique_annotation_texts, load_and_make_ngrams_lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, type=str,
                        help='Whether to train a single model or hyperparameter tune (train/tune)')
    parser.add_argument('--train_corpus', required=True, type=str, help='Gzipped BioC XML with training corpus')
    parser.add_argument('--val_corpus', required=True, type=str, help='Gzipped BioC XML with validation corpus')
    parser.add_argument('--ontology', required=True, type=str, help='Gzipped TSV or JSON with ontology information')
    parser.add_argument('--ngrams_model', required=True, type=str, help='Directory of ngrams model')
    parser.add_argument('--sweep_id', required=False, type=str, help='sweep_id for joining wandb sweep')
    parser.add_argument('--params', required=False, type=str,
                        help='JSON file with parameters to use for saving out model')
    parser.add_argument('--output_dir', required=False, type=str, help='Directory to save model')
    parser.add_argument('--debug', action='store_true', help='Run a tiny subset of data to check things')
    parser.add_argument('--metric_file', required=False, default=None, type=str, help='File to store times')
    args = parser.parse_args()

    assert args.mode in ['train', 'tune'], "--mode must be train or tune"

    assert torch.cuda.is_available(), "Must have a GPU to train"

    with gzip.open(args.train_corpus, 'rt', encoding='utf8') as f:
        train_collection = biocxml.load(f)
    with gzip.open(args.val_corpus, 'rt', encoding='utf8') as f:
        val_collection = biocxml.load(f)

    print(f"Loaded {len(train_collection.documents)} training documents")
    print(f"Loaded {len(val_collection.documents)} validation documents")

    ontology = load_ontology(args.ontology)

    if args.debug:
        print("Trimming down documents and ontology for quick testing")
        train_collection.documents = train_collection.documents[:1]
        val_collection.documents = val_collection.documents[:1]

    train_anno_texts = get_unique_annotation_texts(train_collection, lowercase=True)
    val_anno_texts = get_unique_annotation_texts(val_collection, lowercase=True)
    candidates_lookup = load_and_make_ngrams_lookup(train_anno_texts + val_anno_texts, args.ngrams_model, 10)

    if args.mode == 'train':
        assert args.params, "Must provide a parameter file with --params"
        assert args.output_dir, "Must provide an output directory with --output_dir"

        with open(args.params) as f:
            params = json.load(f)

        process_time = train_multicrossencoder(
            train_collection,
            val_collection,
            ontology,
            candidates_lookup,
            top_k=params["multicrossencoder"]["top_k"],
            learning_rate=params["multicrossencoder"]["learning_rate"],
            batch_size=params["multicrossencoder"]["batch_size"],
            output_dir=args.output_dir,
            crossencoder_model=params["multicrossencoder"]["model_name"],
            crossencoder_size=params["multicrossencoder"]["model_max_length"]
        )

        if args.metric_file is not None:
            with open(args.metric_file, 'w') as f:
                f.write("Training time\t" + str((process_time/1e9)) + " s.")

        print("Done.")

    else:

        tune_with_wandb(
            project='el_multicrossencoder',
            sweep_id=args.sweep_id,
            sweep_method='grid',
            metric_name='val_at_1',
            metric_goal='maximize',
            training_func=train_multicrossencoder,
            fixed_args={
                "train_collection": train_collection,
                "val_collection": val_collection,
                "ontology": ontology,
                "candidates_lookup": candidates_lookup,
                "batch_size": 2,
                "model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                "model_max_length": 512
            },
            tunable_args={
                "top_k": [1, 2, 3, 4, 5],
                "learning_rate": [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5]
            }
        )


if __name__ == '__main__':
    main()

