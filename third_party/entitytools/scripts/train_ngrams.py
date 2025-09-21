import gzip
from bioc import biocxml
import json
import argparse  
import pprint

from entitytools.ontology import load_ontology
from entitytools.ngrams import train_ngrams
from entitytools.wandb import tune_with_wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,type=str,help='Whether to train a single model or hyperparameter tune (train/tune)')
    parser.add_argument('--train_corpus',required=True,type=str,help='Gzipped BioC XML with training corpus')
    parser.add_argument('--val_corpus',required=True,type=str,help='Gzipped BioC XML with validation corpus')
    parser.add_argument('--ontology',required=True,type=str,help='Gzipped TSV or JSON with ontology information')
    parser.add_argument('--sweep_id',required=False,type=str,help='sweep_id for joining wandb sweep')
    parser.add_argument('--params',required=False,type=str,help='JSON file with parameters to use for saving out model')
    parser.add_argument('--output_dir',required=False,type=str,help='Directory to save model')
    parser.add_argument('--debug',action='store_true',help='Run a tiny subset of data to check things')
    parser.add_argument('--metric_file', required=False, default=None, type=str, help='File to store times')
    args = parser.parse_args()

    assert args.mode in ['train','tune'], "--mode must be train or tune"

    with gzip.open(args.train_corpus,'rt',encoding='utf8') as f:
        train_collection = biocxml.load(f)
    with gzip.open(args.val_corpus,'rt',encoding='utf8') as f:
        val_collection = biocxml.load(f)

    
    print(f"Loaded {len(train_collection.documents)} training documents")
    print(f"Loaded {len(val_collection.documents)} validation documents")

    ontology = load_ontology(args.ontology)

    if args.debug:
        print("Trimming down documents and ontology for quick testing")
        train_collection.documents = train_collection.documents[:1]
        val_collection.documents = val_collection.documents[:1]
        
        print(f"{len(ontology)=}")
        concept_ids = set( anno.infons['concept_id'] for collection in [train_collection,val_collection] for doc in collection.documents for passage in doc.passages for anno in passage.annotations )
        print(f"{len(concept_ids)=}")
        ontology = [ e for e in ontology if e['id'] in concept_ids ]
        print(f"{len(ontology)=}")
    
    if args.mode == 'train':
        assert args.params, "Must provide a parameter file with --params"
        assert args.output_dir, "Must provide an output directory with --output_dir"

        with open(args.params) as f:
            params = json.load(f)

        print("Using parameters:")
        pprint.pprint(params["ngrams"])
        
        process_time = train_ngrams(
            train_collection, 
            val_collection, 
            ontology,
            ngram_range=params["ngrams"]["ngram_range"],
            max_features=params["ngrams"]["max_features"],
            min_df=params["ngrams"]["min_df"],
            output_dir=args.output_dir,
            add_training_data_to_ontology=True,
            char_side=params["ngrams"]["char_side"],
            char_count=params["ngrams"]["char_count"]
        )

        if args.metric_file is not None:
            with open(args.metric_file, 'w') as f:
                f.write("Training time\t" + str((process_time / 1e9)) + " s.")

        print("Done.")

    else:

        tune_with_wandb(
            project = 'el_ngrams_charside',
            sweep_id = args.sweep_id,
            sweep_method = 'grid',
            metric_name = 'val_at_5',
            metric_goal = 'maximize',
            training_func = train_ngrams,
            fixed_args = {
                "train_collection": train_collection,
                "val_collection": val_collection,
                "ontology": ontology,
                "add_training_data_to_ontology": True
            },
            tunable_args = {
                "ngram_range": [ (3,3) ],
                "max_features": [50000],
                "min_df": [50],
                "char_side": ['front','back','separate','together'],
                "char_count": [4,8,12,16,20,24,28,32]
            }
        )
    
        
    

if __name__ == '__main__':
    main()
    