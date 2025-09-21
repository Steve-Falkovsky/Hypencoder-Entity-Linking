import argparse
from entitytools.ontology import load_ontology
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import time

from entitytools.dense_vectors import make_dense_vectors

def main():
    parser = argparse.ArgumentParser('Apply a transformer model to ontology term names and save out the vectors')
    parser.add_argument('--ontology',required=True,type=str,help='GZipped ontology file')
    parser.add_argument('--model_name',required=True,type=str,help='Transformer model to use')
    parser.add_argument('--out_vectors',required=True,type=str,help='A Numpy array with the dense vectors')
    parser.add_argument('--metric_file', required=False, type=str, help='File to print metrics')
    args = parser.parse_args()

    print("Loading ontology...")
    ontology = load_ontology(args.ontology)
    onto_texts = [ e['name'].lower() for e in ontology ]
    print(f"Loaded {len(onto_texts)} terms")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device=}")

    print(f"Loading {args.model_name} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    print("Creating dense vectors of ontology names...")
    start_time = time.clock_gettime_ns()
    onto_vectors = make_dense_vectors(model, tokenizer, onto_texts)
    end_time = time.clock_gettime_ns()
    print(f"Got {onto_vectors.shape=}")

    print("Saving...")
    np.save(args.out_vectors, onto_vectors)

    if args.metric_file is not None:
        with open(args.metric_file, 'w') as f:
            f.write("Training time\t" + str(((end_time - start_time)/ 1e9)) + " s.")

    print("Done.")



if __name__ == '__main__':
    main()
