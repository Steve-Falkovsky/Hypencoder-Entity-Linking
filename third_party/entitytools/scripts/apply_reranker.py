import gzip
from bioc import biocxml
import torch
import json
import argparse
import gc
import time

from entitytools.ontology import load_ontology
from entitytools.ngrams import get_unique_annotation_texts, load_and_make_ngrams_lookup
from entitytools.crossencoder import apply_basecrossencoder
from entitytools.multicrossencoder import apply_multicrossencoder
from entitytools.parallelcrossencoder import apply_parallelcrossencoder
from entitytools.passagecrossencoder import apply_passagecrossencoder
from entitytools.documentcrossencoder import apply_documentcrossencoder
from entitytools.utils import threshold_entities


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_corpus', required=True, type=str, help='Gzipped BioC XML with training corpus')
    parser.add_argument('--out_corpus', required=True, type=str, help='Gzipped BioC XML with validation corpus')
    parser.add_argument('--ontology', required=True, type=str, help='Gzipped TSV or JSON with ontology information')
    parser.add_argument('--candidates_lookup', required=False, type=str,
                        help='Gzipped JSON lookup to use for candidates')
    parser.add_argument('--ngrams_model', required=False, type=str, help='Directory of ngrams model')
    parser.add_argument('--threshold', required=False, type=float, help='Threshold to apply to linking scores')
    parser.add_argument('--top_k', type=int, default=10, help='Top k candidates to score and rerank')
    parser.add_argument('--reranker_model', required=True, type=str, help='Directory of multicrossencoder model')
    parser.add_argument('--crosstype', required=True, help='Cross-encoder type: base, parallel, multi, passage, '
                                                           'document')
    parser.add_argument('--model_size', required=False, type=int, help='The size of the context window')
    parser.add_argument('--metric_file', required=False, default=None, type=str,
                        help='The file on which to store inference time')
    args = parser.parse_args()

    with gzip.open(args.in_corpus, 'rt', encoding='utf8') as f:
        collection = biocxml.load(f)

    ontology = load_ontology(args.ontology)

    assert args.candidates_lookup or args.ngrams_model, "Must provide --lookup or --ngram_model"
    assert args.crosstype in ['base', 'parallel', 'multi', 'passage',
                              'document'], "--crosstype must be base, parallel, multi, passage or document"

    if args.candidates_lookup:
        with gzip.open(args.candidates_lookup, 'rt') as f:
            candidates_lookup = json.load(f)
    else:
        anno_texts = get_unique_annotation_texts(collection, lowercase=True, min_length=3)
        print(f"{len(anno_texts)=}")

        candidates_lookup = load_and_make_ngrams_lookup(anno_texts, args.ngrams_model, args.top_k)
        print(f"{len(candidates_lookup)=}")

    if args.crosstype == 'base':
        func = apply_basecrossencoder
    elif args.crosstype == 'parallel':
        func = apply_parallelcrossencoder
    elif args.crosstype == 'multi':
        func = apply_multicrossencoder
    elif args.crosstype == 'passage':
        func = apply_passagecrossencoder
    else:
        func = apply_documentcrossencoder

    process_time = func(
        collection,
        ontology,
        candidates_lookup,
        5,
        args.reranker_model,
        args.model_size
    )

    if args.metric_file is not None:
        with open(args.metric_file, 'w') as f:
            f.write("Inference time\t" + str(process_time/1e9) + " s.")

    if args.threshold:
        threshold_entities(collection, args.threshold)

    print(f"Saving to {args.out_corpus}...")
    with gzip.open(args.out_corpus, 'wt') as f:
        biocxml.dump(collection, f)


if __name__ == '__main__':
    main()
