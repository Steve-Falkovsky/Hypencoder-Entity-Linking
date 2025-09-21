from tqdm import tqdm
import shutil
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import socket
import pickle
from collections import defaultdict
import gzip
import json
import time
from pyterrier_pisa import PisaIndex

from entitytools.utils import collection_to_docs_passages_and_annos, to_mentions, add_links_to_bioc_annotations


def make_dict(sparse_matrix):
    for doc_id, row in enumerate(tqdm(sparse_matrix, total=sparse_matrix.shape[0])):
        _, idxs = row.nonzero()
        toks = {str(idx): row[0, idx] for idx in idxs}
        yield {'docno': str(doc_id), 'toks': toks}


def make_query(sparse_matrix, offset):
    for doc_id, row in enumerate(sparse_matrix):
        _, idxs = row.nonzero()
        toks = {str(idx): row[0, idx] for idx in idxs}
        yield {'qid': str(doc_id + offset), 'query_toks': toks}


from scipy.sparse import hstack


class NGramsVectorizer:
    """
    Class for vectorizing n-grams
    """
    def __init__(self, char_side, char_count, ngram_range, max_features, min_df):
        """
        Initializes the n-grams vectorizing.
        :param char_side: the side of the string to consider.
        :param char_count: the number of characters to consider.
        :param ngram_range: the range of n-grams to consider.
        :param max_features: the maximum number of values.
        :param min_df: the minimum document frequency of a n-gram.
        """
        self.char_side = char_side
        self.char_count = char_count

        self.vectorizer1 = TfidfVectorizer(analyzer='char', ngram_range=tuple(ngram_range), max_features=max_features,
                                           norm='l2', lowercase=True, sublinear_tf=False, min_df=min_df)

        if self.char_side == 'separate':
            self.vectorizer2 = TfidfVectorizer(analyzer='char', ngram_range=tuple(ngram_range),
                                               max_features=max_features, norm='l2', lowercase=True, sublinear_tf=False,
                                               min_df=min_df)

    def transform_texts(self, texts):
        """
        Given a text, processes them before the n-gram transformation.
        :param texts: the original texts.
        :return: a pair of texts (texts2 shall only be different than None if char_side considers both sides of the text.
        """
        texts2 = None
        if self.char_side == 'none':
            texts1 = texts
        elif self.char_side == 'front':
            texts1 = [t[:self.char_count] for t in texts]
        elif self.char_side == 'back':
            texts1 = [t[-self.char_count:] for t in texts]
        elif self.char_side in ['separate', 'together']:
            half_count = self.char_count // 2
            texts1 = [t[:half_count] for t in texts]
            texts2 = [t[-half_count:] for t in texts]

        return texts1, texts2

    def fit(self, texts):
        """
        Fits the vectorizer following the texts.
        :param texts: the texts to vectorize.
        """
        texts1, texts2 = self.transform_texts(texts)

        if self.char_side == 'separate':
            self.vectorizer1.fit(texts1)
            self.vectorizer2.fit(texts2)
        elif self.char_side == 'together':
            self.vectorizer1.fit(texts1 + texts2)
        else:
            self.vectorizer1.fit(texts1)

    def transform(self, texts):
        """
        Transforms texts into vectors.
        :param texts: the texts to transform.
        :return: the transformed texts.
        """
        texts1, texts2 = self.transform_texts(texts)

        X = self.vectorizer1.transform(texts1)
        if self.char_side == 'separate':
            X2 = self.vectorizer2.transform(texts2)
            X = hstack([X, X2])
        elif self.char_side == 'together':
            X2 = self.vectorizer1.transform(texts2)
            X += X2
        return X

    def fit_transform(self, texts):
        """
        Fits and transforms the texts.
        :param texts: the text.
        :return: the vectorized texts.
        """
        self.fit(texts)
        return self.transform(texts)


def train_ngrams(
        train_collection,
        val_collection,
        ontology,
        ngram_range,
        max_features,
        min_df,
        add_training_data_to_ontology,
        char_side,
        char_count,
        output_dir=None,
        wandb_logger=None):
    """
    Trains an n-grams model.
    :param train_collection: the data collection to train the cross-encoder.
    :param val_collection: the validation data to test the cross-encoder.
    :param ontology: the ontology
    :param ngram_range: the range of n-grams to consider.
    :param max_features: the maximum number of different n-grams to consider.
    :param min_df: the minimum number of documents on which an n-gram needs to appear.
    :param add_training_data_to_ontology: True if we want to add the training aliases to the ontology.
    :param char_side: the side of the mention to consider for collecting n-grams.
    :param char_count: the number of characters in the mention to consider for collecting n-grams.
    :param output_dir: output directory on which to store the n-grams model.
    :param wandb_logger: if any, a Weights and Biases logger.
    :return: the time needed to train the n-grams model.
    """

    assert char_side in ['none', 'front', 'back', 'separate', 'together']

    if output_dir is not None and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_docs_passages_and_annos = collection_to_docs_passages_and_annos(train_collection)
    val_docs_passages_and_annos = collection_to_docs_passages_and_annos(val_collection)
    print(f"{len(train_docs_passages_and_annos)=} {len(val_docs_passages_and_annos)=}")

    train_mentions = to_mentions(train_docs_passages_and_annos)
    val_mentions = to_mentions(val_docs_passages_and_annos)

    if add_training_data_to_ontology:
        print("Adding training aliases to ontology")
        ontology_by_id = {entity['id']: entity for entity in ontology}
        for text, start, end, concept_id in train_mentions:
            mention_text = text[start:end]
            e = ontology_by_id[concept_id]
            e['aliases'].append(mention_text)
            e['aliases'] = sorted(set(e['aliases']))

    ontology_id_to_idx = {e['id']: idx for idx, e in enumerate(ontology)}

    print("Preparing text of ontology terms...")
    onto_docs = []
    docno = 0
    for idx, e in enumerate(ontology):
        aliases = [e['name']] + e['aliases']
        aliases = [a.lower() for a in aliases]
        for alias in sorted(set(aliases)):
            onto_docs.append({'docno': str(docno), 'text': alias, 'onto_idx': idx})
            docno += 1

    start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
    print("Vectorizing ontology terms...")
    # vectorizer = TfidfVectorizer(analyzer='char', ngram_range=tuple(ngram_range), max_features=max_features, norm='l2', lowercase=True, sublinear_tf=False, min_df=min_df)
    vectorizer = NGramsVectorizer(char_side, char_count, ngram_range, max_features, min_df)
    onto_vecs = vectorizer.fit_transform([x['text'] for x in onto_docs])

    if output_dir is not None:
        print("Saving vectorizer...")
        with open(f'{output_dir}/vectorizer.pickle', 'wb') as f:
            pickle.dump(vectorizer, f)

    print("Initialising index...")
    if output_dir is not None:
        index_dir = f'{output_dir}/index.pisa'
    else:
        unique_info = f'{socket.gethostname()}_{os.getpid()}'
        index_dir = f'index_{unique_info}.pisa'

    if os.path.isdir(index_dir):
        shutil.rmtree(index_dir)

    index = PisaIndex(index_dir, stemmer='none', stops='none')

    print("Creating ontology index mapping...")
    docno_onto_mapping = pd.DataFrame(onto_docs).drop('text', axis=1)
    docno_onto_mapping['onto_idx'] = docno_onto_mapping['onto_idx'].astype(int)
    docno_onto_mapping['docno'] = docno_onto_mapping['docno'].astype(int)
    if output_dir is not None:
        print("Saving ontology index mapping...")
        docno_onto_mapping.to_pickle(f'{output_dir}/index_mapping.pickle')

    print("Building index...")
    index.toks_indexer().index(make_dict(
        onto_vecs))  # it doesn't need to be a list, you can also use an iterator here to avoid loading it all into memory

    top_k = 10
    retr = index.quantized(num_results=top_k)

    combined_texts = sorted(
        set(text[start:end].lower() for text, start, end, concept_id in train_mentions + val_mentions))
    ngrams_lookup = make_ngrams_lookup(combined_texts, vectorizer, retr, docno_onto_mapping, top_k)

    end_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
    if output_dir is not None:
        with gzip.open(f'{output_dir}/lookup.json.gz', 'wt') as f:
            json.dump(ngrams_lookup, f)

    metrics_to_log = {}
    for split, mentions in [('train', train_mentions), ('val', val_mentions)]:
        print(f"Evaluating {split} split")
        at_1, at_5, at_10, at_all = 0, 0, 0, 0
        for text, start, end, concept_id in mentions:
            correct_onto_idx = ontology_id_to_idx[concept_id]
            retrieved = ngrams_lookup.get(text[start:end].lower(), [])

            at_1 += (correct_onto_idx in retrieved[:1])
            at_5 += (correct_onto_idx in retrieved[:5])
            at_10 += (correct_onto_idx in retrieved[:10])
            at_all += (correct_onto_idx in retrieved)

        at_1 /= len(mentions)
        at_5 /= len(mentions)
        at_10 /= len(mentions)
        at_all /= len(mentions)

        metrics_to_log.update({
            f'{split}_at_1': at_1,
            f'{split}_at_5': at_5,
            f'{split}_at_10': at_10,
            f'{split}_at_all': at_all,
        })

        print(f"\t{at_1=:.3f} {at_5=:.3f} {at_10=:.3f} {at_all=:.3f}")

    print("Reporting metrics...")
    print(f'{metrics_to_log=}')
    if wandb_logger is not None:
        wandb_logger.log(metrics_to_log)

    if output_dir is None:
        print("Cleaning up...")
        del retr
        del index
        if os.path.isdir(index_dir):
            shutil.rmtree(index_dir)

    return end_time - start_time


def get_unique_annotation_texts(collection, lowercase, min_length=0):
    if lowercase:
        anno_texts = set(
            anno.text.lower() for document in collection.documents for passage in document.passages for anno in
            passage.annotations if len(anno.text) >= min_length)
    else:
        anno_texts = set(anno.text for document in collection.documents for passage in document.passages for anno in
                         passage.annotations if len(anno.text) >= min_length)

    return sorted(anno_texts)


def load_and_make_ngrams_lookup(anno_texts, ngrams_model, top_k):
    anno_texts = set(anno_texts)

    print("Loading vectorizer...")
    with open(f'{ngrams_model}/vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)

    print("Loading ngrams index...")
    index = PisaIndex(f'{ngrams_model}/index.pisa', stemmer='none', stops='none')
    retr = index.quantized(num_results=top_k)

    docno_onto_mapping = pd.read_pickle(f'{ngrams_model}/index_mapping.pickle')

    with gzip.open(f'{ngrams_model}/lookup.json.gz', 'rt') as f:
        existing_lookup = json.load(f)
    existing_lookup = {k: v for k, v in existing_lookup.items() if k in anno_texts}
    print(f"Got lookup with {len(existing_lookup)} terms")

    novel_texts = set(anno_texts).difference(existing_lookup.keys())
    print(f"Need to get lookup for {len(novel_texts)} terms")

    if len(novel_texts) > 0:
        novel_lookup = make_ngrams_lookup(sorted(novel_texts), vectorizer, retr, docno_onto_mapping, top_k)
    else:
        novel_lookup = {}

    combined_lookup = {**existing_lookup, **novel_lookup}
    print(f"Created combined lookup for {len(combined_lookup)} terms")

    return combined_lookup


def make_ngrams_lookup(mention_texts, vectorizer, retr, docno_onto_mapping, top_k):
    print("Vectorizing mentions...")
    query_vecs = vectorizer.transform(mention_texts)

    print("Searching index...")
    batch_size = 1000
    retrieved = []
    for batch_offset in tqdm(range(0, query_vecs.shape[0], batch_size)):
        batch = query_vecs[batch_offset:(batch_offset + batch_size), :]
        retrieved.append(pd.DataFrame(retr(make_query(batch, offset=batch_offset))))
    retrieved = pd.concat(retrieved, axis=0, ignore_index=True)

    lookup_by_mention_idx = defaultdict(list)
    if len(retrieved) > 0:
        print("Tidying up results...")
        retrieved['qid'] = retrieved['qid'].astype(int)
        retrieved['docno'] = retrieved['docno'].astype(int)
        retrieved = retrieved[retrieved['rank'] <= top_k]
        retrieved.rename(columns={"qid": "mention_idx"}, inplace=True)
        retrieved.sort_values(by=['mention_idx', 'rank'], inplace=True)

        print("Mapping retrieved to onto indices...")
        retrieved = pd.merge(retrieved, docno_onto_mapping, on=['docno'], how='inner')

        print("Grouping results for output...")
        for _, row in tqdm(retrieved.iterrows(), total=len(retrieved)):
            mention_idx = int(row['mention_idx'])
            onto_idx = int(row['onto_idx'])

            lookup_by_mention_idx[mention_idx].append(onto_idx)

    lookup_by_mention_text = {m.lower(): [] for m in mention_texts}
    for mention_idx, onto_idxs in lookup_by_mention_idx.items():
        unique_onto_idxs_same_order = list(dict.fromkeys(onto_idxs))
        lookup_by_mention_text[mention_texts[mention_idx].lower()] = unique_onto_idxs_same_order

    return lookup_by_mention_text


def apply_ngrams(collection, ontology, candidates_lookup):
    """
    Applies the n-grams model.
    :param collection: the collection to apply the model to.
    :param ontology: the ontology.
    :param candidates_lookup: a candidates lookup.
    :return: the time needed to apply the model.
    """
    start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
    all_candidates = []
    annos_to_scored_links = {}
    for doc in collection.documents:
        for passage in doc.passages:
            for anno in passage.annotations:
                candidates = candidates_lookup.get(anno.text.lower(), [])[:5]
                all_candidates.extend(candidates)
                scores_and_onto_idxs = [(1 - i / len(candidates), onto_idx) for i, onto_idx in enumerate(candidates)]

                annos_to_scored_links[anno] = scores_and_onto_idxs
    end_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
    add_links_to_bioc_annotations(collection, annos_to_scored_links, ontology)
    return end_time - start_time