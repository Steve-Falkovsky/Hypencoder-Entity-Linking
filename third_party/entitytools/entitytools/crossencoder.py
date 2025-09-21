import gzip
import torch
import json
import os
from tqdm.auto import tqdm
from transformers import set_seed, AutoTokenizer, AutoModelForTokenClassification
import random
from more_itertools import chunked
from sklearn.metrics import f1_score
import gc
import math
from scipy.special import softmax
from collections import defaultdict

from entitytools.sentences import mark_sentences, get_sentence_annotations
from entitytools.utils import add_links_to_bioc_annotations

__author__ = "Javier Sanz-Cruzado, Jake Lever"
__license__ = "MIT License"
__version__ = 1.0


def create_basedataset(collection, candidates_lookup, ontology, tokenizer, max_length, top_k=5, no_labels=False):
    """
    Creates the dataset for the cross-encoder to process.
    :param collection: the initial collection.
    :param candidates_lookup: the lookup from a first-stage entity linker matching annotations with candidates.
    :param ontology: the ontology to match against.
    :param tokenizer: the chosen tokenizer.
    :param max_length: maximum length of a token sequence.
    :param top_k: number of candidates to choose from the first-stage candidate retrieval model.
    :param no_labels: ignore labels if true.
    :return: the dataset.
    """
    single_dataset = []
    multi_dataset = []

    count_less_than_5 = 0
    for doc in tqdm(collection.documents):
        for passage in doc.passages:
            for sentence in passage.sentences:
                sentence_start = sentence.offset
                sentence_end = sentence.offset + int(sentence.infons['length'])

                sentence_text = passage.text[(sentence_start - passage.offset):(sentence_end - passage.offset)]
                sentence_annotations = get_sentence_annotations(passage, sentence)

                multi_labels, multi_annos = [], []
                multi_text = sentence_text
                for anno in sentence_annotations:
                    candidates = candidates_lookup.get(anno.text.lower(), [])[:top_k]
                    for onto_rank, onto_idx in enumerate(candidates):
                        entity = ontology[onto_idx]

                        if no_labels:
                            label = 0
                        else:
                            label = int(anno.infons['concept_id'] == entity['id'])

                        single_text = f"{sentence_text}{tokenizer.sep_token}{anno.text}{tokenizer.mask_token}{entity['name']}"
                        tk = tokenizer.tokenize(single_text)
                        if len(tk) >= max_length:
                            continue
                        single_dataset.append(([label], single_text, [(anno, onto_rank, onto_idx)]))
    return single_dataset


def rank_evaluate(collection, annoinfo_and_scores, ontology, prefix='', k=5):
    """
    Evaluate  the outcome of a cross-encoder model.
    :param collection: the collection of models.
    :param annoinfo_and_scores: annotation information and scores for the different mention-entity pairs.
    :param ontology: the ontology
    :param prefix: prefix of the metric names.
    :param k: the maximum number of ranking elements to be considered.
    :return: a dictionary containing the metric values.
    """
    ontology_id_to_idx = {e['id']: idx for idx, e in enumerate(ontology)}

    score_lookup = defaultdict(list)
    for (anno, onto_rank, onto_idx), score in annoinfo_and_scores:
        score_lookup[anno].append((onto_rank, score, onto_idx))

    first_at_1, first_at_5, first_at_10, reranked_at_1, reranked_at_5, reranked_at_10, anno_count = 0, 0, 0, 0, 0, 0, 0
    for doc in collection.documents:
        for passage in doc.passages:
            for anno in passage.annotations:
                retrieved = score_lookup.get(anno, [])

                first_stage = [onto_idx for onto_rank, score, onto_idx in sorted(retrieved, key=lambda x: x[0])]
                reranked = [onto_idx for onto_rank, score, onto_idx in
                            sorted(retrieved, key=lambda x: x[1], reverse=True)]

                correct_onto_idx = ontology_id_to_idx[anno.infons['concept_id']]

                first_at_1 += (correct_onto_idx in first_stage[:1])
                first_at_5 += (correct_onto_idx in first_stage[:5])
                first_at_10 += (correct_onto_idx in first_stage[:10])
                reranked_at_1 += (correct_onto_idx in reranked[:1])
                reranked_at_5 += (correct_onto_idx in reranked[:5])
                reranked_at_10 += (correct_onto_idx in reranked[:10])
                anno_count += 1

    return {
        f'{prefix}_firststage_@1': first_at_1 / anno_count,
        f'{prefix}_firststage_@5': first_at_5 / anno_count,
        f'{prefix}_firststage_@10': first_at_10 / anno_count,
        f'{prefix}_reranked_@1': reranked_at_1 / anno_count,
        f'{prefix}_reranked_@5': reranked_at_5 / anno_count,
        f'{prefix}_reranked_@10': reranked_at_10 / anno_count,
    }


def get_mask_logits(tokenizer, model, mask_token_id, batch, max_length):
    """
    Obtains the logit values for the [MASK] tokens.
    :param tokenizer: the tokenizer.
    :param model: the cross-encoder model.
    :param mask_token_id: the token id of the [MASK] token.
    :param batch: the batch to process.
    :param max_length: the maximum length of the context window.
    :return: the logit values for the [MASK] tokens in the batch.
    """
    tokenized = tokenizer([text for labels, text, coords in batch], return_tensors='pt', truncation=True,
                          max_length=max_length, padding=True)

    model_output = model(**tokenized.to(model.device))
    mask_token_locs = (tokenized['input_ids'] == mask_token_id).nonzero()

    mask_logits = model_output.logits[mask_token_locs[:, 0], mask_token_locs[:, 1]]

    return mask_logits


# based on: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    """
    Class for defining the early stopping. This early stopping strategy decides to stop the training if a particular
    metric has not improved by a certain percentage after a number of iterations.
    """

    def __init__(self, patience=1, smaller_is_better=True, required_perc_change=1):
        """
        Initializes the early stopping.
        :param patience: the maximum number of steps without enough improvement.
        :param smaller_is_better: True if a smaller value of the metric is better.
        :param required_perc_change: the required improvement percentage over the previous base value.
        """
        self.patience = patience
        self.smaller_is_better = smaller_is_better
        self.counter = 0
        self.best_metric = float('inf') if smaller_is_better else float('-inf')
        self.required_perc_change = required_perc_change

    def early_stop(self, metric):
        """
        Determines whether the training should be stopped.
        :param metric: the metric to use by the early stopping strategy.
        :return: True if the training should be stopped, False otherwise.
        """
        if self.smaller_is_better:
            required_improvement = (1 - self.required_perc_change / 100) * self.best_metric
            print(f"{metric=} {self.best_metric=} {required_improvement=} {metric < required_improvement}")
            if metric < required_improvement:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
        else:
            required_improvement = (1 + self.required_perc_change / 100) * self.best_metric
            print(f"{metric=} {self.best_metric=} {required_improvement=} {metric > required_improvement}")
            if metric > required_improvement:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1

        return self.counter >= self.patience


def apply_basecrossencoder(
        collection,
        ontology,
        candidates_lookup,
        top_k,
        crossencoder_model,
        crossencoder_size
):
    """
    Applies the base cross-encoder to generate mention-candidate scores.
    :param collection: the collection to which apply the entity linking.
    :param ontology: the ontology
    :param candidates_lookup: first-phase lookup to obtain the candidates for a mention.
    :param top_k: the number of candidate entities to consider for each mention.
    :param crossencoder_model: the cross-encoder model to use.
    :param crossencoder_size: the context window length of the cross-encoder.
    :return: the time needed to apply the cross-encoder
    """
    sentences_are_marked = len(collection.documents[0].passages[0].sentences) > 0
    if not sentences_are_marked:
        mark_sentences(collection)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForTokenClassification.from_pretrained(crossencoder_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(crossencoder_model)
    mask_token_id = tokenizer.vocab[tokenizer.mask_token]
    data_multi = create_basedataset(collection, candidates_lookup, ontology, tokenizer, crossencoder_size,
                                    top_k=top_k, no_labels=True)

    batch_size = 2
    start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
    annoinfo_and_scores = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(list(chunked(data_multi, batch_size))):
            logits = get_mask_logits(tokenizer, model, mask_token_id, batch, crossencoder_size)
            scores = softmax(logits.cpu().numpy(), axis=1)[:, 1].tolist()

            flattened_annoinfo = [annoinfo for labels, text, annoinfos in batch for annoinfo in annoinfos]
            assert logits.shape[0] == len(flattened_annoinfo)

            annoinfo_and_scores += list(zip(flattened_annoinfo, scores))

    annos_to_scored_links = defaultdict(list)
    for (anno, onto_rank, onto_idx), score in annoinfo_and_scores:
        annos_to_scored_links[anno].append((score, onto_idx))
    end_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
    print(len(annos_to_scored_links))

    add_links_to_bioc_annotations(collection, annos_to_scored_links, ontology)
    return end_time - start_time


def train_basecrossencoder(
        train_collection,
        val_collection,
        ontology,
        candidates_lookup,
        top_k,
        learning_rate,
        batch_size,
        wandb_logger=None,
        output_dir=None,
        crossencoder_model=None,
        crossencoder_size=None
):
    """
        Trains the base cross-encoder to generate mention-candidate scores.
        :param train_collection: the data collection to train the cross-encoder.
        :param val_collection: the validation data to test the cross-encoder.
        :param ontology: the ontology
        :param candidates_lookup: first-phase lookup to obtain the candidates for a mention.
        :param top_k: the number of candidate entities to consider for each mention.
        :param learning_rate: the learning rate of the cross-encoder
        :param batch_size: the batch size for selecting training examples
        :param wandb_logger: if any, the logger for Weights and Biases.
        :param output_dir: the directory on which to store the trained model.
        :param crossencoder_model: the name of the Transformers model.
        :param crossencoder_size: the length of the context window of the model.
        :return: the time needed to train the cross-encoder
    """
    gc.collect()
    torch.cuda.empty_cache()

    set_seed(42)
    random.seed(42)

    max_epochs = 16

    print("Marking sentences (if needed)...")
    train_sentences_are_marked = len(train_collection.documents[0].passages[0].sentences) > 0
    if not train_sentences_are_marked:
        mark_sentences(train_collection)
    val_sentences_are_marked = len(val_collection.documents[0].passages[0].sentences) > 0
    if not val_sentences_are_marked:
        mark_sentences(val_collection)

    print("Creating datasets ready for multiencoder")

    MODEL_NAME = crossencoder_model if crossencoder_model is not None else 'answerdotai/ModernBERT-base'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    size = crossencoder_size if crossencoder_size is not None else 8192

    train_data = create_basedataset(train_collection, candidates_lookup, ontology, tokenizer, size, top_k=top_k)
    val_data = create_basedataset(val_collection, candidates_lookup, ontology, tokenizer, size, top_k=top_k)

    mask_token_id = tokenizer.vocab[tokenizer.mask_token]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    early_stopper = EarlyStopper(patience=3, smaller_is_better=False)

    start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
    for epoch in range(max_epochs):
        random.shuffle(train_data)

        train_loss = 0
        train_labels, train_annoinfo_and_scores, train_predicted = [], [], []
        model.train()
        for batch in tqdm(list(chunked(train_data, batch_size))):
            logits = get_mask_logits(tokenizer, model, mask_token_id, batch, size)
            scores = softmax(logits.detach().cpu().numpy(), axis=1)[:, 1].tolist()

            flattened_labels = [label for labels, text, annoinfos in batch for label in labels]
            flattened_annoinfo = [annoinfo for labels, text, annoinfos in batch for annoinfo in annoinfos]
            assert logits.shape[0] == len(flattened_labels)
            assert logits.shape[0] == len(flattened_annoinfo)

            loss = loss_func(logits, torch.tensor(flattened_labels, device=device, dtype=torch.long))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_predicted += logits.argmax(axis=1).tolist()
            train_annoinfo_and_scores += list(zip(flattened_annoinfo, scores))
            train_labels += flattened_labels

        train_loss /= len(train_data)
        train_f1 = f1_score(train_labels, train_predicted, average='macro', zero_division=0.0)

        val_loss = 0
        val_labels, val_annoinfo_and_scores, val_predicted = [], [], []
        model.eval()

        with torch.no_grad():
            for batch in tqdm(list(chunked(val_data, batch_size))):
                logits = get_mask_logits(tokenizer, model, mask_token_id, batch, crossencoder_size)
                scores = softmax(logits.cpu().numpy(), axis=1)[:, 1].tolist()

                flattened_labels = [label for labels, text, annoinfos in batch for label in labels]
                flattened_annoinfo = [annoinfo for labels, text, annoinfos in batch for annoinfo in annoinfos]
                assert logits.shape[0] == len(flattened_labels)
                assert logits.shape[0] == len(flattened_annoinfo)

                loss = loss_func(logits, torch.tensor(flattened_labels, device=device, dtype=torch.long))

                val_loss += loss.item()
                val_predicted += logits.argmax(axis=1).tolist()
                val_annoinfo_and_scores += list(zip(flattened_annoinfo, scores))
                val_labels += flattened_labels

        val_loss /= len(val_data)
        val_f1 = f1_score(val_labels, val_predicted, average='macro', zero_division=0.0)

        train_rank_metrics = rank_evaluate(train_collection, train_annoinfo_and_scores, ontology, prefix='train')
        val_rank_metrics = rank_evaluate(val_collection, val_annoinfo_and_scores, ontology, prefix='val')

        metrics_to_log = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'train_f1': train_f1,
                          'val_f1': val_f1}
        metrics_to_log.update(train_rank_metrics)
        metrics_to_log.update(val_rank_metrics)
        print(f"{epoch=} {train_loss=:.3f} {val_loss=:.3f} {train_f1=:.3f} {val_f1=:.3f}")

        print("Reporting metrics...")
        print(f'{metrics_to_log=}')
        if wandb_logger is not None:
            wandb_logger.log(metrics_to_log)

        if early_stopper.early_stop(val_f1):
            print("Ending training due to early stopping criterion...")
            break

        if math.isnan(train_loss) or math.isnan(val_loss):
            print("Ending training as loss is NaN...")
            break
    end_time = time.clock_gettime_ns(time.CLOCK_REALTIME)
    if output_dir is not None:
        print("Saving model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    del tokenizer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return end_time - start_time
