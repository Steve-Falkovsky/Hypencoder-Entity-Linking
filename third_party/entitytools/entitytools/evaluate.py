from entitytools.utils import collection_to_docs_passages_and_annos
import pandas as pd


def evaluate(gold_collection, predicted_collection, verbose=True):
    """
    Evaluates a collection
    :param gold_collection: the labelled collection to be used as the ground-truth.
    :param predicted_collection: the collection labelled by entity linking.
    :param verbose: True to print the results.
    :return: a DataFrame with the metric values.
    """
    gold_docs_passages_annos = collection_to_docs_passages_and_annos(gold_collection)
    pred_docs_passages_annos = collection_to_docs_passages_and_annos(predicted_collection)

    assert len(gold_docs_passages_annos) == len(pred_docs_passages_annos), "Mismatch in number of annotations"

    val_list = []

    at_1, at_5, at_10, at_all = 0, 0, 0, 0
    i = 0
    for (_, _, gold_anno), (_, _, pred_anno) in zip(gold_docs_passages_annos, pred_docs_passages_annos):
        assert gold_anno.text == pred_anno.text, "Mismatch in comparing annotations"

        val = []
        val.append(i)
        correct_concept_id = gold_anno.infons['concept_id']

        all_links = pred_anno.infons.get('all_links', '')

        ids_and_scores = [id_and_score.split('=') for id_and_score in all_links.split('|') if id_and_score]
        ids_and_scores = [(identifier, float(score)) for identifier, score in ids_and_scores]
        ids_and_scores = sorted(ids_and_scores, key=lambda x: x[1], reverse=True)  # Do a sort just in case

        ranked_ids = [identifier for identifier, score in ids_and_scores]

        if correct_concept_id in ranked_ids[:1]:
            at_1 += 1
            val.append(1)
        else:
            val.append(0)

        if correct_concept_id in ranked_ids[:5]:
            at_5 += 1
            val.append(1)
        else:
            val.append(0)

        if correct_concept_id in ranked_ids[:10]:
            at_10 += 1
            val.append(1)
        else:
            val.append(0)
        if correct_concept_id in ranked_ids:
            at_all += 1
            val.append(1)
        else:
            val.append(0)
        i += 1
        val_list.append(val)

    at_1 /= len(gold_docs_passages_annos)
    at_5 /= len(gold_docs_passages_annos)
    at_10 /= len(gold_docs_passages_annos)
    at_all /= len(gold_docs_passages_annos)

    if verbose:
        print(f"{at_1=:.4f}")
        print(f"{at_5=:.4f}")
        print(f"{at_10=:.4f}")
        print(f"{at_all=:.4f}")

    return pd.DataFrame(val_list, columns=["id", "at_1", "at_5", "at_10", "at_all"])