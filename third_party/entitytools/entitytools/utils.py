import re
import gzip
import psutil
from more_itertools import chunked
from tqdm.auto import tqdm


def collection_to_docs_passages_and_annos(collection):
    """
    Transforms a collection to a list of document / passage / annotation triplets.
    :param collection: the collection of documents.
    :return: the list of document / passage / annotation triplets.
    """
    return [(document, passage, anno) for document in collection.documents for passage in document.passages for anno in
            passage.annotations]


def to_mention_parts(passage, anno):
    """
    Given a passage and an annotation, indicates (a) the text of the passage, (b) the start of the annotation,
    (c) the end of the annotation, and (d) the concept referred in the annotation.
    :param passage: the passage.
    :param anno: the annotation.
    :return: a tuple containing the elements mentioned before.
    """
    assert len(anno.locations) == 1
    start = anno.locations[0].offset - passage.offset
    end = start + anno.locations[0].length
    concept_id = anno.infons.get('concept_id')
    return passage.text, start, end, concept_id


def to_mentions(docs_passages_and_annos):
    """
    Transforms a list of document/passage/annotation triplets to a list of mentions.
    :param docs_passages_and_annos: the list of document/passage/annotation triplets.
    :return: the list of mentions (see to_mention_parts)
    """
    annos = []
    for doc, passage, anno in docs_passages_and_annos:
        annos.append(to_mention_parts(passage, anno))

    return annos


def create_windowed_text(text, start, end, window=100):
    """
    Creates a window of text around an annotated mention.
    :param text: the text.
    :param start: start of the mention.
    :param end: end of the mention.
    :param window: length of the window.
    :return: the window around the text (text, start, end triplet)
    """
    # Create a nice window around the annotated mention
    word_boundaries = [m.start() for m in re.finditer(r'\b', text)]
    earlier_boundaries = [w for w in word_boundaries if w < (start - window)]
    earlier = earlier_boundaries[-1] if earlier_boundaries else 0
    later_boundaries = [w for w in word_boundaries if w > (end + window)]
    later = later_boundaries[0] if later_boundaries else len(text)

    text = text[earlier:later]
    start, end = start - earlier, end - earlier
    return text, start, end


def get_process_memory():
    """
    Obtains the memory consumed by the process.
    :return: the memory consumed by the process.
    """
    process = psutil.Process()
    mb = process.memory_info().rss / (1024 * 1024)
    return mb


def get_annotation_texts(collections):

    annotation_texts = set()

    for collection in collections:
        for doc in collection.documents:
            for passage in doc.passages:
                for anno in passage.annotations:
                    annotation_texts.add(anno.text)

    return annotation_texts


def ceiling_division(n, d):
    return -(n // -d)


def tqdm_chunked(l, n):
    return tqdm(chunked(l, n), total=ceiling_division(len(l), n))


def threshold_entities(collection, threshold):
    for doc in collection.documents:
        for passage in doc.passages:
            passage.annotations = [anno for anno in passage.annotations if
                                   'concept_score' in anno.infons and float(anno.infons['concept_score']) > threshold]


def add_links_to_bioc_annotations(collection, annos_to_scored_links, ontology):
    # Clean any existing metadata first (and set concept_id to be none by default)
    for doc in collection.documents:
        for passage in doc.passages:
            for anno in passage.annotations:
                anno.infons = {'concept_id': 'none'}

    counter = 0
    pos_counter = 0
    for anno, scores_and_onto_idxs in annos_to_scored_links.items():
        counter += 1
        if len(scores_and_onto_idxs) > 0:
            pos_counter += 1
            reranked = sorted(scores_and_onto_idxs, key=lambda x: x[0], reverse=True)

            top_score, top_onto_idx = reranked[0]
            e = ontology[top_onto_idx]

            anno.infons['concept_id'] = e['id']
            if 'tags' in e:
                anno.infons['tags'] = "|".join(e['tags'])
            anno.infons['concept_score'] = f"{top_score:.4f}"

            all_links = "|".join(f"{ontology[onto_idx]['id']}={score:.4f}" for score, onto_idx in reranked)
            anno.infons['all_links'] = all_links
    print(counter)
    print(pos_counter)
